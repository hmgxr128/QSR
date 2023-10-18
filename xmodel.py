import timm
import torch
import torch.optim as optim
from train_utils import group_weight, count_correct, is_bn, eval_param_norm, yield_optimizer_state
from distributed_utils import get_flat_tensor_from_tensor_sequence, is_main_process, get_flat_grad_from, reduce_value, set_flat_grad_to, set_flat_tensor_to_tensor_sequence
from torch.optim.lr_scheduler import LinearLR
from torch.cuda.amp import autocast
import torch.nn as nn
import wandb
from grad_scaler import KGradScaler, GradScaleTooLargeError
from localopt import LocalOptimizer
import os
import composer.functional as cf
import numpy as np


class XModel():
    model: nn.Module
    acc_ctr: int = 0
    m: float = 0.
    v: float = 0.
    m1: float = 0.
    v1: float = 0.
    step_ctr: int = 0
    def __init__(self, model, args) -> None:
        self.args = args
        self.model = model
        self.warmup = args.warmup
        self.device = args.device
        self.cur_stats = torch.zeros(8, device=self.device)
        self.mixing = 0
        self.batch_target_perm = None
        self.copied_grad = {}
        if self.args.resume_pth is not None:
            self.step_ctr = self.args.resume_from_step
        

        if args.grad_scaler:
            self.grad_scaler = KGradScaler(init_scale=args.grad_upscale, growth_factor=args.grad_scaler_growth_factor, backoff_factor=args.grad_scaler_backoff_factor, growth_interval=args.grad_scaler_growth_interval)

    
    def create_optimizer(self, refer_lr, **kwargs):
        if_group_weight = self.args.group_weight
        if if_group_weight:
            model_param = group_weight(self.model)
            if is_main_process():
                print("Grouping weight")
        else:
            model_param = self.model.parameters()
        if self.args.optimizer == 'sgd':
            self.optimizer = optim.SGD(
                model_param, lr=refer_lr, weight_decay=self.args.wd, momentum=self.args.momentum, nesterov=self.args.nesterov
            )
        elif self.args.optimizer == 'adamw':
            self.optimizer = optim.AdamW(
                    model_param, lr=refer_lr, weight_decay=self.args.wd, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps
                )
        elif self.args.optimizer == 'localadamw':
            optimizer = optim.AdamW(model_param, lr=refer_lr, weight_decay=self.args.wd, betas=(self.args.beta1, self.args.beta2), eps=self.args.eps)
            fields_to_avg = []
            if self.args.avg_m:
                fields_to_avg.append('exp_avg')
            if self.args.avg_v:
                fields_to_avg.append('exp_avg_sq')
            self.optimizer = LocalOptimizer(optim=optimizer, warmup_steps=kwargs['warmup_steps'], total_steps = kwargs['total_steps'], alpha=self.args.alpha, power=self.args.power, min_h=self.args.min_h, init_h=self.args.init_h, step_ctr=self.step_ctr, optim_fields_to_average=fields_to_avg )
        elif self.args.optimizer == 'localsgd':
            optimizer = optim.SGD(model_param, lr=refer_lr, weight_decay=self.args.wd, momentum=self.args.momentum, nesterov=self.args.nesterov)
            self.optimizer = LocalOptimizer(optim=optimizer, warmup_steps=kwargs['warmup_steps'], total_steps = kwargs['total_steps'], alpha=self.args.alpha, power=self.args.power, min_h=self.args.min_h, init_h=self.args.init_h, optim_fields_to_average=[] )
        else:
            raise NotImplementedError
        if is_main_process():
            print(f"Optimizer: {self.args.optimizer}")
    
    
    def get_local_step(self):
        if 'local' not in self.args.optimizer:
            return 1
        else:
            return self.optimizer.get_local_step()

    def load_optimizer_state(self):
        if self.args.optimizer_resume_pth is not None:
            if not self.args.multiple_optimizers:
                opt_state_dict = torch.load(self.args.optimizer_resume_pth, map_location=self.device)
                self.optimizer.load_state_dict(opt_state_dict)
                print(opt_state_dict['state'].keys())

            else:
                pth = os.path.join(self.args.optimizer_resume_pth, f'rank={self.args.rank}.pt')
                opt_state_dict = torch.load(pth, map_location=self.device)
                self.optimizer.load_state_dict(opt_state_dict)
    

    def get_optimizer_state_norm(self):
        # if self.args.optimizer == 'adamw':
        #     opt = self.optimizer
        # else:
        #     raise NotImplementedError
        m_vec = get_flat_tensor_from_tensor_sequence(yield_optimizer_state(model=self.model, optimizer=self.optimizer, key='exp_avg'))
        v_vec = get_flat_tensor_from_tensor_sequence(yield_optimizer_state(model=self.model, optimizer=self.optimizer, key='exp_avg_sq'))
        return torch.norm(m_vec), torch.norm(v_vec), torch.norm(m_vec, p=1), torch.norm(v_vec, p=1), {"v1000": v_vec[1000], "v50000": v_vec[50000], "v100000": v_vec[100000]}
    
    
    def update_step(self, batch_image, batch_target, criterion, acc_times):
        def step(input_img, input_target, **kwargs):
            self.optimizer.zero_grad()
            
            with autocast(dtype=self.args.dtype):
                output = self.model(input_img)
                if self.args.strong_aug:
                    loss_train = (1 - kwargs['mixing']) * criterion(output, input_target) + kwargs['mixing'] * criterion(output, kwargs['batch_target_perm'])
                else:
                    loss_train = criterion(output, input_target)
                loss_train /= acc_times
            
            rescaled_loss = loss_train * self.grad_scaler.scale if self.args.grad_scaler else loss_train

            rescaled_loss.backward()
            if self.args.grad_scaler:
                self.grad_scaler.unscale_(self.optimizer)
            return output, loss_train
        
        def optimizer_step():
            if 'local' in self.args.optimizer:
                averaged = self.optimizer.step()
            else:
                averaged = True
                self.optimizer.step()
            if self.args.grad_scaler:
                self.grad_scaler.update()
            return averaged
        
        def log_step():
            if 'local' in self.args.optimizer:
                h = self.optimizer.get_local_step()
            else:
                h = 1
            wandb_dict = {}
            if self.args.log_per_step and is_main_process():
                wandb_dict = {"train_step": self.step_ctr,
                                "train_step/loss":self.cur_stats[2]/self.cur_stats[3], 
                                "train_step/acc1": self.cur_stats[0]/self.cur_stats[3], 
                                "train_step/acc5": self.cur_stats[1]/self.cur_stats[3],
                                "train_step/lr": self.optimizer.param_groups[0]['lr'],
                                "train_step/h": h,
                                "train_step/m": self.m, 
                                "train_step/v": self.v,
                                "train_step/m1": self.m1, 
                                "train_step/v1": self.v1, 
                                "train_step/grad_norm": self.grad_norm,
                                "train_step/param_norm": eval_param_norm(self.model)
                                }
                if 'local' in self.args.optimizer:
                    wandb_dict.update({"train_step/lr_inside": self.optimizer.optim.param_groups[0]['lr']})
                if "adam" in self.args.optimizer:
                    wandb_dict.update(self.v_samples)
                if self.args.grad_scaler:
                    wandb_dict.update({'train_step/grad_scaler': self.grad_scaler.scale})
            return wandb_dict


        def count_step(output, loss_train):
            with torch.no_grad():
                train_correct1, train_correct5 = count_correct(
                    output=output,
                    target=batch_target,
                    topk=(1,5)
                )

                if self.args.strong_aug:
                    train_correct1_perm, train_correct5_perm = count_correct(
                    output=output,
                    target=self.batch_target_perm,
                    topk=(1,5)
                )
                    train_correct1 = (1 - self.mixing) * train_correct1 + self.mixing * train_correct1_perm
                    train_correct5 = (1 - self.mixing) * train_correct5 + self.mixing * train_correct5_perm

            
                cur_stats = torch.stack([
                    train_correct1, train_correct5, loss_train * batch_image.shape[0] * acc_times,
                    torch.as_tensor(batch_image.shape[0], dtype=loss_train.dtype, device=loss_train.device), 
                    torch.as_tensor(0., dtype=loss_train.dtype, device=loss_train.device),
                    torch.as_tensor(0., dtype=loss_train.dtype, device=loss_train.device),
                    torch.as_tensor(0., dtype=loss_train.dtype, device=loss_train.device),
                    torch.as_tensor(0., dtype=loss_train.dtype, device=loss_train.device)
                ])
            return cur_stats
        
        #begin gradient step
        if not self.model.training: # optimize for speed
            self.model.train()
        averaged = None
        kwdct = {}
        input_target = batch_target
        # generate permuted batch
        if self.args.strong_aug:
            batch_image_perm, batch_target_perm, mixing = cf.mixup_batch(batch_image, batch_target, alpha=self.args.mixup_alpha)
            self.batch_target_perm = batch_target_perm
            self.mixing = mixing
            kwdct['batch_target_perm'] = batch_target_perm
            kwdct['mixing'] = mixing
            input_img = batch_image_perm
        else:
            input_img = batch_image



        if self.args.grad_scaler:
            success = False
            for t in range(self.args.grad_scaler_max_retries):
                try:
                    output, loss_train = step(input_img=input_img, input_target=input_target, **kwdct)
                    success = True
                    break
                except GradScaleTooLargeError:
                    pass
            if not success:
                raise ValueError("Cannot find grad_scaler!")
        else:
            output, loss_train = step()

        torch.cuda.synchronize()


        cur_stats = count_step(output, loss_train)
        self.cur_stats += cur_stats

        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name in self.copied_grad:
                        self.copied_grad[name] += param.grad.clone()
                    else:
                        self.copied_grad[name] = param.grad.clone()
        self.acc_ctr += 1
        # if is_main_process():
        #     print(f"acc_ctr {self.acc_ctr}, train step {self.step_ctr}", )

        wandb_dict = {}
        if self.acc_ctr == acc_times:
            # Average gradients
            with torch.no_grad():
                flat_grad = torch.cat([grad.view(-1) for grad in self.copied_grad.values()])

            # if not local methods, average gradients among all gpus
            if 'local' not in self.args.optimizer:
                flat_grad = reduce_value(flat_grad, average=True, group=None)

            
            torch.cuda.synchronize()
            set_flat_grad_to(self.model, flat_grad)
            

            self.grad_norm = torch.norm(flat_grad)

            # clip the gradients
            if not np.isinf(self.args.gradient_clipping):
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.gradient_clipping)
            averaged = optimizer_step()


            self.step_ctr += 1
            if "adam" in self.args.optimizer and self.args.log_per_step:
                self.m, self.v, self.m1, self.v1, self.v_samples = self.get_optimizer_state_norm()
            wandb_dict = log_step()
            if is_main_process() and self.step_ctr <= 300 and self.args.debug:
                print(f"step: {self.step_ctr}, lr: {self.optimizer.param_groups[0]['lr']}")
            #, communication time for gradients {time_comm - time_opt_comm}, communication time for optimizer {time_opt_comm}"
            
            # reset ctr and stats
            cur_stats[4] = self.m
            cur_stats[5] = self.v
            cur_stats[6] = self.m1
            cur_stats[7] = self.v1
            self.acc_ctr = 0
            self.cur_stats = torch.zeros(8, device=self.device)
            self.copied_grad = {}

        
        return cur_stats, averaged, wandb_dict
    def save_model_state_dict(self, pth):
        if self.model.training:
            self.model.eval()
        torch.save(self.model.state_dict(), pth)
    
    def save_optimizer_state_dict(self, pth):
        torch.save(self.optimizer.state_dict(), pth)


    @torch.no_grad()
    def eval_step(self, val_loader, criterion):
        if self.model.training: # optimize for speed
            self.model.eval()
        
        val_stats = torch.zeros(4, device=self.device)
        for images, targets in val_loader:
            with autocast(dtype=self.args.dtype):
                output = self.model(images)
                # loss_val = criterion(output, targets)
                # skip evaluating val loss
                
                val_correct1, val_correct5 = count_correct(
                    output=output,
                    target=targets,
                    topk=(1,5)
                )
                loss_val = torch.zeros_like(val_correct1)

                val_stats += torch.stack([
                val_correct1, val_correct5, loss_val * images.shape[0],
                torch.as_tensor(images.shape[0], dtype=loss_val.dtype, device=loss_val.device)
            ])
            torch.cuda.synchronize()
        val_stats = reduce_value(val_stats, average=False)
        ret = val_stats[:3] / val_stats[3]
        return ret

    def update_bn(self, idx, images):
        for m in self.model.modules():
            if is_bn(m):
                m.momentum = 1 / (1 + idx)
        with torch.no_grad():
            with autocast(dtype=self.args.dtype):
                self.model(images)

    def buffers_to_average(self):
        for name, buffer in self.model.named_buffers():
            if name.endswith("mean") or name.endswith("var"):
                yield buffer

    @torch.no_grad()
    def estimate_BN_params(self, bn_loader):
        if is_main_process():
            print("Estimating BN")
        if not self.model.training: # optimize for speed
            self.model.train()
        bn_loader_iter = iter(bn_loader)
        for idx, (images, targets) in enumerate(bn_loader_iter):
            if idx >= self.args.bn_batches // self.args.world_size:
                break
            self.update_bn(idx, images)
            torch.cuda.synchronize()
        bn_loader_iter.close()

        flat = get_flat_tensor_from_tensor_sequence(self.buffers_to_average())
        flat = reduce_value(flat, average=True)
        set_flat_tensor_to_tensor_sequence(flat, self.buffers_to_average())
