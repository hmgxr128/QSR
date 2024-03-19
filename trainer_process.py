from typing import Callable, Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
import os
from prepare_data import get_loader, get_train_loader_strong_aug
from distributed_utils import cleanup, reduce_value, is_main_process
import deepspeed.comm as dist
from train_utils import mkdir, if_enough_space, yield_optimizer_state
import torchvision
from xmodel import XModel
import wandb
import math
from torch.optim.lr_scheduler import LinearLR
import ffcv
from xscheduler import *
from vit_pytorch import SimpleViT
import socket
from timm.loss import LabelSmoothingCrossEntropy
VAL_B = 256

class TrainerProcess:
    device: torch.device

    train_loader: ffcv.loader.Loader
    bn_loader: ffcv.loader.Loader
    val_loader: ffcv.loader.Loader

    num_train: int
    num_val: int

    epoch: int = 0

    best_acc = 0
    # number of steps of this run
    phase_step_ctr: int = 0
    # phase steps plus the number of steps for the checkpoint
    total_step_ctr: int = 0
    
    comm_round: int = 0

    current_lr: float = 0

    # next time to average model parameters
    next_tta: int = 0
    # next time to save model parameters
    next_tts: int = 0

    # number of local steps
    h: int = 0
    tmp_h: int = 0

    diff_norm: float = 0

    warmup_steps: int = 0


    total_time: float = 0

    save_ctr: int = 0
    eval_and_log_time: float = 0


    callbacks: Dict[str, List[Callable]] = {
        'log': []
    }

    def __init__(self, args):
        self.total_time_start = torch.cuda.Event(enable_timing=True)
        self.total_time_end = torch.cuda.Event(enable_timing=True)
        self.total_time_start.record()
        self.args = args

        self.device = args.device
        # initialize logs and stats
        # correct1, correct5, loss, number of samples passed
        self.train_stats = torch.zeros(4, device=self.device)
        self.round_log = []
        self.epoch_log = []
        self.idx_log = []
        self.round_idx_log = []
        self.log_at_avg = []
        self.step_log = torch.zeros(8, device=self.device)




        
        self.batches_per_step = args.acc_times
        self.create_loader()
        if self.args.steps_per_epoch == -1:
            steps_per_epoch = len(self.train_loader) // self.batches_per_step
            self.args.steps_per_epoch = len(self.train_loader) // self.batches_per_step
        else:
            steps_per_epoch = self.args.steps_per_epoch

        


        args.useful_batches = steps_per_epoch * self.batches_per_step
        assert args.useful_batches <= len(self.train_loader)

        
        self.num_train = args.total_batch_size * steps_per_epoch
        self.num_val = 50000
        len_val = len(self.val_loader)



        if is_main_process():
            print(f"Number of training samples: {self.num_train}, length of val loader: {len_val}, steps per epoch: {self.args.steps_per_epoch}, useful batches {args.useful_batches}, batches per step {self.batches_per_step}")
        
        #Define loss function
        if self.args.label_smoothing > 0:
            self.train_criterion = LabelSmoothingCrossEntropy(smoothing=self.args.label_smoothing)
            if is_main_process():
                print(f"Using label smoothing {self.args.label_smoothing}")
        else:
            self.train_criterion = nn.CrossEntropyLoss()
        
        self.val_criterion = nn.CrossEntropyLoss()




        self.total_steps = self.args.epochs_for_sche * self.args.steps_per_epoch

    def create_loader(self):
        loader_bs = self.args.physical_batch_size
        if not self.args.strong_aug:
            self.train_loader = get_loader(
            data_pth=self.args.train_pth, batch_size=loader_bs,
            num_workers=self.args.nw, drop_last=True, local_rank=self.args.gpu, train=1, seed=self.args.seed,
            distributed=1, res=224, in_memory=1
        )
        else:
            self.train_loader = get_train_loader_strong_aug(data_pth=self.args.train_pth, batch_size=self.args.physical_batch_size,
                num_workers=self.args.nw, drop_last=True, local_rank=self.args.gpu, seed=self.args.seed,
                distributed=1, res=224, in_memory=1, depth=self.args.rand_aug_depth, severity=self.args.rand_aug_severity)
        if self.args.bn:
            if not self.args.strong_aug:
                self.bn_loader = get_loader(
                    data_pth=self.args.train_pth, batch_size=loader_bs,
                    num_workers=self.args.nw, drop_last=True, local_rank=self.args.gpu, train=1, seed=self.args.seed,
                    distributed=1, res=224, in_memory=1
                )
            else:
                self.bn_loader = get_train_loader_strong_aug(
                    data_pth=self.args.train_pth, batch_size=loader_bs,
                    num_workers=self.args.nw, drop_last=True, local_rank=self.args.gpu, seed=self.args.seed,
                    distributed=1, res=224, in_memory=1, depth=self.args.rand_aug_depth, severity=self.args.rand_aug_severity
                )
        self.val_loader = get_loader(
            data_pth=self.args.val_pth, batch_size=VAL_B,
            num_workers=self.args.nw, drop_last=False, local_rank=self.args.gpu, train=0, seed=self.args.seed,
            distributed=1, res=224, in_memory=1
            )
        

    def init_model(self):
        if is_main_process():
            print("=> creating model '{}'".format(self.args.model))
        
        if self.args.init_model_by_seed != -1:
            torch.manual_seed(self.args.init_model_by_seed)


        if self.args.model.lower() == 'vit_base':
            model = SimpleViT(image_size=224, patch_size=16, num_classes=1000, dim=768, depth=12, heads=12, mlp_dim=3072).to(self.device)
        else:
            model = torchvision.models.__dict__[self.args.model]().to(self.device)


        if self.args.resume_pth is not None:
            model.load_state_dict(torch.load(self.args.resume_pth, map_location=self.device))
            self.total_step_ctr = self.args.resume_from_step
            # self.next_tta = self.args.resume_from_step + self.args.init_h
            self.next_tts = self.args.resume_from_epoch + self.args.save_freq
            self.epoch = self.args.resume_from_epoch

        else:
            if self.args.init_model_by_seed == -1:
                init_pth = os.path.join(self.args.init_pth, f"H={self.args.init_h}_init.pt")
                if is_main_process():
                    torch.save(model.state_dict(), init_pth)
                
                dist.barrier()
                model.load_state_dict(torch.load(init_pth, map_location=self.device))

            # self.next_tta = self.args.init_h
            self.next_tts = self.args.save_freq
            self.epoch = 0
        
        return model


    def compute_warmup_steps(self):
        if self.args.warmup_steps is None:
            self.warmup_steps = self.args.warmup_epochs * self.args.steps_per_epoch
        else:
            self.warmup_steps = self.args.warmup_steps


    
    def create_tp_scheduler(self):
        warmup_steps = self.warmup_steps

        if self.args.scheduler == 'step':
            self.tp_scheduler = XStepScheduler(max_lr=self.args.max_lr, final_lr=self.args.final_lr, total_epochs=self.args.epochs_for_sche, steps_per_epoch=self.args.steps_per_epoch, warmup_steps=warmup_steps, decay_points=self.args.decay_points, gamma=self.args.gamma)
            if is_main_process():
                print(f'step decay, decay points = {self.tp_scheduler.decay_points}, warmup steps = {self.tp_scheduler.warmup_steps}, gama={self.tp_scheduler.gamma}')
        
        elif self.args.scheduler == 'cosine':
            self.tp_scheduler = XCosineScheduler(max_lr=self.args.max_lr, final_lr=self.args.final_lr, total_epochs=self.args.epochs_for_sche, steps_per_epoch=self.args.steps_per_epoch, warmup_steps=warmup_steps)
            if is_main_process():
                print(f'Using cosine lr decay, warmup steps for cos decay {self.tp_scheduler.warmup_steps}, total epochs for cos {self.tp_scheduler.total_steps_cos / self.args.steps_per_epoch}')

        elif self.args.scheduler == 'cosine_step':
            self.tp_scheduler = XCosineStepScheduler(max_lr=self.args.max_lr, final_lr=self.args.final_lr, total_epochs=self.args.epochs_for_sche, steps_per_epoch=self.args.steps_per_epoch, warmup_steps=warmup_steps, base=self.args.base)
            if is_main_process():
                    print(f'Using cosine lr decay, warmup steps for cos decay {self.tp_scheduler.warmup_steps}, total epochs for cos {self.tp_scheduler.total_steps_cos / self.args.steps_per_epoch}')
        
        elif self.args.scheduler == 'linear':
            self.tp_scheduler = XLinearScheduler(max_lr=self.args.max_lr, final_lr=self.args.final_lr, total_epochs=self.args.epochs_for_sche, steps_per_epoch=self.args.steps_per_epoch, warmup_steps=warmup_steps)
            if is_main_process():
                print(f'Linear decay, total steps for linear = {self.tp_scheduler.total_steps_linear}, warmup steps = {self.tp_scheduler.warmup_steps}')
        else:
            raise NotImplementedError(f"Scheduler {self.args.scheduler} not implemented")
        self.tp_scheduler.adjust_lr(self.xmodel.optimizer, self.xmodel.step_ctr)



    def run(self):
        self.compute_warmup_steps()
        
        
        if is_main_process():
            print(f"max lr: {self.args.max_lr}, warmup steps: {self.warmup_steps}")


        # Initialize model
        model = self.init_model()
        self.xmodel = XModel(model, self.args)
        args_dict = {'warmup_steps': self.warmup_steps, 'total_steps': self.total_steps, 'init_h': self.args.init_h} if 'local' in self.args.optimizer else {}
        if self.args.scheduler == 'cosine_step':
            warmup_max_lr = self.args.base ** np.round(np.log(self.args.max_lr) / np.log(self.args.base))
        else:
            warmup_max_lr = self.args.max_lr
        self.xmodel.create_optimizer(warmup_max_lr, **args_dict)
        if is_main_process():
            print(f'warmup max lr {warmup_max_lr}')

        self.xmodel.load_optimizer_state()
        self.create_tp_scheduler()
        if "local" in self.args.optimizer:
            self.xmodel.optimizer.adjust_h()
            if is_main_process() and self.args.debug:
                print(f"step {self.xmodel.step_ctr}, h = {self.xmodel.get_local_step()}")

        self.next_tta = self.xmodel.step_ctr + self.xmodel.get_local_step()
        if is_main_process() and self.args.debug:
            print(f'next tta: {self.next_tta}')
    
        #images size ([?, 3, 224, 224])
        start_val = torch.cuda.Event(enable_timing=True)
        end_val = torch.cuda.Event(enable_timing=True)
        start_val.record()
        self.eval_and_log_step()
        end_val.record()
        torch.cuda.synchronize()
        self.eval_and_log_time += (start_val.elapsed_time(end_val)) / 1000
        for nepoch in range(self.args.epochs):
            if self.args.resume_pth is not None:
                self.epoch = self.args.resume_from_step // self.args.steps_per_epoch + 1 + nepoch
            else:
                self.epoch = nepoch + 1
            if is_main_process():
                print(f"Entering epoch {self.epoch}")
            self.train_epoch()
        dist.barrier()
        cleanup()

    
    def train_epoch(self):
        train_loader_iter = iter(self.train_loader)
        

        start_val = torch.cuda.Event(enable_timing=True)
        end_val = torch.cuda.Event(enable_timing=True)


        for batch_idx, (images, targets) in enumerate(train_loader_iter):
            if batch_idx >= self.args.useful_batches:
                train_loader_iter.close()
                break
            
            cur_stats, averaged, wandb_dict = self.xmodel.update_step(images, targets, self.train_criterion, self.args.acc_times)

            if wandb_dict:
                self.log(wandb_dict)
            # acc1, acc5, loss, samples passed
            self.train_stats += cur_stats[:4]
            self.step_log += cur_stats
            
            # adjust lr every step
            if batch_idx % self.args.acc_times == self.args.acc_times - 1:
                

                self.tp_scheduler.adjust_lr(self.xmodel.optimizer, self.xmodel.step_ctr)

                
                self.epoch_log.append(self.step_log.clone())
                self.idx_log.append(self.xmodel.step_ctr)
                self.round_log.append(self.step_log.clone()[:4])
                self.step_log.fill_(0)
            # adjust h every round
            if averaged:
                if "local" in self.args.optimizer:
                    self.xmodel.optimizer.adjust_h()
                    if is_main_process() and self.args.debug:
                        print(f"step {self.xmodel.step_ctr}, h = {self.xmodel.get_local_step()}")
                self.comm_round += 1
                self.next_tta += self.xmodel.get_local_step()
                self.log_at_avg.append(self.round_log[0])
                self.round_log = []
                self.round_idx_log.append(self.comm_round)

                if self.check_time_to_eval():
                    start_val.record()
                    self.eval_and_log_step()
                    end_val.record()
                    torch.cuda.synchronize()
                    self.eval_and_log_time += (start_val.elapsed_time(end_val)) / 1000
                    if if_enough_space(self.args.log_pth):
                        self.save_step()


        

    def eval_and_log_step(self):
        def eval_step():
            if self.args.bn:
                self.xmodel.estimate_BN_params(self.bn_loader)
            val_stats = self.xmodel.eval_step(self.val_loader, self.val_criterion)
            return val_stats
        


        def log_avg_worker_step():
            stacked_stats = torch.stack(self.epoch_log)
            stacked_stats = reduce_value(stacked_stats, average=False)
            
            assert stacked_stats.shape[0] == len(self.idx_log)
            if is_main_process():
                for i in range(len(self.idx_log)):
                    self.log({"avg_worker_step/step_idx": self.idx_log[i],
                                "avg_worker_step/train_acc1": stacked_stats[i, 0] / stacked_stats[i, 3],
                                "avg_worker_step/train_acc5": stacked_stats[i, 1] / stacked_stats[i, 3],
                                "avg_worker_step/train_loss": stacked_stats[i, 2] / stacked_stats[i, 3],
                                "avg_worker_step/m": stacked_stats[i, 4] / self.args.world_size,
                                "avg_worker_step/v": stacked_stats[i, 5] / self.args.world_size,
                                "avg_worker_step/m1": stacked_stats[i, 6] / self.args.world_size,
                                "avg_worker_step/v1": stacked_stats[i, 7] / self.args.world_size,
                        })
            self.epoch_log = []
            self.idx_log = []

        def log_at_avg():
            stacked_stats = torch.stack(self.log_at_avg)
            stacked_stats = reduce_value(stacked_stats, average=False)
            assert len(self.round_idx_log) == stacked_stats.shape[0]
            sum_stats = stacked_stats.sum(axis=0)
            if is_main_process():
                # print(f"sum_stats {sum_stats}")
            
                for i in range(len(self.round_idx_log)):
                    self.log({
                        'at_avg_step/round_idx': self.round_idx_log[i],
                        'at_avg_step/train_acc1': stacked_stats[i, 0] / stacked_stats[i, 3],
                        'at_avg_step/train_acc5': stacked_stats[i, 1] / stacked_stats[i, 3],
                        'at_avg_step/train_loss': stacked_stats[i, 2] / stacked_stats[i, 3],
                    })

            wandb_dct = {"avg_train_acc1": sum_stats[0] / sum_stats[3],
                         "avg_train_acc5": sum_stats[1] / sum_stats[3],
                        "avg_train_loss": sum_stats[2] / sum_stats[3]}
            self.log_at_avg = []
            self.round_idx_log = []
            return wandb_dct


        if self.comm_round > 0:
            train_stats = self.average_train_stats()
            if is_main_process():
                print(
                        f"Samples between eval {train_stats[3]}, Epoch {self.epoch}, round {self.comm_round}, "
                        f"train top1 {train_stats[0]}, "
                        f"train top5 {train_stats[1]}, "
                        f"train loss {train_stats[2]}, "
                    )
            
            log_avg_worker_step()
            
            wandb_dct = {}
            wandb_dct.update(log_at_avg())
            val_stats = eval_step()
            self.best_acc = max(self.best_acc, val_stats[0])

            self.total_time_end.record()
            torch.cuda.synchronize()
            self.total_time = (self.total_time_start.elapsed_time(self.total_time_end)) / 1000
            wandb_dct.update({'epoch': self.epoch,
                            'val_acc1': val_stats[0], 
                        'val_acc5': val_stats[1],
                        'best_acc': self.best_acc,
                        'val_loss': val_stats[2],
                        'train_acc1': train_stats[0],
                        'train_acc5': train_stats[1],
                        'train_loss': train_stats[2], 
                        'total_step': self.xmodel.step_ctr,
                        'phase_step': self.xmodel.step_ctr - self.args.resume_from_step,
                        'time/total_time': self.total_time,
                        'time/eval_and_log_time': self.eval_and_log_time})
            if is_main_process():
                self.log(wandb_dct)
                print(f"Epoch {self.epoch}, val acc1 {val_stats[0]}, val acc5 {val_stats[1]}, best acc {self.best_acc}, total_time {self.total_time}, eval_and_log_time {self.eval_and_log_time}")
            self.train_stats.fill_(0)
        



        else:
            val_stats = eval_step()
            if is_main_process():
                print(f"Epoch {self.epoch}, val acc1 {val_stats[0]}, val acc5 {val_stats[1]}, best acc {self.best_acc}, total_time {self.total_time}, eval_and_log_time {self.eval_and_log_time}")
                self.log({'epoch': self.epoch, 'val_acc1': val_stats[0], 'val_acc5': val_stats[1], 'val_loss': val_stats[2]})


    def check_time_to_eval(self):
        next_epoch_end = self.args.steps_per_epoch * math.ceil(self.xmodel.step_ctr / self.args.steps_per_epoch)
        if self.next_tta > next_epoch_end:
            return True
        else:
            return False
    
    def save_step(self):
        if self.epoch >= self.next_tts or self.epoch in self.args.ckpt_to_save:
            if self.epoch >= self.next_tts:
                self.next_tts = self.epoch + self.args.save_freq
            if is_main_process():
                self.xmodel.save_model_state_dict(os.path.join(self.args.log_pth, f"step={self.xmodel.step_ctr}-epoch={self.epoch}.pt"))
                self.save_ctr += 1

            if self.args.save_opt:
                if (self.args.optimizer == 'localadamw' and self.args.avg_m and self.args.avg_v) or ('local' not in self.args.optimizer):
                    if is_main_process():
                        self.xmodel.save_optimizer_state_dict(os.path.join(self.args.log_pth, f"step={self.xmodel.step_ctr}-lopt_epoch={self.epoch}.pt"))
                        self.save_ctr += 1
                else:
                    save_pth = os.path.join(self.args.log_pth, f"optpth_step={self.xmodel.step_ctr}-epoch={self.epoch}")
                    if is_main_process():
                        mkdir(save_pth)
                    dist.barrier()
                    self.xmodel.save_optimizer_state_dict(os.path.join(save_pth, f"rank={self.args.rank}.pt"))



    def average_train_stats(self):
        avg_train_stats = reduce_value(self.train_stats, average=False)
        avg_train_stats[:3] /= avg_train_stats[3]
        
        return avg_train_stats.clone()

    
    def log(self, wandb_dict):
        wandb.log(wandb_dict)
        self.do_callback('log', wandb_dict)


    def register_callback(self, key, callback):
        self.callbacks[key].append(callback)

    def do_callback(self, key, *args, **kwargs):
        for callback in self.callbacks[key]:
            callback(*args, **kwargs)
