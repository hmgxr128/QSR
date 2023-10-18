import torch
# import torch.distributed.algorithms.model_averaging.averagers as averagers
import deepspeed.comm as dist

# import torch.distributed.algorithms.model_averaging.utils as utils
import math
import warnings

def is_dist_avail_and_initialized():
    """check if distributed training is supported"""
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True

def is_main_process():
    return dist.get_rank() == 0

def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


@torch.no_grad()
def reduce_value(value, average=True, group=None):
    world_size = dist.get_world_size(group=group)
    if world_size < 2:  # single gpu
        return value
    dist.all_reduce(tensor=value, group=group)
    if average:
        value /= world_size
    return value


def get_flat_tensor_from_tensor_sequence(seq):
    all = []
    for p in seq:
        all.append(p.view(-1))
    return torch.cat(all)


def get_mean_flat_tensor_from_tensor_sequences(seqs):
    all = []
    for ps in zip(*seqs):
        all.append(torch.stack(ps).mean(dim=0).view(-1))
    return torch.cat(all)


def set_flat_tensor_to_tensor_sequence(flat, seq):
    idx = 0
    for p in seq:
        n = p.numel()
        p.data.copy_(flat[idx : idx + n].view_as(p))
        idx += n




class LocalOptimizer(torch.optim.Optimizer):

    def __init__(self, optim: torch.optim.Optimizer, warmup_steps, total_steps, alpha=0., power=2,  min_h=1, init_h=1, step_ctr=0, optim_fields_to_average: list = None):
        self.optim = optim
        self.param_groups = self.optim.param_groups
        self.optim_fields_to_average = optim_fields_to_average if optim_fields_to_average is not None else []
        self.step_ctr = step_ctr
        self.local_step_cnt = 0
        self.min_h = min_h
        self.h = init_h
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        # number of local steps at warmup
        self.init_h = init_h
        self.alpha = alpha
        self.power = power
        


    @property
    def state(self):
        return self.optim.state


    def __repr__(self):
        return self.optim.__repr__()

    def inplace_average_tenosr_sequence(self):
        

        flat = get_flat_tensor_from_tensor_sequence(self.get_tensors_to_average())
        torch.cuda.synchronize()
        reduce_value(flat)
        torch.cuda.synchronize()
        
        set_flat_tensor_to_tensor_sequence(flat, self.get_tensors_to_average())



    def state_dict(self):
        optim_state_dict = self.optim.state_dict()
        optim_state_dict.update({'h': self.h, 'step_ctr': self.step_ctr, 'warmup_steps': self.warmup_steps})
        return optim_state_dict


    def load_state_dict(self, state_dict):
        self.optim.load_state_dict(state_dict)
        property_list = ["h", "step_ctr", "warmup_steps"]
        if all(p in list(state_dict.keys()) for p in property_list):
            self.h = state_dict["h"]
            self.step_ctr = state_dict["step_ctr"]
            self.warmup_steps = state_dict["warmup_steps"]
            self.param_groups = self.optim.param_groups
            if is_main_process():
                print(f"Loaded h = {self.h}. warmup_steps = {self.warmup_steps}. step_ctr = {self.step_ctr}")

        else:
            warnings.warn("Loaded state dict does not contain the number of local steps, step_ctr, warmup_steps ")
            self.param_groups = self.optim.param_groups
            # raise NotImplementedError("Loaded state dict does not contain the number of local steps h, step ctr and warmup steps. ")
    
    def get_local_step(self):
        return self.h


    def adjust_h(self):
        if self.step_ctr < self.warmup_steps or self.alpha == 0:
            self.h = self.init_h
        else:
            lr = self.param_groups[0]['lr']
            remaining_steps = self.total_steps - self.step_ctr
            self.h = max(min(max(int((self.alpha / lr) ** self.power), self.min_h),remaining_steps), 1)
    # pack all tensors to be averaged together
    def get_tensors_to_average(self):
        for group in self.param_groups:
            for p in group['params']:
                if isinstance(p, torch.nn.Parameter):
                    if p.grad is not None:
                        yield p
                    for field in self.optim_fields_to_average:
                        if isinstance(self.optim.state[p][field], torch.Tensor):
                            yield self.optim.state[p][field]
                        else:
                            raise NotImplementedError(f"optim state of type {type(self.optim.state[p][field])} is not supported")
                else:
                    print(p)
                    raise NotImplementedError(f"parameter of type {type(p)} is not supported. p={p}")


    def step(self):
        r"""
        Performs a single optimization step (parameter update).
        """
        self.optim.step()

        self.local_step_cnt += 1
        self.step_ctr += 1
        if self.local_step_cnt >= self.h:
            self.inplace_average_tenosr_sequence()
            self.local_step_cnt = 0
            return True
        return False


    def zero_grad(self, set_to_none: bool = True):  # type: ignore[override]
        self.optim.zero_grad(set_to_none=set_to_none)


    def add_param_group(self, param_group):
        self.optim.add_param_group(param_group)
