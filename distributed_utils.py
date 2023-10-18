import torch
import deepspeed
import deepspeed.comm as dist
import os
import numpy as np


from deepspeed.accelerator import get_accelerator


def init_deepspeed():
    deepspeed.init_distributed()
    local_rank = int(os.environ['LOCAL_RANK'])
    get_accelerator().set_device(local_rank)

def init_comm_groups(args):
    assert args.world_size % args.num_nodes == 0, "world size should be divisible by number of nodes"
    args.gpus_per_node = args.world_size // args.num_nodes
    args.node_idx = args.rank // args.gpus_per_node
    groups = []
    for i in range(args.num_nodes):
        groups.append(dist.new_group(list(range(i * args.gpus_per_node, (i + 1) * args.gpus_per_node))))
    args.comm_group = groups[args.node_idx]
    

def update_distributed_args(args):
    args.rank = int(os.environ["RANK"])
    args.world_size = int(os.environ['WORLD_SIZE'])
    if args.local_rank == -1:
        args.local_rank = int(os.environ['LOCAL_RANK'])
    args.gpu = args.local_rank # Is equal to rank in single 
    args.device = torch.device(args.gpu)

def sync_all():
    get_accelerator().synchronize()
    dist.barrier()


def cleanup():
    dist.destroy_process_group()

def get_rank():
    return dist.get_rank()

def is_main_process():
    return dist.get_rank() == 0

def get_world_size():
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

def get_flat_buffers_from(model):
    buffers = []
    for name, buffer in model.named_buffers():
        if name.endswith("mean") or name.endswith("var"):
            buffers.append(buffer.data.view(-1))
    flat_buffers = torch.cat(buffers)
    return flat_buffers



def set_flat_buffers_to(model, flat_buffers):
    prev_ind = 0
    for name, buffer in model.named_buffers():
        if name.endswith("mean") or name.endswith("var"):
            flat_size = int(np.prod(list(buffer.size())))
            buffer.data.copy_(flat_buffers[prev_ind:prev_ind + flat_size].view(buffer.size()))
            prev_ind += flat_size
@torch.no_grad()
def get_flat_grad_from(model):
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            grads.append(param.grad.data.view(-1))
    flat_grads = torch.cat(grads)
    return flat_grads

@torch.no_grad()
def set_flat_grad_to(model, flat_grad):
    idx = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            n = param.grad.data.numel()
            param.grad.data.copy_(flat_grad[idx : idx + n].view_as(param.grad.data))
            idx += n



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
