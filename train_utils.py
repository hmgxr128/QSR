import argparse
import deepspeed
import shutil
import torch
import socket
import yaml
import os
from distributed_utils import is_main_process
import math
from normalizer import *

def parse_trainer_args(parser: argparse.ArgumentParser):
    parser = deepspeed.add_config_arguments(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()

    recipe_args = argparse.Namespace(**yaml.load(open(args.recipe_pth), Loader=yaml.FullLoader))
    args, _ = parser.parse_known_args(namespace=recipe_args)
    
    args = parser.parse_args(namespace=recipe_args)

    return args

@torch.no_grad()
def compute_param_diff(model1, model2):
    ssq = 0
    param_norm_sq = 0
    num_elements = 0
    for param1, param2 in zip(model1.parameters(),model2.parameters()):
        if param1.requires_grad:
            ssq += ((param1 - param2) ** 2).sum()
            param_norm_sq += (param1 ** 2).sum()
            num_elements += param1.numel()
    
    return math.sqrt(ssq), math.sqrt(param_norm_sq), num_elements
def is_bn(m):
    return isinstance(m, nn.BatchNorm1d) or isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm3d) or isinstance(m, GhostBatchNorm1d) or isinstance(m, GhostBatchNorm2d)

def get_data_pth(config_pth='./.config.yml'):
    config = yaml.load(open(config_pth), Loader=yaml.FullLoader)
    return config['imagenet_ffcv_train_pth'], config['imagenet_ffcv_val_pth']

def mkdir(path):
    # remove space
    path=path.strip()
    # remove \ at the end
    path=path.rstrip("\\")
    # judge whether the paths exists
    isExists=os.path.exists(path)
    # judge the result
    if not isExists:
        '''
        differences between os.mkdir(path) and os.makedirs(path): os.mkdirs will create the parent directory but os.mkdir will not
        '''
        # use utf-8 encoding
        os.makedirs(path) 
        print(path + ' is successfully made')
        return True
    else:
        # if the path already exists
        print(path + 'already exists')
        return False

def count_correct(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0)
            res.append(correct_k)
        return res


def if_enough_space(pth, thres=10):
    # Get the disk usage statistics for the specified path
    usage = shutil.disk_usage(pth)

    # Calculate the free space in bytes
    free_space = usage.free

    # Convert the free space to a more readable format
    free_space_gb = free_space / (1024**3)  # Convert bytes to gigabytes

    if free_space_gb > thres:
        return True
    else:
        print('No space left, skip saving')
        return False

def print_lr(step_ctr, optimizer):
    if is_main_process():
        print("Step {}, learning rate {}".format(step_ctr, optimizer.param_groups[0]['lr']))

def adjust_client_lr(client_list, gamma):
    for client in client_list:
        client.decay_lr(gamma)

def print_client_lr(optimizer):
    print(f"learning rate {optimizer.param_groups[0]['lr']}")


@torch.no_grad()
def eval_param_norm(model):
    norm_sq = 0.

    for name, param in model.named_parameters():
            if param.requires_grad:
                norm_sq += torch.sum(param ** 2).item()
            
    return math.sqrt(norm_sq)


def group_weight(module):
    group_decay = []
    group_no_decay = []
    for m in module.modules():
        if isinstance(m, nn.Linear):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.Conv2d):
            group_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)
        elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
            if m.weight is not None:
                group_no_decay.append(m.weight)
            if m.bias is not None:
                group_no_decay.append(m.bias)

    assert len(list(module.parameters())) == len(group_decay) + len(group_no_decay)
    groups = [dict(params=group_decay), dict(params=group_no_decay, weight_decay=.0)]
    return groups

def yield_optimizer_state(model, optimizer, key):
    for p in model.parameters():
        yield optimizer.state[p][key]