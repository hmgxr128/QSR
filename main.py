import argparse
import torch
from distributed_utils import init_deepspeed, is_main_process, update_distributed_args
from train_utils import mkdir, get_data_pth, parse_trainer_args
import torchvision
import os
import wandb
from wandb_osh.hooks import TriggerWandbSyncHook
import socket
from trainer_process import TrainerProcess
import numpy as np
import deepspeed.comm as dist
import yaml

PROJ_NAME = 'ada-localsgd'
WANDB_ENTITY = 'gxr-team'
os.environ["WANDB__SERVICE_WAIT"] = "300"

def main(args):
    if torch.cuda.is_available() is False:
        raise EnvironmentError("not find GPU device for training.")
    
    # Initialize all processes


    assert args.total_batch_size % args.world_size == 0, 'Total batch size should be divisible by world size'
    args.batch_size_per_gpu = args.total_batch_size // args.world_size
    if args.physical_batch_size is None:
        args.physical_batch_size = args.batch_size_per_gpu

    args.acc_times = args.batch_size_per_gpu // args.physical_batch_size
 
    if is_main_process():
        print(f"alpha {args.alpha}, power {args.power}, min_h {args.min_h}")
        print(f"Total batch size {args.total_batch_size}, batch size per gpu {args.batch_size_per_gpu}, physical batch size {args.physical_batch_size}, accumulation times {args.acc_times}")

    

    # if not on cloud, init pth is the same as log pth
    if is_main_process():
        mkdir(args.init_pth)

    if args.resume_pth is not None and is_main_process():
        print(f"Resume training on model {args.resume_pth}")

    if is_main_process():
        set_wandb(args)
    
    tp = TrainerProcess(args)

    if args.wandb == 'offline-sync':
        hook = TriggerWandbSyncHook()
        tp.register_callback('log', lambda _: hook())
    
    tp.run()


def set_wandb(args):
    config = vars(args)

    if args.wandb == 'offline-sync':
        mode = 'offline'
    else:
        mode = args.wandb

    run = wandb.init(
        mode=mode,
        project=PROJ_NAME,
        entity=WANDB_ENTITY,
        name=args.wandb_name,
        config=config,
        settings=wandb.Settings(_service_wait=300),
        tags=args.wandb_tags
    )
    wandb.run.log_code(".")
    #settings=wandb.Settings(start_method='fork'),


if __name__ == '__main__':
    init_deepspeed()



    model_names = sorted(name for name in torchvision.models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(torchvision.models.__dict__[name]))
    model_names.extend(['vit_base'])


    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe-pth', type=str, default="./recipe/vit-B=4096-adamw-cos.yml")
    parser.add_argument('--config-pth', type=str, default=".config")
    # parser.add_argument('--train-pth', type=str, default=None)
    # parser.add_argument('--val-pth', type=str, default=None)
    parser.add_argument('--nw', type=int, default=12)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--local_rank',
                        type=int,
                        default=-1,
                        help='local rank passed from distributed launcher')

    parser.add_argument('--physical-batch-size', type=int, default=None)
    # parser.add_argument('--batch-size-per-gpu', type=int, default=None)
    parser.add_argument('--device', default='cuda', help='device id (i.e. 0 or 0,1 or cpu)')
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--epochs-for-sche', type=int, default=300)
    parser.add_argument('--steps-per-epoch', type=int, default=-1)
    parser.add_argument('--total-batch-size', type=int, default=4096)
    
    parser.add_argument('--init-model-by-seed', type=int, default=-1, help='init model by file (-1) or by seed')
    parser.add_argument('--model', type=str, default='resnet50', choices=model_names, help='The model architecture')


    # Optimizer Hyperparameter
    parser.add_argument('--optimizer', type=str, default="adamw", choices=['adamw', 'sgd', 'localadamw', 'localsgd'])
    parser.add_argument('--max-lr', type=float, default=0.01)
    parser.add_argument('--min-lr', type=float, default=1e-6)
    # parser.add_argument('--final-lr', type=float, default=1e-6, help='final learning rate')

    parser.add_argument('--momentum', type=float, default=0.9, help='Momentum value')
    parser.add_argument('--nesterov', type=int, default=0, help='Whether to use nesterov momentum')
    parser.add_argument('--wd', type=float, default=5e-4, help='Weight decay value')

    
    parser.add_argument('--beta1', type=float, default=0.9)
    parser.add_argument('--beta2', type=float, default=0.999)
    parser.add_argument('--eps', type=float, default=1e-8)
    parser.add_argument('--scheduler', type=str, default='cosine', choices=['step', 'cosine', 'cosine_step', 'linear','const'])
    parser.add_argument('--decay-points', nargs='+', type=int)
    parser.add_argument('--gamma', type=float, default=0.1, help='The factor for stepwise learning rate decay')
    parser.add_argument('--warmup-epochs', type=int, default=5, help='will be overwritten by warmup-steps if the latter is not none')
    parser.add_argument('--warmup-steps', type=int, default=None)


    parser.add_argument('--log-pth', type=str, default=None, help='The path to save model params')

    parser.add_argument('--resume-from-epoch', type=int, default=0, help='The epoch to continue training')
    parser.add_argument('--resume-from-step', type=int, default=0, help='The number of steps to continue training')


    parser.add_argument('--resume-pth', type=str, default=None, help='The path to load the model to resume')
    parser.add_argument('--optimizer-resume-pth', type=str, default=None, help='The path to load the optimizer state to resume')

    parser.add_argument('--multiple-optimizers', type=int, default=0)





    
    parser.add_argument('--wandb', type=str, default='online', choices=['online', 'offline', 'disabled'], help='wandb mode')
    parser.add_argument('--wandb-tags', nargs='+', type=str)
    
    
    


    parser.add_argument('--bn', type=int, default=1, help='whether model uses bn')
    parser.add_argument('--bn-batches', type=int, default=100)
    
    parser.add_argument('--eval-on-start', type=int, default=1)




    parser.add_argument('--save-freq', type=int, default=50)
    # parser.add_argument('--save-latest', type=int, default=1)
    parser.add_argument('--save-opt', type=int, default=1, help='whether to save optimizer state')



    parser.add_argument('--log-per-step', type=int, default=0)


    #correction for precision problem
    parser.add_argument('--dtype', type=int, default=16)
    parser.add_argument('--grad-scaler', type=int, default=1, help='whether to use grad scaler')
    parser.add_argument('--grad-upscale', type=float, default=65536)
    parser.add_argument('--grad-scaler-max-retries', type=int, default=50)
    parser.add_argument('--grad-scaler-growth-factor', type=float, default=2)
    parser.add_argument('--grad-scaler-backoff-factor', type=float, default=0.5)
    parser.add_argument('--grad-scaler-growth-interval', type=int, default=100)

    parser.add_argument('--debug', type=int, default=1, help='whether to turn on debug mode')

    parser.add_argument('--avg-m', type=int, default=0, help='whether to average momentum buffer')

    parser.add_argument('--avg-v', type=int, default=0, help='whether to average squared sg buffer')

    parser.add_argument('--base', type=float, default=2, help='base of lr')

    parser.add_argument('--ckpt-to-save', nargs='+', type=int)

    
    parser.add_argument('--alpha', type=float, default=0, help='multiplier for 1 / lr, if set as zero, means do not take adaptiave steps')
    parser.add_argument('--init-h', type=int, default=1)
    parser.add_argument('--min-h', type=int, default=1, help='minimum number of local steps')
    parser.add_argument('--power', type=float, default=1.0, help='the power for schedule')

    parser.add_argument('--log-each-layer', type=int, default=0)
    # label smoothing
    parser.add_argument('--label-smoothing', type=float, default=0)
    #params for augmentation
    parser.add_argument('--strong-aug', type=int, default=0, help='whether to add randaug and mixup')
    parser.add_argument('--rand-aug-depth', type=int, default=2)
    parser.add_argument('--rand-aug-severity', type=int, default=10)
    parser.add_argument('--mixup-alpha', type=float, default=0.2)
    parser.add_argument('--gradient-clipping', type=float, default=float('inf'), help='threshold for gradient clipping')

    
    args = parse_trainer_args(parser)
    update_distributed_args(args)






    if args.dtype == 16:
        args.dtype = torch.float16
    if args.dtype == 32:
        args.dtype = torch.float32

    if args.resume_pth == 'None':
        args.resume_pth = None
    
    
    if args.ckpt_to_save == None:
        args.ckpt_to_save = []
    

    args.optimizer = args.optimizer.lower()
    args.model = args.model.lower()

    args.final_lr = args.min_lr
    

    
    args.wandb_name = f'opt={args.optimizer}-max_lr={args.max_lr}-alpha={args.alpha}-wd={args.wd}-m={args.model}-B={args.total_batch_size}'
    
    log_pth = args.wandb_name + f'-seed={args.seed}'
    

    host_name = socket.gethostname()
    args.config_pth = f"{args.config_pth}_{host_name}.yml"
    config = yaml.load(open(args.config_pth), Loader=yaml.FullLoader)
    print(args.config_pth)
    args.log_pth = os.path.join(config['log_pth'], log_pth)
    args.init_pth = args.log_pth

    args.train_pth, args.val_pth = get_data_pth(args.config_pth)
    



    main(args)