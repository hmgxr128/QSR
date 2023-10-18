# A Quadratic Synchronization Rule for Distributed Deep Learning
This repository provides the code for the paper "A Quadratic Synchronization Rule for Distributed Deep Learning" by Xinran Gu, Kaifeng Lyu, Sanjeev Arora, Jingzhao Zhang, and Longbo Huang. This paper introduces the Quadratic Synchronization Rule (QSR) to determine the synchronization period, denoted as $H$, for local gradient methods. Specifically, we suggest setting $H$ in proportion to $\eta^{-2}$ as the learning rate $\eta$ decays over time. Below, we'd like to provide a guide on reproducing the experiments in our paper and seamlessly incorporating QSR into any PyTorch code.

## Reproduce Our Results

To launch muti-machine training, our implementation uses DeepSpeed with ```torch.distributed``` NCCL backend. Please take a look at [this link](https://www.deepspeed.ai/getting-started/) for a quick start guide on DeepSpeed. We provide sample shell scripts to reproduce our experiments. Please specify the data path and the log path in the ```.config+{yourhostname}.yml``` file. Also, set up ```hostfile``` according to [this link](https://www.deepspeed.ai/getting-started/#resource-configuration-multi-node) to specify the hostnames and the GPU count on each host. Run ```run-vit-adamw-cosine-decay.sh```, ```run-vit-localadamw-cosine-decay.sh```,  ```run-resnet-sgd-cosine-decay.sh``` and ```run-resnet-localsgd-cosine-decay.sh``` to reproduce our results on ViT+AdamW, ViT+LocalAdamW+QSR, ResNet+LocalSGD and ResNet+LocalSGD+QSR, respectively. 

We use [wandb](https://wandb.ai/) to log training statistics. Modify ```WANDB_ENTITY``` and ```PROJ_NAME``` in ```main.py``` to specify your wandb account and project name.



Below, we introduce the usage of each command line argument. 

- ```--recipe-pth```: the yaml file to specify some base configurations, e.g., the model architecture, optimizer, learning rate, and weight decay. Refer to the examples in the ```recipe``` folder. Note that these configurations can be overridden by command line arguments.

- ```--config-pth```: the yaml file to specify the data path and log path. Since machines might store data in different locations, one should provide a unique config file for each machine. Please name the config file as ```.config_{hostname}.yml``` for each respective machine.

- ```--nw```: number of workers for the data loader.

- ```--seed```: random seed for data loader. The provided script will automatically generate a different seed for each execution.

- ```--local-rank```: the local rank passed from the distributed launcher. You don't need to specify it manually.

- ```--physical-batch-size```: The batch size for each back propogation. The program will automatically perform gradient accumulation if the physical batch size is smaller than the total batch size divided by the world size.

- ```--device```: the gpu device corresponding to the process. You don't need to specify it manually.

- ```--epochs```: the number of epochs for the program to run.

- ```--epochs-for-sche```: total number of epochs for the learning rate schedule. We set this argument since the learning rate for a specified iteration may depend on the total training budget. The value of ```--epochs-for-sche``` can differ from ```--epochs``` when you resume training from some checkpoint. For example, for the whole 300-epoch training procedure, when you want to resume training from epoch 150, the program will run for only 150 epochs but the total budget for the learning rate schedule is 300 epochs. 

- ```--steps-per-epoch```: the number of iterations per epoch. If set to '-1', it will be automatically determined as (the size of the training data / total batch size).

- ```--total-batch-size```: total batch size.

- ```--init-model-by-seed```: the seed for model initialization. All processes will use the same seed to guarantee identical initial model parameters. If this argument is set to -1, the model weights will be initialized from a file.

- ```--model```: the model architecture.

- ```--optimizer```: the optimizer.

- ```--max-lr```: the peak learning rate.

- ```--min-lr```: the minimum learning rate.

- ```--momentum```: the momentum coefficient for SGD.

- ```--nesterov```: whether to use nesterov momentum for SGD.

- ```--wd```: the weight decay.

- ```--beta1```: beta1 for AdamW optimizer.

- ```--beta2```: beta2 for AdamW optimizer.

- ```--eps```: epsilon for AdamW optimizer.

- ```--scheduler```: the learning rate schedule.

- ```--decay-points```: This argument defines the epochs at which the learning rate should decay. Provide a list of numbers separated by spaces to specify these epochs.

- ```--gamma```: At each decay point, the learning rate is multiplied by the value of ```gamma```.

- ```--warmup-epochs```: the number of warmup epochs. This argument will be overridden by ```--warmup-steps``` if the latter is not None. 

- ```--warmup-steps```: the number of warmup steps. If ```--warmup-steps``` is provided, it will override this argument.

- ```--log-pth```: the path to save model checkpoints and optimizer states.

- ```--resume-from-epoch```: the epoch of the checkpoint from which to continue training.

- ```--resume-from-step```: the iteration of the checkpoint from which to continue training.

- ```--resume-pth```: the path to the model weights.

- ```--optimizer-resume-pth```: the path to the optimizer state.

- ```--multiple-optimizers```: whether different processes should load different optimizer states. Local gradient methods can lead to different optimizer states across workers. Therefore, we should save the optimizer state for each process. Also, when continuing training from some checkpoint, ensure that each process loads its respective optimizer state.


- ```--wandb```: the mode for wandb. Please choose from "online", "offline" and "disabled".

- ```--wandb-tags```: the tags to be added to the wandb run. Provide a list of strings separated by spaces to specify these tags.

- ```--bn```" whether the model architecture uses BatchNorm. 


- ```--bn-batches```: the number of micro batches to estimate the BatchNorm statistics.

- ```--eval-on-start```: whether to evaluate the accuracy of the initial model.

- ```--save-freq```: The program will save checkpoints ever ```save-freq``` epochs.

- ```--save-opt```: whether to save the optimizer state.

## How can I Incorporate QSR into My Code?

<!-- Our implementation of incorporate any PyTorch optimizer.  -->

## Prepare the Data
Our experiments focus on the ImageNet classification task. You can download the data from [here](https://image-net.org/). To accelerate data loading, we employ FFCV. Please refer to [their website](https://ffcv.io/) for package installation and data preprocessing instructions.

## Requirements

python == 3.9.16

deepspeed == 0.9.5

torch == 2.0.0.post200

torchvision == 0.15.2a0+e7c8f94

wandb == 0.15.6

wandb-osh == 1.1.2

vit-pytorch == 1.2.8

composer == 0.15.1

ffcv == 1.0.2 







