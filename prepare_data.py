import torch
from torch.utils.data import DistributedSampler
from typing import List
import os
import numpy as np
from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from rand_augment import RandomAugment
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, RandomResizedCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from distributed_utils import is_main_process
from ffcv.traversal_order.base import TraversalOrder
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406]) * 255
IMAGENET_STD = np.array([0.229, 0.224, 0.225]) * 255
DEFAULT_CROP_RATIO = 224/256
from typing import Sequence

import numpy as np
from torch.utils.data import DistributedSampler



class XRandom(TraversalOrder):
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              num_replicas=self.world_size,
                                              rank=self.rank,
                                              shuffle=True,
                                              seed=self.seed,
                                              drop_last=False)


    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            generator = np.random.default_rng(self.seed + epoch if self.seed is not None else None)
            return generator.permutation(self.indices)

        self.sampler.set_epoch(epoch)

        return self.indices[np.array(list(self.sampler))]


class XSequential(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              num_replicas=self.world_size,
                                              rank=self.rank,
                                              shuffle=False,
                                              seed=self.seed,
                                              drop_last=False)
        

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            return self.indices
        
        self.sampler.set_epoch(epoch)
        
        return self.indices[np.array(list(self.sampler))]

class XPartition(TraversalOrder):
    
    def __init__(self, loader:'Loader'):
        super().__init__(loader)
        self.rank = int(os.environ["RANK"])
        self.world_size = int(os.environ['WORLD_SIZE'])
        if self.distributed:
            self.sampler = DistributedSampler(self.indices,
                                              num_replicas=self.world_size,
                                              rank=self.rank,
                                              shuffle=True,
                                              seed=self.seed,
                                              drop_last=False)
        

    def sample_order(self, epoch: int) -> Sequence[int]:
        if not self.distributed:
            raise ValueError('Partition order is only available in distributed mode')

        # self.sampler.set_epoch(epoch)
        if self.rank == 0:
            print(f"sampled indices {self.indices[np.array(list(self.sampler))][:10]}")
        return np.random.permutation(self.indices[np.array(list(self.sampler))])

def get_loader(
    data_pth, batch_size, num_workers, drop_last, local_rank, train, seed, shuffle=1,
    distributed=1, res=224, in_memory=1) -> Loader:

    this_device = f'cuda:{local_rank}'
    
    if train:
        decoder = RandomResizedCropRGBImageDecoder((res, res))
        image_pipeline: List[Operation] = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device), non_blocking=True)
        ]

        # order = OrderOption.RANDOM
        if shuffle:
            order = XRandom
        else:
            if is_main_process():
                print('Do not shuffle data among workers')
            order = XPartition

        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=drop_last,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=3)
    else:
        cropper = CenterCropRGBImageDecoder((res, res), ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(torch.device(this_device), non_blocking=True),
            ToTorchImage(),
            NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        ]

        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(torch.device(this_device),
            non_blocking=True)
        ]
        #OrderOption.SEQUENTIAL
        loader = Loader(data_pth,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=XSequential,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline
                        },
                        distributed=distributed,
                        seed=seed,
                        batches_ahead=3)
    

    return loader

def get_train_loader_strong_aug(
    data_pth, batch_size, num_workers, drop_last, local_rank, seed, shuffle=1,
    distributed=1, res=224, in_memory=1, depth=2, severity=10) -> Loader:

    this_device = f'cuda:{local_rank}'
    
    

    decoder = RandomResizedCropRGBImageDecoder((res, res))
    image_pipeline: List[Operation] = [
        decoder,
        RandomHorizontalFlip(),
        RandomAugment(severity=severity, depth=depth),
        ToTensor(),
        ToDevice(torch.device(this_device), non_blocking=True),
        ToTorchImage(),
        NormalizeImage(IMAGENET_MEAN, IMAGENET_STD, np.float16)
        
    ]
    if is_main_process():
        print(f'severity {severity}')
    

    label_pipeline: List[Operation] = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(torch.device(this_device), non_blocking=True)
    ]

    if shuffle:
            order = XRandom
    else:
        if is_main_process():
            print('Do not shuffle data.')
        order = XPartition

    loader = Loader(data_pth,
                    batch_size=batch_size,
                    num_workers=num_workers,
                    order=order,
                    os_cache=in_memory,
                    drop_last=drop_last,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed,
                    seed=seed,
                    batches_ahead=3)
    

    return loader

if __name__ == "__main__":
    # loader = get_loader(data_pth="/home/guxinran/ffcv_imagenet/train_500_0.50_90.ffcv", \
    #         batch_size=128, num_workers=12, drop_last=True, rank=0, train=1, \
    #             distributed=0, res=224, in_memory=1)
    # print(len(loader))
    print(issubclass(XRandom, TraversalOrder))
