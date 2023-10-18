"""
Random augmentation
"""
from dataclasses import replace
from typing import Callable, Optional, Tuple
from ffcv.pipeline.allocation_query import AllocationQuery
from ffcv.pipeline.operation import Operation
from ffcv.pipeline.state import State
from ffcv.pipeline.compiler import Compiler
import composer.functional as cf
from composer.algorithms.utils import augmentation_sets
import PIL
import torch
import numpy as np

class RandomAugment(Operation):
    def __init__(self, severity, depth, augmentation_set=augmentation_sets['all']):
        super().__init__()
        self.severity = severity
        self.depth = depth
        self.augmentation_set = augmentation_set
    
    def generate_code(self) -> Callable:
        def xrandaug(images, dst):
            my_range = Compiler.get_iterator()
            for i in my_range(images.shape[0]):
                #HWC format
                tmp = images[i]
                pil_image = PIL.Image.fromarray(tmp)

                tmp = np.asarray(cf.randaugment_image(img=pil_image, severity=self.severity, depth=self.depth, augmentation_set=self.augmentation_set))
                #.transpose(1, 2, 0)
                # print(f"output shape {tmp.shape}")
                dst[i] = tmp

            return dst
        
        xrandaug.is_parallel = True

        return xrandaug
    
    def declare_state_and_memory(self, previous_state: State) -> Tuple[State, Optional[AllocationQuery]]:
        return (replace(previous_state, jit_mode=False), 
                AllocationQuery(previous_state.shape, previous_state.dtype))
        
