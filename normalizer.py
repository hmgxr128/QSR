import torch
from torch import nn
import torch.nn.functional as F


def _repeat_running_stats(self, n):
    if self.track_running_stats:
        if self.running_mean.shape[0] != n * self.num_features:
            self.running_mean = torch.mean(self.running_mean.view(-1, self.num_features), dim=0).repeat(n)
        if self.running_var.shape[0] != n * self.num_features:
            self.running_var = torch.mean(self.running_var.view(-1, self.num_features), dim=0).repeat(n)


class GhostBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, num_features, bn_batch_size=32, **kw):
        super().__init__(num_features, **kw)
        self.bn_batch_size = bn_batch_size


    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            _repeat_running_stats(self, 1)
        return super().train(mode)
    

    def forward(self, input):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training or not self.track_running_stats:
            N, C, H, W = input.shape
            assert N % self.bn_batch_size == 0
            num_splits = N // self.bn_batch_size
            _repeat_running_stats(self, num_splits)
            return F.batch_norm(
                input.reshape(-1, C * num_splits, H, W), self.running_mean, self.running_var,
                self.weight.repeat(num_splits), self.bias.repeat(num_splits),
                True, exponential_average_factor, self.eps).reshape(N, C, H, W)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias,
                False, exponential_average_factor, self.eps)


class GhostBatchNorm1d(nn.BatchNorm1d):
    def __init__(self, num_features, bn_batch_size, **kw):
        super().__init__(num_features, **kw)
        self.bn_batch_size = bn_batch_size


    def train(self, mode=True):
        if (self.training is True) and (mode is False):  # lazily collate stats when we are going to use them
            _repeat_running_stats(self, 1)
        return super().train(mode)
    

    def forward(self, input):
        # exponential_average_factor is set to self.momentum
        # (when it is available) only so that it gets updated
        # in ONNX graph when this node is exported to ONNX.
        if self.momentum is None:
            exponential_average_factor = 0.0
        else:
            exponential_average_factor = self.momentum

        if self.training and self.track_running_stats:
            if self.num_batches_tracked is not None:  # type: ignore[has-type]
                self.num_batches_tracked.add_(1)  # type: ignore[has-type]
                if self.momentum is None:  # use cumulative moving average
                    exponential_average_factor = 1.0 / float(self.num_batches_tracked)
                else:  # use exponential moving average
                    exponential_average_factor = self.momentum
        
        if self.training or not self.track_running_stats:
            N, C = input.shape
            assert N % self.bn_batch_size == 0
            num_splits = N // self.bn_batch_size
            _repeat_running_stats(self, num_splits)
            return F.batch_norm(
                input.reshape(-1, C * num_splits), self.running_mean, self.running_var,
                self.weight.repeat(num_splits), self.bias.repeat(num_splits),
                True, exponential_average_factor, self.eps).reshape(N, C)
        else:
            return F.batch_norm(
                input, self.running_mean, self.running_var,
                self.weight, self.bias,
                False, exponential_average_factor, self.eps)