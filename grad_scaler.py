import torch


class GradScaleTooLargeError(Exception):
    pass


class KGradScaler:
    
    def __init__(self, init_scale=65536.0, growth_factor=2.0, backoff_factor=0.5, growth_interval=100):
        self.scale = init_scale
        self.growth_factor = growth_factor
        self.backoff_factor = backoff_factor
        self.growth_interval = growth_interval
        self.step = 0
    

    def unscale_(self, optimizer):
        for g in optimizer.param_groups:
            for p in g['params']:
                if p.grad is not None:
                    p.grad /= self.scale
                    if not torch.isfinite(p.grad).all():
                        self.scale *= self.backoff_factor
                        #print(f"grad too large, shrinking sclae to {self.scale}")
                        raise GradScaleTooLargeError()


    def update(self):
        self.step += 1
        if self.step % self.growth_interval == 0:
            self.scale *= self.growth_factor
