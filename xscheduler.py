import numpy as np

class XCosineScheduler():
    def __init__(self, max_lr, final_lr, total_epochs, steps_per_epoch, warmup_steps):
        assert max_lr >= final_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps_cos = total_epochs * steps_per_epoch - warmup_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.final_lr + step * (self.max_lr - self.final_lr) / self.warmup_steps
        else:
            delta_step = min(step - self.warmup_steps, self.total_steps_cos)
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + np.cos(delta_step / self.total_steps_cos * np.pi))
            return lr
    
    def adjust_lr(self, optimizer, step):
        lr = self.get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

class XLinearScheduler():
    def __init__(self, max_lr, final_lr, total_epochs, steps_per_epoch, warmup_steps):
        assert max_lr >= final_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps_linear = total_epochs * steps_per_epoch - warmup_steps

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.final_lr + step * (self.max_lr - self.final_lr) / self.warmup_steps
        else:
            delta_step = min(step - self.warmup_steps, self.total_steps_linear)
            delta_final = self.total_steps_linear - delta_step
            lr = self.final_lr + delta_final * (self.max_lr - self.final_lr) / self.total_steps_linear
            return lr
    
    def adjust_lr(self, optimizer, step):
        lr = self.get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr



class XCosineStepScheduler():
    def __init__(self, max_lr, final_lr, total_epochs, steps_per_epoch, warmup_steps, base):
        assert max_lr >= final_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.total_steps_cos = total_epochs * steps_per_epoch - warmup_steps
        self.type = type
        self.base = base

    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.final_lr + step * (self.max_lr - self.final_lr) / self.warmup_steps
        else:
            delta_step = min(step - self.warmup_steps, self.total_steps_cos)
            lr = self.final_lr + 0.5 * (self.max_lr - self.final_lr) * (1 + np.cos(delta_step / self.total_steps_cos * np.pi))
            lr = self.base ** np.round(np.log(lr) / np.log(self.base))
            return lr
    
    def adjust_lr(self, optimizer, step):
        lr = self.get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr

class XStepScheduler():
    def __init__(self, max_lr, final_lr, total_epochs, steps_per_epoch, warmup_steps, decay_points, gamma=0.1):
        assert max_lr >= final_lr
        self.max_lr = max_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        
        self.decay_points= decay_points
        self.decay_points.insert(0, 0)
        self.decay_points.append(total_epochs + 5)
        self.decay_points = np.array(self.decay_points) * steps_per_epoch
        self.num_decay_points = len(decay_points)
        self.gamma = gamma


    def get_lr(self, step):
        if step < self.warmup_steps:
            return self.final_lr + step * (self.max_lr - self.final_lr) / self.warmup_steps
        else:
            for i in range(self.num_decay_points):
                if step >= self.decay_points[i] and step < self.decay_points[i + 1]:
                    lr = self.max_lr * (self.gamma ** i)
            return lr
            
    
    def adjust_lr(self, optimizer, step):
        lr = self.get_lr(step)
        for g in optimizer.param_groups:
            g['lr'] = lr



