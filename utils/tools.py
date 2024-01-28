import os
import numpy as np
import torch


def getModelSize(model):
    param_size = 0
    param_sum = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
        param_sum += param.nelement()

    buffer_size = 0
    buffer_sum = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
        buffer_sum += buffer.nelement()

    all_size = (param_size + buffer_size) / 1024 / 1024
    return param_sum, buffer_sum, all_size


def load_model(model, ema=None, optimizer=None, lr_scheduler=None, loss_scaler=None, path=None, only_model=False):
    start_epoch, start_step, early_stop = 0, 0, 0
    min_loss = np.inf
    if os.path.exists(path):
        ckpt = torch.load(path)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        if only_model:
            model.load_state_dict(ckpt['model'])
            model.to(device)
            min_loss = ckpt["min_loss"]
        else:
            model.load_state_dict(ckpt['model'])
            model.to(device)    # 如果没有这句话且使用gpu时，会导致optimizer.step报错数据不在同一设备上
            optimizer.load_state_dict(ckpt['optimizer'])
            start_epoch = ckpt["epoch"]
            start_step = ckpt["step"]
            min_loss = ckpt["min_loss"]
            early_stop = ckpt["early_stop"]
            if ckpt['loss_scaler'] is not None:
                loss_scaler.load_state_dict(ckpt['loss_scaler'])
            if ckpt['loss_scaler'] is not None:
                loss_scaler.load_state_dict(ckpt['loss_scaler'])
            if ema is not None:
                ema.model = model
                ema.register()
                ema.decay = ckpt['ema']['decay']
                ema.shadow = ckpt['ema']['shadow']
                ema.backup = {}

    return start_epoch, start_step, min_loss, early_stop


def save_model(model, epoch=0, step=0, optimizer=None, lr_scheduler=None, loss_scaler=None, min_loss=0,
               early_stop=0, ema=None, path=None, only_model=False):

    if only_model:
        states = {
            'model': model.state_dict(),
        }
    else:
        states = {
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict() if lr_scheduler is not None else None,
            'loss_scaler': loss_scaler.state_dict() if loss_scaler is not None else None,
            'epoch': epoch,
            'step': step,
            'min_loss': min_loss,
            'early_stop': early_stop,
            'ema': {
                'decay': ema.decay,
                'shadow': ema.shadow,
            } if ema is not None else None,
        }

    torch.save(states, path)


class EMA():
    def __init__(self, model, decay):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
        self.shadow = {k: v.to(param.device) for k, v in self.shadow.items()}
        self.decay = torch.tensor(self.decay, device=param.device)

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                if name in self.shadow:
                    self.shadow[name] = param.data.clone().to(param.device)
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}
