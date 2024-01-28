import os
import pandas as pd
import matplotlib.pyplot as plt

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

"""
需要修改训练代码，修改input和loss部分
在训练时告知是训练过程，需要掩码
在求loss时，只计算重构出来的掩码部分的loss
"""


class MaskedMSELoss(nn.Module):
    def __init__(self, reduction: str = 'mean'):
        super().__init__()
        self.reduction = reduction
        self.mse_loss = nn.MSELoss(reduction=self.reduction)

    def forward(self,
                y_pred: torch.Tensor, y_true: torch.Tensor, mask: torch.BoolTensor = None) -> torch.Tensor:
        """Compute the loss between a target value and a prediction.

        Args:
            y_pred: Estimated values
            y_true: Target values
            mask: boolean tensor with 0s at places where values should be ignored and 1s where they should be considered

        Returns
        -------
        if reduction == 'none':
            (num_active,) Loss for each active batch element as a tensor with gradient attached.
        if reduction == 'mean':
            scalar mean loss over batch as a tensor with gradient attached.
        """

        if mask is None:
            return self.mse_loss(y_pred, y_true)
        else:
            # for this particular loss, one may also elementwise multiply y_pred and y_true with the inverted mask
            mask = mask.bool() if not mask.dtype == torch.bool else mask
            masked_pred = torch.masked_select(y_pred, mask)
            masked_true = torch.masked_select(y_true, mask)

            return self.mse_loss(masked_pred, masked_true)


def random_masking(x, mask_ratio):
    """
    Perform per-sample random masking by per-sample shuffling.
    Per-sample shuffling is done by argsort random noise.
    x: [N, L, D], sequence
    """
    N, L, D = x.shape  # batch, length, dim
    len_keep = int(D * (1 - mask_ratio))

    noise = torch.rand((N, L, D), device=x.device)  # noise in [0, 1]
    # print(noise)

    # sort noise for each sample
    ids_shuffle = torch.argsort(noise, dim=2)  # ascend: small is keep, large is remove

    # keep the first subset
    ids_keep = ids_shuffle[:, :, :len_keep]
    x_masked = x.scatter(dim=-1, index=ids_keep, value=-10000) # 将ids_keep中的索引对应的值替换为-10000

    mask = torch.zeros([N, L, D], device=x.device)
    mask = mask.scatter(dim=-1, index=ids_keep, value=1)    # 将ids_keep中的索引对应的值替换为1

    return x_masked, mask


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int = 7, max_len: int = 5000, device='cpu'):
        super().__init__()

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(-torch.arange(0, d_model, 2) * (math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model, requires_grad=False)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term) if d_model % 2 == 0 else torch.cos(position * div_term)[:, :-1]
        pe = pe.transpose(0, 1).to(device)
        self.register_buffer('pe', pe)  # 保存在buffer中的数据不会被更新，也不会被backward

    def forward(self, x: 'Tensor', pos: [int,int]) -> 'Tensor':
        """
        Arguments:
            x: Tensor, shape ``[batch_size, seq_len, embedding_dim]``
        """
        assert x.size(1) == pos[1]-pos[0], f'positional encoding size error! {x.size(1)} != {pos[1]-pos[0]}'
        x = x + self.pe[:, pos[0]:pos[1]]
        return x


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size

        self.stem = nn.Sequential(
            nn.Linear(input_size, hid_size),
            nn.ReLU()
        )
        self.attn = nn.MultiheadAttention(hid_size, 1)

        self.Linear = nn.Linear(seq_len, hid_len)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.stem(x)    # (batch, seq_len, input_size) -> (batch, seq_len, hid_size)
        x = self.attn(x, x, x)[0]   # (batch, seq_len, hid_size)
        x = x.transpose(1, 2)   # (batch, seq_len, hid_size) -> (batch, hid_size, seq_len)

        out = self.Linear(x)    # (batch, hid_size, seq_len) -> (batch, hid_size, hid_len)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Decoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        # self.proj = nn.Sequential(
        #     nn.Linear(hid_len, seq_len),
        #     nn.ReLU(),
        #     nn.Linear(seq_len, input_size)
        # )
        self.Linear1 = nn.Linear(hid_len, seq_len)
        self.relu = nn.ReLU()
        self.Linear2 = nn.Linear(hid_size, input_size)

    def forward(self, x):
        """
        input: (batch, hid_size, hid_len)
        output: (batch, seq_len, input_size)
        """
        out = self.Linear1(x)
        out = self.relu(out)
        out = out.transpose(1, 2)
        out = self.Linear2(out)
        return out


class Model(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64, mask_ratio=0.2):
        super(Model, self).__init__()
        self.mask_ratio = mask_ratio
        self.pos_enc = PositionalEncoding(d_model=7, max_len=seq_len, device='cuda')
        self.encoder = Encoder(seq_len, hid_len, input_size, hid_size)
        self.decoder = Decoder(seq_len, hid_len, input_size, hid_size)

    def forward(self, x):
        mask_flag = False
        if isinstance(x, tuple):
            x, _ = x
            x = self.pos_enc(x, [0, x.size(1)])
            x, mask = random_masking(x, self.mask_ratio)
        else:
            x = self.pos_enc(x, [0, x.size(1)])

        x = self.encoder(x)    # (batch, seq_len, input_size) -> (batch, hid_size, hid_len)
        x = self.decoder(x)    # (batch, hid_size, hid_len) -> (batch, seq_len, input_size)

        if mask_flag:
            return x, mask
        else:
            return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model()
    y = model(x)
    print(y.shape)