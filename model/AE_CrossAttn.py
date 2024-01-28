import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class CrossAttn(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(CrossAttn, self).__init__()
        self.attn_time = nn.ModuleList([
            nn.Linear(seq_len, hid_len) if seq_len != hid_len else nn.Identity(),
            nn.MultiheadAttention(hid_len, 1, batch_first=True),
        ])
        self.attn_var = nn.ModuleList([
            nn.Linear(input_size, hid_size) if input_size != hid_size else nn.Identity(),
            nn.MultiheadAttention(hid_size, 1, batch_first=True),
        ])

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, hid_len, hid_size)
        """
        x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        out = self.attn_time[0](x)
        out = self.attn_time[1](out, out, out)[0].transpose(1, 2)
        out = self.attn_var[0](out)
        out = self.attn_var[1](out, out, out)[0]
        return out


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        self.attn = CrossAttn(seq_len, hid_len, input_size, hid_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, hid_len, hid_size)
        """
        # x = x.transpose(0, 1)   # (batch, seq_len, input_size) -> (seq_len, batch, input_size)
        out = self.attn(x)  # (batch, seq_len, input_size) -> (batch, hid_len, hid_size)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, hid_len=64, out_len=96, hid_size=64, out_size=7):
        super().__init__()
        self.attn = CrossAttn(hid_len, out_len, hid_size, out_size)

    def forward(self, x):
        """
        input: (batch, hid_len, hid_size)
        output: (batch, out_len, out_size)
        """
        out = self.attn(x)
        return out


class Model(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Model, self).__init__()
        self.encoder = Encoder(seq_len, hid_len, input_size, hid_size)
        self.decoder = Decoder(hid_len, seq_len, hid_size, input_size)

    def forward(self, x):
        # x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model(96, 64, 7, 64)
    y = model(x)
    print(y.shape)