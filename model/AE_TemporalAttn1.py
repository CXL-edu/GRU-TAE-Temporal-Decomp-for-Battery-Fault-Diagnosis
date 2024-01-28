import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class TemporalAttn(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(TemporalAttn, self).__init__()
        self.attn_time = nn.ModuleList([
            nn.Linear(input_size, hid_size) if seq_len != hid_len else nn.Identity(),
            nn.MultiheadAttention(seq_len, 1, batch_first=True),
            # nn.Linear(seq_len, hid_len) if seq_len != hid_len else nn.Identity(),
        ])

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, seq_len, hid_size)
        """
        out = self.attn_time[0](x)  # (batch, seq_len, input_size) -> (batch, seq_len, hid_size)
        out = out.transpose(1, 2)   # (batch, seq_len, hid_size) -> (batch, hid_size, seq_len)
        out = self.attn_time[1](out, out, out)[0].transpose(1, 2)   # (batch, hid_size, seq_len) -> (batch, seq_len, hid_size)
        # out = self.attn_time[2](out).transpose(1, 2)    # (batch, seq_len, hid_size) -> (batch, hid_size, hid_len)

        return out


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        # self.attn = CrossAttn(seq_len, hid_len, input_size, hid_size)
        # self.attn = TemporalAttn(seq_len, hid_len, input_size, hid_size)
        self.linear0 = nn.Linear(input_size, hid_size)
        self.attn = nn.MultiheadAttention(seq_len, 1, batch_first=True)
        self.linear = nn.Linear(seq_len, hid_len)
        self.ln = nn.LayerNorm(hid_len)
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, hid_len, hid_size)
        """
        x = self.linear0(x)
        x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (seq_len, batch, input_size)
        out = self.attn(x, x, x)[0]  # (batch, seq_len, input_size) -> (batch, hid_len, hid_size)
        out = self.linear(out)
        # out = self.ln(out)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, hid_len=64, out_len=96, hid_size=64, out_size=7):
        super().__init__()
        # self.attn = CrossAttn(hid_len, out_len, hid_size, out_size)
        # self.attn = TemporalAttn(hid_len, out_len, hid_size, out_size)
        self.linear = nn.Linear(hid_len, out_len)
        # self.ln = nn.LayerNorm(out_len)
        self.attn = nn.MultiheadAttention(out_len, 1, batch_first=True)
        self.proj = nn.Linear(hid_size, out_size)

    def forward(self, x):
        """
        input: (batch, hid_size, hid_len)
        output: (batch, out_len, out_size)
        """
        x = self.linear(x)
        x = self.attn(x, x, x)[0]

        # x = self.ln(x)
        # out = self.attn(x, x, x)[0]
        out = x.transpose(1, 2)
        out = self.proj(out)
        return out


class Model(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(Model, self).__init__()
        self.encoder = Encoder(seq_len, seq_len, input_size, hid_size)
        self.decoder = Decoder(seq_len, seq_len, hid_size, input_size)

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, seq_len, input_size)
        """
        # x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = self.encoder(x)
        x = self.decoder(x)
        # x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    # model = Model(96, 64, 7, 64)
    # model = TemporalAttn(96, 64, 7, 64)
    model = Model(96, 64, 7, 64)
    y = model(x)
    print(y.shape)