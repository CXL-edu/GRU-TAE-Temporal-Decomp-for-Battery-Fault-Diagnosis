import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn

from .AE_TemporalAttn import Model as AE_Attn
from .AE_LSTM import Model as AE_LSTM

class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """
    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, seq_len, input_size)
        """
        # padding on the both ends of time series
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """
    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, seq_len, input_size)
        """
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class TemporalAttn(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64):
        super(TemporalAttn, self).__init__()
        self.attn_time = nn.ModuleList([
            nn.Linear(input_size, hid_size) if seq_len != hid_len else nn.Identity(),
            nn.MultiheadAttention(seq_len, 1, batch_first=True)
        ])

    def forward(self, x):
        """
        input: (batch, seq_len, input_size)
        output: (batch, seq_len, hid_size)
        """
        out = self.attn_time[0](x)  # (batch, seq_len, input_size) -> (batch, seq_len, hid_size)
        out = out.transpose(1, 2)   # (batch, seq_len, hid_size) -> (batch, hid_size, seq_len)
        out = self.attn_time[1](out, out, out)[0].transpose(1, 2)   # (batch, hid_size, seq_len) -> (batch, seq_len, hid_size)

        return out


class GRUBlock(nn.Module):
    def __init__(self, input_size=7, hid_size=64):
        super(GRUBlock, self).__init__()
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hid_size)
        self.h0 = nn.Parameter(torch.randn((1, 1, hid_size)))

    def forward(self, x):
        # 把h0的第二个维度复制batch次
        h0 = self.h0.repeat(1, x.size(1), 1)
        out, hn = self.gru(x, h0)  # (seq_len, batch, hidden_size), (n_layers, batch, hidden_size)
        out = self.relu(out)
        return out


class AE_Block(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64, kernel_size=9):
        super(AE_Block, self).__init__()
        self.seq_len, self.hid_len = seq_len, hid_len
        self.channels = input_size

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        self.relu = nn.ReLU()

        # self.AE_trend = AE_LSTM(input_size=input_size, hid_size=hid_size)
        # self.AE_seasonal = AE_Attn(seq_len=seq_len, input_size=input_size, hid_size=hid_size)

        self.AE_trend = GRUBlock(input_size=input_size, hid_size=input_size)
        self.AE_seasonal = TemporalAttn(seq_len=seq_len, input_size=input_size, hid_size=input_size)


    def forward(self, x, seasonal_init=None):
        """
         x: [Batch, seq_len, input_size]
         seasonal_output, trend_output: [Batch, input_size, hid_len]
        """
        seasonal, trend = self.decompsition(x)  # (batch, seq_len, input_size) -> (batch, seq_len, input_size)
        seasonal = seasonal + seasonal_init if seasonal_init is not None else seasonal
        trend_output = self.AE_trend(trend)
        seasonal_output = self.AE_seasonal(seasonal)

        return seasonal_output, trend_output


class Model(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, hid_size=64, kernel_size=9, block_num=1):
        super(Model, self).__init__()

        self.AE = nn.ModuleList([
            AE_Block(seq_len=seq_len,
                     hid_len=hid_len,
                     input_size=input_size,
                     hid_size=hid_size,
                     kernel_size=kernel_size
                     ) for _ in range(block_num)
        ])
        # self.proj = nn.Conv1d(in_channels=2*input_size, out_channels=input_size, kernel_size=3, padding=1,
        #                       groups=input_size, padding_mode='replicate')
        self.proj = nn.Linear(in_features=2*input_size, out_features=input_size)

    def forward(self, x):
        # x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        seasonal = None
        seasonal, trend = self.AE[0](x, seasonal)
        for block in self.AE[1:]:
            seasonal, trend = block(trend, seasonal)
            # print(seasonal.shape, trend.shape)
        # x = seasonal + trend

        # temp = torch.zeros((x.shape[0], x.shape[1], x.shape[2]*2)).to(x.device)
        # temp[:, :, ::2] = seasonal
        # temp[:, :, 1::2] = trend

        temp = torch.cat([seasonal, trend], dim=2)

        # x = self.proj(temp.transpose(1, 2)).transpose(1, 2)
        x = self.proj(temp)



        # x = self.decoder(seasonal, trend)
        # x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model()
    y = model(x)
    print(y.shape)