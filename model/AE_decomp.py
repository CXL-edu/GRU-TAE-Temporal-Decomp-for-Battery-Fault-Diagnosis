import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


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


# class Encoder(nn.Module):
#     def __init__(self, seq_len=96, hid_size=64, kernel_size=9, individual_channel=False):
#         super(Encoder, self).__init__()
#         self.seq_len, self.hid_size = seq_len, hid_size
#         self.individual_channel = individual_channel
#         self.Linear = nn.ModuleList()
#         self.relu = nn.ReLU()
#         self.decompsition = series_decomp(kernel_size)
#         if individual_channel:
#             for i in range(7):
#                 self.Linear.append(nn.Linear(seq_len, hid_size))
#         else:
#             self.Linear.append(nn.Linear(seq_len, hid_size))
#
#     def forward(self, x):
#
#         if self.individual_channel:
#             out = torch.zeros(x.shape[0], 7, self.hid_size).to(x.device)
#             for i in range(7):
#                 out[:, i, :] = self.Linear[i](x[:, i, :])
#         else:
#             out = self.Linear[0](x)
#         out = self.relu(out)
#         return out


# class Decoder(nn.Module):
#     def __init__(self, seq_len=96, hid_size=64, individual_channel=False):
#         super(Decoder, self).__init__()
#         self.seq_len, self.hid_size = seq_len, hid_size
#         self.individual_channel = individual_channel
#         self.Linear = nn.ModuleList()
#         if individual_channel:
#             for i in range(7):
#                 self.Linear.append(nn.Linear(hid_size, seq_len))
#         else:
#             self.Linear.append(nn.Linear(hid_size, seq_len))
#
#     def forward(self, x):
#         if self.individual_channel:
#             out = torch.zeros(x.shape[0], 7, self.seq_len).to(x.device)
#             for i in range(7):
#                 out[:, i, :] = self.Linear[i](x[:, i, :])
#         else:
#             out = self.Linear[0](x)
#         return out


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, kernel_size=9, individual=False):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_len = seq_len, hid_len
        self.individual = individual
        self.channels = input_size

        # Decompsition Kernel Size
        self.decompsition = series_decomp(kernel_size)
        self.relu = nn.ReLU()

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.seq_len, self.hid_len))
                self.Linear_Trend.append(nn.Linear(self.seq_len, self.hid_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.seq_len, self.hid_len)
            self.Linear_Trend = nn.Linear(self.seq_len, self.hid_len)

    def forward(self, x):
        """
         x: [Batch, seq_len, input_size]
         seasonal_output, trend_output: [Batch, input_size, hid_len]
        """

        seasonal_init, trend_init = self.decompsition(x)
        seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_init.size(0), seasonal_init.size(1), self.hid_len],
                                          dtype=seasonal_init.dtype).to(seasonal_init.device)
            trend_output = torch.zeros([trend_init.size(0), trend_init.size(1), self.hid_len],
                                       dtype=trend_init.dtype).to(trend_init.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_init[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_init[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_init)
            trend_output = self.Linear_Trend(trend_init)

        seasonal_output, trend_output = self.relu(seasonal_output), self.relu(trend_output)
        # x = seasonal_output + trend_output
        return seasonal_output, trend_output


class Decoder(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, kernel_size=9, individual=False):
        super(Decoder, self).__init__()
        self.seq_len, self.hid_len = seq_len, hid_len
        self.individual = individual
        self.channels = input_size

        # Decompsition Kernel Size
        # self.decompsition = series_decomp(kernel_size)
        self.relu = nn.ReLU()

        if self.individual:
            self.Linear_Seasonal = nn.ModuleList()
            self.Linear_Trend = nn.ModuleList()
            for i in range(self.channels):
                self.Linear_Seasonal.append(nn.Linear(self.hid_len, self.seq_len))
                self.Linear_Trend.append(nn.Linear(self.hid_len, self.seq_len))
        else:
            self.Linear_Seasonal = nn.Linear(self.hid_len, self.seq_len)
            self.Linear_Trend = nn.Linear(self.hid_len, self.seq_len)

    def forward(self, seasonal_in, trend_in):
        """
            seasonal_in, trend_in: [Batch, input_size, hid_len]
            x: [Batch, seq_len, input_size]
        """
        # seasonal_init, trend_init = self.decompsition(x)
        # seasonal_init, trend_init = seasonal_init.permute(0, 2, 1), trend_init.permute(0, 2, 1)
        if self.individual:
            seasonal_output = torch.zeros([seasonal_in.size(0), seasonal_in.size(1), self.seq_len],
                                          dtype=seasonal_in.dtype).to(seasonal_in.device)
            trend_output = torch.zeros([trend_in.size(0), trend_in.size(1), self.seq_len],
                                       dtype=trend_in.dtype).to(trend_in.device)
            for i in range(self.channels):
                seasonal_output[:, i, :] = self.Linear_Seasonal[i](seasonal_in[:, i, :])
                trend_output[:, i, :] = self.Linear_Trend[i](trend_in[:, i, :])
        else:
            seasonal_output = self.Linear_Seasonal(seasonal_in)
            trend_output = self.Linear_Trend(trend_in)

        x = seasonal_output + trend_output

        return x.transpose(1, 2)



class Model(nn.Module):
    def __init__(self, seq_len=96, hid_len=64, input_size=7, individual=False):
        super(Model, self).__init__()
        self.encoder = Encoder(seq_len, hid_len=hid_len, input_size=input_size, individual=individual)
        self.decoder = Decoder(seq_len, hid_len=hid_len, input_size=input_size, individual=individual)

    def forward(self, x):
        # x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        seasonal, trend = self.encoder(x)
        x = self.decoder(seasonal, trend)
        # x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model(individual=True)
    y = model(x)
    print(y.shape)