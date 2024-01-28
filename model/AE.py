import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_size=28, individual_channel=False):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        self.individual_channel = individual_channel
        self.Linear = nn.ModuleList()
        self.relu = nn.ReLU()
        if individual_channel:
            for i in range(7):
                self.Linear.append(nn.Linear(seq_len, hid_size))
        else:
            self.Linear.append(nn.Linear(seq_len, hid_size))

    def forward(self, x):
        if self.individual_channel:
            out = torch.zeros(x.shape[0], 7, self.hid_size).to(x.device)
            for i in range(7):
                out[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            out = self.Linear[0](x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, seq_len=96, hid_size=64, individual_channel=False):
        super(Decoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        self.individual_channel = individual_channel
        self.Linear = nn.ModuleList()
        if individual_channel:
            for i in range(7):
                self.Linear.append(nn.Linear(hid_size, seq_len))
        else:
            self.Linear.append(nn.Linear(hid_size, seq_len))

    def forward(self, x):
        if self.individual_channel:
            out = torch.zeros(x.shape[0], 7, self.seq_len).to(x.device)
            for i in range(7):
                out[:, i, :] = self.Linear[i](x[:, i, :])
        else:
            out = self.Linear[0](x)
        return out


class Model(nn.Module):
    def __init__(self, seq_len=96, hid_size=64, individual_channel=False):
        super(Model, self).__init__()
        self.encoder = Encoder(seq_len, hid_size, individual_channel)
        self.decoder = Decoder(seq_len, hid_size, individual_channel)

    def forward(self, x):
        x = x.transpose(1, 2)   # (batch, seq_len, input_size) -> (batch, input_size, seq_len)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(1, 2)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model(individual_channel=True)
    y = model(x)
    print(y.shape)