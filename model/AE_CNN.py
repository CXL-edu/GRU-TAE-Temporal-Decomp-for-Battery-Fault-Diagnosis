import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, seq_len=96, hid_size=64, individual_channel=False):
        super(Encoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        self.individual_channel = individual_channel
        self.cnn = nn.ModuleList()
        self.relu = nn.ReLU()
        if individual_channel:
            self.cnn.append(nn.Conv1d(7, hid_size, 5, groups=7))
        else:
            self.cnn.append(nn.Conv1d(7, hid_size, 5))

    def forward(self, x):
        out = self.cnn[0](x)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, seq_len=96, hid_size=64, individual_channel=False):
        super(Decoder, self).__init__()
        self.seq_len, self.hid_size = seq_len, hid_size
        self.individual_channel = individual_channel
        self.cnn = nn.ModuleList()
        if individual_channel:
            self.cnn.append(nn.ConvTranspose1d(hid_size, 7, 5, groups=7))
        else:
            self.cnn.append(nn.ConvTranspose1d(hid_size, 7, 5))

    def forward(self, x):
        out = self.cnn[0](x)
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
    model = Model(96, 63, individual_channel=True)
    y = model(x)
    print(y.shape)