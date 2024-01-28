import os
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_size=7, hid_size=64):
        super(Encoder, self).__init__()
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=input_size, hidden_size=hid_size)
        self.h0 = nn.Parameter(torch.randn((1, 1, hid_size)))

    def forward(self, x):
        # 把h0的第二个维度复制batch次
        h0 = self.h0.repeat(1, x.size(1), 1)
        out, hn = self.gru(x, h0)  # (seq_len, batch, hidden_size), (n_layers, batch, hidden_size)
        out = self.relu(out)
        return out


class Decoder(nn.Module):
    def __init__(self, input_size=7, hid_size=64):
        super(Decoder, self).__init__()
        self.relu = nn.ReLU()
        self.gru = nn.GRU(input_size=hid_size, hidden_size=input_size)
        self.h0 = nn.Parameter(torch.randn((1, 1, input_size)))

    def forward(self, x):
        # 把h0的第二个维度复制batch次
        h0 = self.h0.repeat(1, x.size(1), 1)
        out, hn = self.gru(x, h0)  # (seq_len, batch, hidden_size), (n_layers, batch, hidden_size)
        return out


class Model(nn.Module):
    def __init__(self, input_size=7, hid_size=64):
        super(Model, self).__init__()
        self.encoder = Encoder(input_size, hid_size)
        self.decoder = Decoder(input_size, hid_size)

    def forward(self, x):
        x = x.transpose(0, 1)   # (batch, seq_len, input_size) -> (seq_len, batch, input_size)
        x = self.encoder(x)
        x = self.decoder(x)
        x = x.transpose(0, 1)
        return x


if __name__ == "__main__":
    x = torch.randn(128, 96, 7)
    model = Model(input_size=7, hid_size=24)
    y = model(x)
    print(y.shape)