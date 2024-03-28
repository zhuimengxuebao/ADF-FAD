import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear, Conv1d, Dropout
from torchsummary import summary
import torch.nn.functional as F
from models.Mutihead_Attention import MutiheadAttention
import math
import typing


def make_conv1d(in_channels: int, out_channels: int, kernel_size: typing.Union[int, tuple], stride: int,
                padding: int, dilation=1, groups=1,
                bias=True) -> nn.Module:
    """
    produce a Conv1D with Batch Normalization and ReLU
    return: my conv1d module
    """
    module = nn.Sequential(
        Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation,
               groups=groups,
               bias=bias),
        nn.BatchNorm1d(out_channels),
        nn.ReLU())
    return module


class Decoder(nn.Module):
    def __init__(self, config):
        super(Decoder, self).__init__()
        self.input_dim = config.dim  # 输入维度64
        self.rd_dim_3 = config.rd_dim_2  # 第一个维度128
        self.rd_dim_4 = config.dim2      # 第二个维度256
        self.rd_dim_5 = config.rd_dim_1  # 第二个维度512
        self.out_dim = config.input_dim  # 自注意维度28546
        self.act_fun = nn.Tanh()
        self.dropout = Dropout(config.dropout_rate)
        '''
        # (1,24,64) -->(1,24,128)
        self.RNN_1 = nn.LSTM(self.input_dim, self.rd_dim_3, batch_first=True)
        # (1,24,128) -->(1,24,256)
        self.RNN_2 = nn.LSTM(self.rd_dim_3, self.rd_dim_4, batch_first=True)
        # (1,24,256) -->(1,24,512)
        self.RNN_3 = nn.LSTM(self.rd_dim_4, self.rd_dim_5, batch_first=True)
        '''
        # (1,24,64) -->(1,24,512)
        self.RNN = nn.LSTM(self.input_dim, self.rd_dim_5, batch_first=True)
        # (1,24,512) -->(1,24,28546)
        self.Linear_1 = Linear(in_features=self.rd_dim_5, out_features=self.out_dim,
                               bias=True)

    def forward(self, x):
        '''
        # (1,24,64) -->(1,24,128)
        out1, (h1, c1) = self.RNN_1(x)
        # (1,24,128) -->(1,24,256)
        out2, (h2, c2) = self.RNN_2(out1)
        # (1,24,256) -->(1,24,512)
        out, (h3, c3) = self.RNN_3(out2)
        '''
        # (1,24,64) -->(1,24,512)
        out, (h, c) = self.RNN(x)
        # (1,24,512) -->(1,24,28546)
        x = self.Linear_1(out)
        x = self.act_fun(x)
        # x = self.dropout(x)
        return x
