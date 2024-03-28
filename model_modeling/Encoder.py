import numpy as np
import torch
import torch.nn as nn
import math
from torch.nn import Dropout, Linear
from torchsummary import summary
import torch.nn.functional as F
from models.Mutihead_Attention import MutiheadAttention
from models.REtransformer import REtransformer

class Position_Encoding(nn.Module):
    def __init__(self, config):
        super(Position_Encoding, self).__init__()
        self.dropout = nn.Dropout(p=config.dropout)
        # 假设d_model=64,则公式中的pos代表0,1,2,3,4...63的每个位置
        pe = torch.zeros(config.tim_seq, config.d_model)  # [tim_seq, d_model]
        position = torch.arange(0, config.tim_seq, dtype=torch.float).unsqueeze(1)  # [tim_seq, 1]
        # 实现公式中10000^2i/dmodel
        div_term = torch.exp(
            torch.arange(0, config.d_model, 2).float() * (-math.log(10000.0) / config.d_model))  # [d_model/2]
        # 从0开始到最后，步长为2，其实代表的偶数位置
        pe[:, 0::2] = torch.sin(position * div_term)  # [tim_seq, d_model]
        # 从1开始到最后，步长为2，其实代表的奇数位置
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # [tim_seq, 1, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        :param x: [x_len, batch_size, emb_size]
        :return: [x_len, batch_size, emb_size]
        """
        x = x + self.pe[:x.size(0), :]  # !!! [tim_seq, batch_size, d_model]
        return self.dropout(x)


# 8. PoswiseFeedForwardNet
class PoswiseFeedForwardNet(nn.Module):
    def __init__(self, config):
        super(PoswiseFeedForwardNet, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=config.d_model, out_channels=config.d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=config.d_ff, out_channels=config.d_model, kernel_size=1)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, inputs):
        residual = inputs  # inputs : [batch_size, time_seq, d_model]
        output = nn.ReLU()(self.conv1(inputs.transpose(1, 2)))
        output = self.conv2(output).transpose(1, 2)
        return self.layer_norm(output + residual)


# 5. EncoderLayer ：包含两个部分，多头注意力机制和前馈神经网络
class EncoderLayer(nn.Module):
    def __init__(self, config):
        super(EncoderLayer, self).__init__()
        self.enc_self_attn = MutiheadAttention(config)
        self.pos_ffn = PoswiseFeedForwardNet(config)

    def forward(self, enc_inputs):
        # # 下面这个就是做自注意力层，输入是enc_inputs，形状是[batch_size x seq_len_q x d_model]
        # 需要注意的是最初始的QKV矩阵是等同于这个输入的，去看一下enc_self_attn函数 6.
        enc_outputs = self.enc_self_attn(enc_inputs, enc_inputs, enc_inputs)  # enc_inputs to same Q,K,V
        enc_outputs = self.pos_ffn(enc_outputs)  # enc_outputs: [batch_size x len_q x d_model]
        return enc_outputs


class MyTransformerEncoder(nn.Module):
    def __init__(self, config):
        super(MyTransformerEncoder, self).__init__()
        self.input_size = config.dim
        self.rnn_type = config.rnn_type
        self.ksize = config.ksize
        self.n_level = config.n_level
        self.n = config.n
        self.h = config.h
        self.dropout = config.RTdropout
        self.embedding = nn.Linear(in_features=config.rd_dim_2, out_features=config.dim)
        self.rt = REtransformer(self.input_size, self.rnn_type, self.ksize, self.n_level, self.n, self.h, self.dropout)


    def forward(self, x):
        """
        :param x:编码器数入的部分，形状为[batch_size,tim_seq,embed_dim]
        :return:
        """
        x = self.rt(x)
        return x


class Encoder_Transformer(nn.Module):
    def __init__(self, config):
        super(Encoder_Transformer, self).__init__()
        self.encoder = MyTransformerEncoder(config)

    def forward(self, enc_inputs):
        enc_outputs = self.encoder(enc_inputs)
        return enc_outputs


class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()
        self.input_dim = config.input_dim 
        self.rd_dim_1 = config.rd_dim_1  
        self.rd_dim_2 = config.dim2  
        self.rd_dim_3 = config.rd_dim_2  

        self.dim = config.dim 
        self.act_fun = nn.Tanh()
        self.dropout = Dropout(config.dropout_rate)

        self.Linear_1 = Linear(in_features=self.input_dim, out_features=self.rd_dim_1,
                               bias=True)

        self.RNN = nn.LSTM(self.rd_dim_1, self.dim, batch_first=True)

        self.Encoder_transfomer = Encoder_Transformer(config)

    def forward(self, x):
        x = self.Linear_1(x)
        x = self.act_fun(x)
        x = self.dropout(x)
        out, (h, c) = self.RNN(x)
        x = self.Encoder_transfomer(out)
        return x
