import torch
import torch.nn as nn
import numpy as np
import math
from torch.nn import Dropout, Linear
from torchsummary import summary


class MutiheadAttention(nn.Module):
    def __init__(self, config):
        super(MutiheadAttention, self).__init__()
        # 输入进来的Q,K,V是相等的，会使用映射linear做一个映射得到参数矩阵Wq,Wk,Wv
        self.n_heads = config.n_heads
        self.d_k = config.d_k
        self.d_v = config.d_v
        self.W_Q = nn.Linear(config.d_model, self.d_k * self.n_heads)
        self.W_K = nn.Linear(config.d_model, self.d_k * self.n_heads)
        self.W_V = nn.Linear(config.d_model, self.d_v * self.n_heads)
        self.linear = nn.Linear(self.n_heads * self.d_v, config.d_model)
        self.ScaledDotProductAttention = ScaledDotProductAttention(config)
        self.layer_norm = nn.LayerNorm(config.d_model)

    def forward(self, Q, K, V):
        # 多头分为如下步骤：首先映射分头，然后计算atten_scores,最后计算atten_value
        # 输入进来的形状Q:[batch_size,tim_seq,d_model]
        #              K:[batch_size,tim_seq,d_model]
        #              V:[batch_size,tim_seq,d_model]
        residual = Q  # (tim_seq, batch_size, d_model)
        bsz, tim_seq, embed_dim = Q.size()  # (batch_size, tim_seq, d_model)
        q_s = self.W_Q(Q).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)  # q_s:[batch_size,n_heads,tim_seq,d_k]
        k_s = self.W_K(K).view(bsz, -1, self.n_heads, self.d_k).transpose(1, 2)  # k_s:[batch_size,n_heads,tim_seq,d_k]
        v_s = self.W_V(V).view(bsz, -1, self.n_heads, self.d_v).transpose(1, 2)  # v_s:[batch_size,n_heads,tim_seq,d_v]

        # 计算多头注意力
        # 得到的结果有两个 context:[batch_size,n_heads,tim_seq,d_v]
        #                   attn:[batch_size,n_heads,tim_seq,tim_seq]
        context = self.ScaledDotProductAttention(q_s, k_s, v_s)
        # context: [batch_size,tim_seq,n_heads,d_v]
        context = context.transpose(1, 2).contiguous().view(bsz, -1, self.n_heads * self.d_v)
        output = self.linear(context)
        # output [batch_size,tim_seq,d_model]
        return self.layer_norm(output + residual)


class ScaledDotProductAttention(nn.Module):
    def __init__(self, config):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = config.d_k

    def forward(self, Q, K, V):
        # 输入进来的维度分别是 K: [batch_size,n_heads,time_seq,d_k]
        # 输入进来的维度分别是 Q: [batch_size,n_heads,time_seq,d_k]
        # 输入进来的维度分别是 V: [batch_size,n_heads,time_seq,d_k]
        # 经过matmul得到的score形状是:[batch_size,n_heads,time_seq,time_seq]
        scores = torch.matmul(Q, K.transpose(-1, -2)) / np.sqrt(self.d_k)
        attn = nn.Softmax(dim=-1)(scores)
        context = torch.matmul(attn, V)
        return context
