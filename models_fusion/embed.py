# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math

from os.path import join as pjoin

import torch
import torch.nn as nn
import numpy as np

from torch.nn import BCEWithLogitsLoss,CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage

import models_fusion.configs as configs
from models_fusion.attention import Attention
import pdb

class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """
    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        # self.hybrid = None
        # img_size = _pair(img_size)
        # tk_lim = config.cc_len  # 40
        # num_lab = config.lab_len  # 92
        #  KKI 5565 NYU 4465  # PU 5050
        self.data_embeddings = Linear(4465, config.data_hidden_size)
        self.Index_embeddings = Linear(1, config.hidden_size)
        # self.Inattentive_embeddings = Linear(1, config.hidden_size)
        # self.HyperAndImpulsive_embeddings = Linear(1, config.hidden_size)

        self.pe_data = nn.Parameter(torch.zeros(config.batchsize, config.data_hidden_size))
        self.pe_Index = nn.Parameter(torch.zeros(config.batchsize, config.hidden_size))
        # self.pe_Inattentive = nn.Parameter(torch.zeros(config.batchsize, config.hidden_size))
        # self.pe_HyperAndImpulsive = nn.Parameter(torch.zeros(config.batchsize, config.hidden_size))

        # self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])
        self.dropout_Index = Dropout(config.transformer["dropout_rate"])
        # self.dropout_Inattentive = Dropout(config.transformer["dropout_rate"])
        # self.dropout_HyperAndImpulsive = Dropout(config.transformer["dropout_rate"])

    def forward(self, x, Index):
        # Index, Inattentive, HyperAndImpulsive
        # B = x.shape[0]
        # cls_tokens = self.cls_token.expand(B, -1, -1)
        # print("\nembed前x.shape:", x.shape)  # torch.Size([2, 4465])
        # print("embed前gender.shape:", gender.shape)  # torch.Size([2, 1])
        # print("embed前age.shape:", age.shape)  # torch.Size([2, 1])
        if Index is None :
            x = self.data_embeddings(x)
            embeddings = x + self.pe_data
            embeddings = self.dropout(embeddings)
            return embeddings, Index

        else:
            x = self.data_embeddings(x)
            Index = self.Index_embeddings(Index)
            embeddings = x + self.pe_data
            Index_embeddings = Index + self.pe_Index
            embeddings = self.dropout(embeddings)
            Index_embeddings = self.dropout_Index(Index_embeddings)
            return embeddings, Index_embeddings
        # Inattentive = self.Inattentive_embeddings(Inattentive)
        # HyperAndImpulsive = self.HyperAndImpulsive_embeddings(HyperAndImpulsive)
        # print("编码后x.shape:", x.shape)  # torch.Size([2, 2232])
        # print("编码后gender.shape:", gender.shape)  # torch.Size([2, 2232])
        # print("编码后age.shape:", age.shape)  # torch.Size([2, 2232])
        #
        # # x = x.flatten(2)
        # # x = x.transpose(-1, -2)
        # # x = torch.cat((cls_tokens, x), dim=1)
        # print("位置信息pe_data.shape:", self.pe_data.shape)  # torch.Size([2, 2232])
        # print("位置信息pe_gender.shape:", self.pe_gender.shape)  # torch.Size([2, 2232])
        # print("位置信息pe_age.shape:", self.pe_age.shape)  # torch.Size([2, 2232])

        # Inattentive_embeddings = Inattentive + self.pe_Inattentive
        # HyperAndImpulsive_embeddings = HyperAndImpulsive + self.pe_HyperAndImpulsive

        # print("位置信息添加后embeddings.shape:", embeddings.shape)  # torch.Size([2, 2232])
        # print("位置信息添加后gender_embeddings.shape:", gender_embeddings.shape)  # torch.Size([2, 2232])
        # print("位置信息添加后age_embeddings.shape:", age_embeddings.shape)  # torch.Size([2, 2232])


        # Inattentive_embeddings = self.dropout_Inattentive(Inattentive_embeddings)
        # HyperAndImpulsive_embeddings = self.dropout_HyperAndImpulsive(HyperAndImpulsive_embeddings)
        # return embeddings, Index_embeddings, Inattentive_embeddings, HyperAndImpulsive_embeddings




