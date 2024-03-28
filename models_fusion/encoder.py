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
from models_fusion.embed import Embeddings
from models_fusion.mlp import Mlp
from models_fusion.block import Block

class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.Linear = Linear(config.data_hidden_size*2, config.data_hidden_size)
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.data_hidden_size, eps=1e-6)
        for i in range(config.transformer["num_layers"]):
            if i < 1:
                layer = Block(config, vis, mm=True)
            else:
                layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states, text):
        attn_weights = []
        
        for (i, layer_block) in enumerate(self.layer):
            if i == 1:
                if text is None:
                    hidden_states = hidden_states
                    # hidden_states = self.Linear(hidden_states)
                    hidden_states, text, weights = layer_block(hidden_states)
                else:
                    hidden_states = torch.cat((hidden_states, text), 1)
                    hidden_states = self.Linear(hidden_states)
                    hidden_states, text, weights = layer_block(hidden_states)
            elif i < 1:
                hidden_states, text, weights = layer_block(hidden_states, text)
            else:
                hidden_states, text, weights = layer_block(hidden_states)

            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


