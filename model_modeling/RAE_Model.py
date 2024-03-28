import numpy as np
import torch
import torch.nn as nn
from torch.nn import Dropout, Linear
from torchsummary import summary
from utils.Encoder_model_config import Encoder_get_config
from utils.Decoder_model_config import Decoder_get_config
import torch.nn.functional as F
from models.Encoder import Encoder
from models.Decoder import Decoder


class RAE(nn.Module):
    def __init__(self, Encoder_config, Decoder_config):
        super(RAE, self).__init__()
        self.encoder = Encoder(Encoder_config)  # 实例化encoder对象
        self.decoder = Decoder(Decoder_config)  # 实例化decoder对象

    def forward(self, x):
        x = self.encoder(x)  # 降维
        x = self.decoder(x)  # 自主注意力
        return x
