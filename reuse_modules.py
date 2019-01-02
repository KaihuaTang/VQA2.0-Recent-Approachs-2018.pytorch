
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config


class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class FCNet(nn.Module):
    def __init__(self, in_size, out_size, relu=True, drop=0.0):
        super(FCNet, self).__init__()
        self.use_relu = relu
        self.drop_value = drop
        self.lin = weight_norm(nn.Linear(in_size, out_size), dim=None)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.drop_value > 0:
            x = self.drop(x)
        x = self.lin(x)
        if self.use_relu:
            x = self.relu(x)
        return x
