##########################
# Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
# Paper Link: https://arxiv.org/abs/1812.05252
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence

import config
import word_embedding

from reuse_modules import Fusion, FCNet

class Net(nn.Module):
    def __init__(self, words_list):
        super(Net, self).__init__()
        question_features = 1024
        vision_features = config.output_features
        glimpses = 2

        self.text = word_embedding.TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.0,
        )

        self.attention = Attention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=1024,
            glimpses=glimpses,
            drop=0.2,)
            
        self.classifier = Classifier(
            in_features=(glimpses * vision_features, question_features),
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.5,)

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        q = self.text(q, list(q_len.data))  # [batch, 1024]
        if config.v_feat_norm: 
            v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v) # [batch, 2048, 36]

        a = self.attention(v.transpose(1,2), q) # [batch, 36, num_glimpse]
        v = apply_attention(v, a) # [batch, 2048 * num_glimpse]
        answer = self.classifier(v, q)

        return answer


class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin11 = FCNet(in_features[0], mid_features)
        self.lin12 = FCNet(in_features[1], mid_features)
        self.lin2 = FCNet(mid_features, mid_features)
        self.lin3 = FCNet(mid_features, out_features, relu=False, drop=drop)

    def forward(self, v, q):
        #x = self.fusion(self.lin11(v), self.lin12(q))
        x = self.lin11(v) * self.lin12(q)
        x = self.lin2(x)
        x = self.lin3(x)
        return x

class Attention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(Attention, self).__init__()
        self.lin_v = FCNet(v_features, mid_features)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features)
        self.lin = FCNet(mid_features, glimpses, relu=False, drop=drop)

    def forward(self, v, q):
        """
        v = batch, num_obj, dim
        q = batch, dim
        """
        v = self.lin_v(v)
        q = self.lin_q(q)
        batch, num_obj, _ = v.shape
        _, q_dim = q.shape
        q = q.unsqueeze(1).expand(batch, num_obj, q_dim)

        x = v * q
        x = self.lin(x) # batch, num_obj, glimps
        x = F.softmax(x, dim=1)
        return x


def apply_attention(input, attention):
    """ 
    input = batch, dim, num_obj
    attention = batch, num_obj, glimps
    """
    batch, dim, _ = input.shape
    _, _, glimps = attention.shape
    x = input @ attention # batch, dim, glimps
    assert(x.shape[1] == dim)
    assert(x.shape[2] == glimps)
    return x.view(batch, -1)
