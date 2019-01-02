##########################
# Implementation of Bilinear Attention Networks
# Paper Link: https://arxiv.org/abs/1805.07932
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

from counting import Counter
import config
import word_embedding

from reuse_modules import Fusion, FCNet

class Net(nn.Module):
    def __init__(self, words_list):
        super(Net, self).__init__()
        num_hid = 1280
        question_features = num_hid
        vision_features = config.output_features
        glimpses = 12
        objects = 10

        self.text = word_embedding.TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=question_features,
            use_hidden=False, 
            drop=0.0,
        )

        self.count = Counter(objects)

        self.attention = BiAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=num_hid,
            glimpses=glimpses,
            drop=0.5,)

        self.apply_attention = ApplyAttention(
            v_features=vision_features,
            q_features=question_features,
            mid_features=num_hid,
            glimpses=glimpses,
            num_obj=objects,
            count = self.count,
            drop=0.2,
        )
            
        self.classifier = Classifier(
            in_features=num_hid,
            mid_features=num_hid * 2,
            out_features=config.max_answers,
            drop=0.5,)

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        '''
        v: visual feature      [batch, 2048, num_obj]
        b: bounding box        [batch, 4, num_obj]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        q = self.text(q, list(q_len.data))  # [batch, len, dim]
         
        if config.v_feat_norm: 
            v = v / (v.norm(p=2, dim=1, keepdim=True) + 1e-12).expand_as(v) # [batch, 2048, 36]
        v = v.transpose(1,2)

        atten, logits = self.attention(v, q, v_mask, q_mask) # batch x glimpses x v_num x q_num

        new_q = self.apply_attention(v, q, b, v_mask, q_mask, atten, logits)
        answer = self.classifier(new_q)

        return answer


class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features)
        self.lin2 = FCNet(mid_features, out_features, relu=False, drop=drop)

    def forward(self, q):
        x = self.lin1(q)
        x = self.lin2(x)
        return x

class BiAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, drop=0.0):
        super(BiAttention, self).__init__()
        self.hidden_aug = 1
        self.glimpses = glimpses
        self.lin_v = FCNet(v_features, int(mid_features * self.hidden_aug), drop=drop/2.5)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, int(mid_features * self.hidden_aug), drop=drop/2.5)
        
        self.h_weight = nn.Parameter(torch.Tensor(1, glimpses, 1, int(mid_features * self.hidden_aug)).normal_())
        self.h_bias = nn.Parameter(torch.Tensor(1, glimpses, 1, 1).normal_())

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        """
        v_num = v.size(1)
        q_num = q.size(1)

        v_ = self.lin_v(v).unsqueeze(1)  # batch, 1, v_num, dim
        q_ = self.lin_q(q).unsqueeze(1)  # batch, 1, q_num, dim
        v_ = self.drop(v_)

        h_ = v_ * self.h_weight # broadcast:  batch x glimpses x v_num x dim
        logits = h_ @ q_.transpose(2,3) # batch x glimpses x v_num x q_num
        logits = logits + self.h_bias

        # apply v_mask, q_mask
        logits.masked_fill_(v_mask.unsqueeze(1).unsqueeze(3).expand(logits.shape) == 0, -float('inf'))
        logits.masked_fill_(q_mask.unsqueeze(1).unsqueeze(2).expand(logits.shape) == 0, -float('inf'))

        atten = F.softmax(logits.view(-1, self.glimpses, v_num * q_num), 2)
        return atten.view(-1, self.glimpses, v_num, q_num), logits


class ApplyAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, glimpses, num_obj, count=None, drop=0.0):
        super(ApplyAttention, self).__init__()
        self.glimpses = glimpses
        layers = []
        for g in range(self.glimpses):
            layers.append(ApplySingleAttention(v_features, q_features, mid_features, num_obj, count, drop))
        self.glimpse_layers = nn.ModuleList(layers)
    
    def forward(self, v, q, b, v_mask, q_mask, atten, logits):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x glimpses x v_num x q_num
        logits:  batch x glimpses x v_num x q_num
        """
        for g in range(self.glimpses):
            atten_h, count_h = self.glimpse_layers[g](v, q, b, v_mask, q_mask, atten[:,g,:,:], logits[:,g,:,:])
            # residual (in original paper)
            q = q + atten_h + count_h
        q = q * q_mask.unsqueeze(2)
        return q.sum(1)

class ApplySingleAttention(nn.Module):
    def __init__(self, v_features, q_features, mid_features, num_obj, count, drop=0.0):
        super(ApplySingleAttention, self).__init__()
        self.count = count
        self.lin_v = FCNet(v_features, mid_features, drop=drop)  # let self.lin take care of bias
        self.lin_q = FCNet(q_features, mid_features, drop=drop)
        self.lin_atten = FCNet(mid_features, mid_features, relu=False, drop=drop)
        self.lin_count = FCNet(num_obj + 1, mid_features, drop=0)
        
    def forward(self, v, q, b, v_mask, q_mask, atten, logits):
        """
        v = batch, num_obj, dim
        q = batch, que_len, dim
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        atten:  batch x v_num x q_num
        logits:  batch x v_num x q_num
        """

        # apply single glimpse attention
        v_ = self.lin_v(v).transpose(1,2).unsqueeze(2) # batch, dim, 1, num_obj
        q_ = self.lin_q(q).transpose(1,2).unsqueeze(3) # batch, dim, que_len, 1
        v_ = v_ @ atten.unsqueeze(1) # batch, dim, 1, que_len
        h_ = v_ @ q_ # batch, dim, 1, 1
        h_ = h_.squeeze(3).squeeze(2) # batch, dim
        atten_h = self.lin_atten(h_).unsqueeze(1)

        # counting module
        count_embed = self.count(b, logits.max(2)[0])
        count_h = self.lin_count(count_embed).unsqueeze(1)

        return atten_h, count_h
