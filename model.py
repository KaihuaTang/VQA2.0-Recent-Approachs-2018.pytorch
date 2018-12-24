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


class Net(nn.Module):
    """
    Implementation of Dynamic Fusion with Intra- and Inter-modality Attention Flow for Visual Question Answering (DFAF)
    Based on code from https://github.com/Cyanogenoid/vqa-counting
    """
    def __init__(self, words_list):
        super(Net, self).__init__()
        self.question_features = 1280
        self.vision_features = config.output_features
        self.hidden_features = 512
        self.num_inter_head = 8
        self.num_intra_head = 8
        self.num_block = 2
        self.visual_normalization = True

        assert(self.hidden_features % self.num_inter_head == 0)
        assert(self.hidden_features % self.num_intra_head == 0)
        words_list = list(words_list)
        words_list.insert(0, '__unknown__')

        self.text = word_embedding.TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=self.question_features,
            drop=0.1,
        )

        self.interIntraBlocks = SingleBlock(
            num_block=self.num_block,
            v_size=self.vision_features,
            q_size=self.question_features,
            output_size=self.hidden_features,
            num_inter_head=self.num_inter_head,
            num_intra_head=self.num_intra_head,
            drop=0.1,
        )

        self.classifier = Classifier(
            in_features=self.hidden_features,
            mid_features=1024,
            out_features=config.max_answers,
            drop=0.1,)

        for m in self.modules():
            if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
                init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, v, b, q, v_mask, q_mask):
        '''
        v: visual feature      [batch, 2048, num_obj]
        b: bounding box        [batch, 4, num_obj]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        # prepare v & q features
        v = v.transpose(1,2).contiguous()
        b = b.transpose(1,2).contiguous()
        q = self.text(q)  # [batch, max_len, 1280]
        if self.visual_normalization:
            v = v / (v.norm(p=2, dim=2, keepdim=True) + 1e-12).expand_as(v) # [batch, max_obj, 2048]
        v_mask = v_mask.float()
        q_mask = q_mask.float()
        
        v, q = self.interIntraBlocks(v, q, v_mask, q_mask)

        answer = self.classifier(v, q, v_mask, q_mask)

        return answer

class Fusion(nn.Module):
    """ Crazy multi-modal fusion: negative squared difference minus relu'd sum
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # found through grad student descent ;)
        return - (x - y)**2 + F.relu(x + y)

class ReshapeBatchNorm(nn.Module):
    def __init__(self, feat_size, affine=True):
        super(ReshapeBatchNorm, self).__init__()
        self.bn = nn.BatchNorm1d(feat_size, affine=affine)

    def forward(self, x):
        assert(len(x.shape) == 3)
        batch_size, num, _ = x.shape
        x = x.view(batch_size * num, -1)
        x = self.bn(x)
        return x.view(batch_size, num, -1)

class Classifier(nn.Sequential):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU()
        #self.fusion = Fusion()
        self.lin1 = nn.Linear(in_features, mid_features)
        self.lin2 = nn.Linear(mid_features, out_features)
        self.bn = nn.BatchNorm1d(mid_features)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, 512]
        q: question            [batch, max_len, 512]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        out = self.lin1(self.drop(v_mean * q_mean))
        out = self.lin2(self.drop(self.relu(self.bn(out))))
        return out

class SingleBlock(nn.Module):
    """
    Single Block Inter-/Intra-modality stack multiple times
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_inter_head, num_intra_head, drop=0.0):
        super(SingleBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_inter_head = num_inter_head
        self.num_intra_head = num_intra_head
        self.num_block = num_block

        self.v_lin = nn.Linear(v_size, output_size)
        self.q_lin = nn.Linear(q_size, output_size)

        self.interBlock = InterModalityUpdate(output_size, output_size, output_size, num_inter_head, drop)
        self.intraBlock = DyIntraModalityUpdate(output_size, output_size, output_size, num_intra_head, drop)

        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        # transfor features
        v = self.v_lin(self.drop(v))
        q = self.q_lin(self.drop(q))
        for i in range(self.num_block):
            v, q = self.interBlock(v, q, v_mask, q_mask)
            v, q = self.intraBlock(v, q, v_mask, q_mask)
        return v,q

class MultiBlock(nn.Module):
    """
    Multi Block Inter-/Intra-modality
    """
    def __init__(self, num_block, v_size, q_size, output_size, num_head, drop=0.0):
        super(MultiBlock, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head
        self.num_block = num_block

        blocks = []
        blocks.append(InterModalityUpdate(v_size, q_size, output_size, num_head, drop))
        blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        for i in range(num_block - 1):
            blocks.append(InterModalityUpdate(output_size, output_size, output_size, num_head, drop))
            blocks.append(DyIntraModalityUpdate(output_size, output_size, output_size, num_head, drop))
        self.multi_blocks = nn.ModuleList(blocks)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        for i in range(self.num_block):
            v, q = self.multi_blocks[i*2+0](v, q, v_mask, q_mask)
            v, q = self.multi_blocks[i*2+1](v, q, v_mask, q_mask)
        return v,q

class InterModalityUpdate(nn.Module):
    """
    Inter-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(InterModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size + v_size, output_size)
        self.q_output = nn.Linear(output_size + q_size, output_size)

        self.relu = nn.ReLU()
        self.drop = nn.Dropout(drop)

    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # transfor features
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply multi-head
        vk_set = torch.split(v_k, v_k.size(2) // self.num_head, dim=2)
        vq_set = torch.split(v_q, v_q.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(q_k, q_k.size(2) // self.num_head, dim=2)
        qq_set = torch.split(q_q, q_q.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            # inner product & set padding object/word attention to negative infinity & normalized by square root of hidden dimension
            q2v = (vq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, num_obj, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            v2q = (qq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, max_len, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            # softmax attention
            interMAF_q2v = F.softmax(q2v, dim=2) #[batch, num_obj, max_len]
            interMAF_v2q = F.softmax(v2q, dim=2) #[batch, max_len, num_obj]
            # calculate update input (each head of multi-head is calculated independently and concatenate together)
            v_update = interMAF_q2v @ qv_slice if (i==0) else torch.cat((v_update, interMAF_q2v @ qv_slice), dim=2)
            q_update = interMAF_v2q @ vv_slice if (i==0) else torch.cat((q_update, interMAF_v2q @ vv_slice), dim=2)
        # update new feature
        cat_v = torch.cat((v, v_update), dim=2)
        cat_q = torch.cat((q, q_update), dim=2)
        updated_v = self.v_output(self.drop(cat_v))
        updated_q = self.q_output(self.drop(cat_q))
        return updated_v, updated_q

class DyIntraModalityUpdate(nn.Module):
    """
    Dynamic Intra-modality Attention Flow
    """
    def __init__(self, v_size, q_size, output_size, num_head, drop=0.0):
        super(DyIntraModalityUpdate, self).__init__()
        self.v_size = v_size
        self.q_size = q_size
        self.output_size = output_size
        self.num_head = num_head

        self.v4q_gate_lin = nn.Linear(v_size, output_size)
        self.q4v_gate_lin = nn.Linear(q_size, output_size)

        self.v_lin = nn.Linear(v_size, output_size * 3)
        self.q_lin = nn.Linear(q_size, output_size * 3)

        self.v_output = nn.Linear(output_size, output_size)
        self.q_output = nn.Linear(output_size, output_size)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.drop = nn.Dropout(drop)
    
    def forward(self, v, q, v_mask, q_mask):
        """
        v: visual feature      [batch, num_obj, feat_size]
        q: question            [batch, max_len, feat_size]
        v_mask                 [batch, num_obj]
        q_mask                 [batch, max_len]
        """
        batch_size, num_obj = v_mask.shape
        _         , max_len = q_mask.shape
        # conditioned gating vector
        v_mean = (v * v_mask.unsqueeze(2)).sum(1) / v_mask.sum(1).unsqueeze(1)
        q_mean = (q * q_mask.unsqueeze(2)).sum(1) / q_mask.sum(1).unsqueeze(1)
        v4q_gate = self.sigmoid(self.v4q_gate_lin(self.drop(self.relu(v_mean)))).unsqueeze(1) #[batch, 1, feat_size]
        q4v_gate = self.sigmoid(self.q4v_gate_lin(self.drop(self.relu(q_mean)))).unsqueeze(1) #[batch, 1, feat_size]

        # key, query, value
        v_trans = self.v_lin(self.drop(self.relu(v)))
        q_trans = self.q_lin(self.drop(self.relu(q)))
        # mask all padding object/word features
        v_trans = v_trans * v_mask.unsqueeze(2)
        q_trans = q_trans * q_mask.unsqueeze(2)
        # split for different use of purpose
        v_k, v_q, v_v = torch.split(v_trans, v_trans.size(2) // 3, dim=2)
        q_k, q_q, q_v = torch.split(q_trans, q_trans.size(2) // 3, dim=2)
        # apply conditioned gate
        new_vq = (1 + q4v_gate) * v_q
        new_vk = (1 + q4v_gate) * v_k
        new_qq = (1 + v4q_gate) * q_q
        new_qk = (1 + v4q_gate) * q_k

        # apply multi-head
        vk_set = torch.split(new_vk, new_vk.size(2) // self.num_head, dim=2)
        vq_set = torch.split(new_vq, new_vq.size(2) // self.num_head, dim=2)
        vv_set = torch.split(v_v, v_v.size(2) // self.num_head, dim=2)
        qk_set = torch.split(new_qk, new_qk.size(2) // self.num_head, dim=2)
        qq_set = torch.split(new_qq, new_qq.size(2) // self.num_head, dim=2)
        qv_set = torch.split(q_v, q_v.size(2) // self.num_head, dim=2)
        # multi-head
        for i in range(self.num_head):
            vk_slice, vq_slice, vv_slice = vk_set[i], vq_set[i], vv_set[i]  #[batch, num_obj, feat_size]
            qk_slice, qq_slice, qv_slice = qk_set[i], qq_set[i], qv_set[i]  #[batch, max_len, feat_size]
            # calculate attention
            v2v = (vq_slice @ vk_slice.transpose(1,2)).masked_fill(v_mask.unsqueeze(1).expand([batch_size, num_obj, num_obj]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            q2q = (qq_slice @ qk_slice.transpose(1,2)).masked_fill(q_mask.unsqueeze(1).expand([batch_size, max_len, max_len]) == 0, -1e9) / ((self.output_size // self.num_head) ** 0.5)
            dyIntraMAF_v2v = F.softmax(v2v, dim=2)
            dyIntraMAF_q2q = F.softmax(q2q, dim=2)
            # calculate update input
            v_update = dyIntraMAF_v2v @ vv_slice if (i==0) else torch.cat((v_update, dyIntraMAF_v2v @ vv_slice), dim=2)
            q_update = dyIntraMAF_q2q @ qv_slice if (i==0) else torch.cat((q_update, dyIntraMAF_q2q @ qv_slice), dim=2)
        # update
        updated_v = self.v_output(self.drop(v + v_update))
        updated_q = self.q_output(self.drop(q + q_update))
        return updated_v, updated_q
