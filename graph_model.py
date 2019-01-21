
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable
from torch.nn.utils import weight_norm
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.parameter import Parameter
import numpy as np
import config
import word_embedding

from reuse_modules import Fusion, FCNet

class Net(nn.Module):
    def __init__(self, words_list):
        super(Net, self).__init__()
        mid_features = 1024
        question_features = mid_features
        vision_features = config.output_features
        self.top_k_sparse = 16
        num_kernels = 8
        sparse_graph = True

        self.text = word_embedding.TextProcessor(
            classes=words_list,
            embedding_features=300,
            lstm_features=question_features,
            drop=0.0,
        )

        self.pseudo_coord = PseudoCoord()

        self.graph_learner = GraphLearner(
            v_features=vision_features+4, 
            q_features=question_features, 
            mid_features=512, 
            dropout=0.5,
            sparse_graph=sparse_graph,
        )

        self.graph_conv1 = GraphConv(
            v_features=vision_features+4, 
            mid_features=mid_features * 2, 
            num_kernels=num_kernels, 
            bias=False
        )

        self.graph_conv2 = GraphConv(
            v_features=mid_features*2, 
            mid_features=mid_features, 
            num_kernels=num_kernels, 
            bias=False
        )
    
        self.classifier = Classifier(
            in_features=mid_features,
            mid_features=mid_features*2,
            out_features=config.max_answers,
            drop=0.5,)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, v, b, q, v_mask, q_mask, q_len):
        '''
        v: visual feature      [batch, num_obj, 2048]
        b: bounding box        [batch, num_obj, 4]
        q: question            [batch, max_q_len]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        q_mask: question length [batch, max_len]   1 is word, 0 is none
        answer: predict logits [batch, config.max_answers]
        '''
        q = self.text(q, list(q_len.data))  # [batch, 1024]
        v = self.dropout(v)
        v = torch.cat((v, b), dim=2) # [batch, 2048+4]

        new_coord = self.pseudo_coord(b) #[batch, num_obj, num_obj, 2]
        adj_matrix, top_ind = self.graph_learner(v, q, v_mask, top_K=self.top_k_sparse) #[batch, num_obj, K]
        
        hid_v1 = self.graph_conv1(v, v_mask, new_coord, adj_matrix, top_ind, weight_adj=True)
        hid_v1 = self.dropout(self.relu(hid_v1))

        hid_v2 = self.graph_conv2(hid_v1, v_mask, new_coord, adj_matrix, top_ind, weight_adj=False)
        hid_v2 = self.relu(hid_v2) # [batch, num_obj, dim]

        #hid_v2 = hid_v2 * v_mask.unsqueeze(-1)
        max_pooled_v = torch.max(hid_v2, dim=1)[0] # [batch, dim]

        answer = self.classifier(max_pooled_v, q)
            
        return answer


class Classifier(nn.Module):
    def __init__(self, in_features, mid_features, out_features, drop=0.0):
        super(Classifier, self).__init__()
        self.lin1 = FCNet(in_features, mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, out_features, drop=drop)
        self.relu = nn.ReLU()

    def forward(self, v, q):
        x = v * self.relu(q)
        x = self.lin1(x)
        x = self.lin2(x)
        return x

class PseudoCoord(nn.Module):
    def __init__(self):
        super(PseudoCoord, self).__init__()

    def forward(self, b):
        '''
        Input: 
        b: bounding box        [batch, num_obj, 4]  (x1,y1,x2,y2)
        Output:
        pseudo_coord           [batch, num_obj, num_obj, 2] (rho, theta)
        '''
        batch_size, num_obj, _ = b.shape

        centers = (b[:,:,2:] + b[:,:,:2]) * 0.5

        relative_coord = centers.view(batch_size, num_obj, 1, 2) - \
                            centers.view(batch_size, 1, num_obj, 2)  # broadcast: [batch, num_obj, num_obj, 2]
        
        rho = torch.sqrt(relative_coord[:,:,:,0]**2 + relative_coord[:,:,:,1]**2)
        theta = torch.atan2(relative_coord[:,:,:,0], relative_coord[:,:,:,1])
        new_coord = torch.cat((rho.unsqueeze(-1), theta.unsqueeze(-1)), dim=-1)
        return new_coord

class GraphLearner(nn.Module):
    def __init__(self, v_features, q_features, mid_features, dropout=0.0, sparse_graph=True):
        super(GraphLearner, self).__init__()
        self.sparse_graph = sparse_graph
        self.lin1 = FCNet(v_features + q_features, mid_features, activate='relu')
        self.lin2 = FCNet(mid_features, mid_features, activate='relu')

    def forward(self, v, q, v_mask, top_K):
        '''
        Input:
        v: visual feature      [batch, num_obj, 2048]
        q: bounding box        [batch, 1024]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none

        Return:
        adjacent_logits        [batch, num_obj, K(sum=1)]
        adjacent_matrix        [batch, num_obj, K(sum=1)]
        '''
        batch_size, num_obj, _ = v.shape 
        q_repeated = q.unsqueeze(1).repeat(1, num_obj, 1)

        v_cat_q = torch.cat((v, q_repeated), dim=2)

        h = self.lin1(v_cat_q)
        h = self.lin2(h)
        h = h.view(batch_size, num_obj, -1)  # batch_size, num_obj, feat_size

        adjacent_logits = torch.matmul(h, h.transpose(1, 2)) # batch_size, num_obj, num_obj

        # object mask
        #mask = torch.matmul(v_mask.unsqueeze(2),  v_mask.unsqueeze(1))
        #adjacent_logits = adjacent_logits * mask
        # sparse adjacent matrix
        if self.sparse_graph:
            top_value, top_ind = torch.topk(adjacent_logits, k=top_K, dim=-1, sorted=False)  # batch_size, num_obj, K
        # softmax attention
        adjacent_matrix = F.softmax(top_value, dim=-1) # batch_size, num_obj, K

        return adjacent_matrix, top_ind

class GraphConv(nn.Module):
    def __init__(self, v_features, mid_features, num_kernels, bias=False):
        super(GraphConv, self).__init__()
        self.num_kernels = num_kernels
        # for graph conv
        self.conv_weights = nn.ModuleList([nn.Linear(
            v_features, mid_features//(num_kernels), bias=bias) for i in range(num_kernels)])
        # for gaussian kernels
        self.mean_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.mean_theta = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_rho = Parameter(torch.FloatTensor(num_kernels, 1))
        self.precision_theta = Parameter(torch.FloatTensor(num_kernels, 1))

        self.init_param()

    def init_param(self):
        self.mean_rho.data.uniform_(0, 1.0)
        self.mean_theta.data.uniform_(-np.pi, np.pi)
        self.precision_rho.data.uniform_(0, 1.0)
        self.precision_theta.data.uniform_(0, 1.0)

    def forward(self, v, v_mask, coord, adj_matrix, top_ind, weight_adj=True):
        """
        Input:
        v: visual feature      [batch, num_obj, 2048]
        v_mask: number of obj  [batch, max_obj]   1 is obj,  0 is none
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord
        adj_matrix: sparse     [batch, num_obj, K(sum=1)]
        top_ind:               [batch, num_obj, K]
        Output:
        v: visual feature      [batch, num_obj, dim]
        """
        batch_size, num_obj, feat_dim = v.shape
        K = adj_matrix.shape[-1]

        conv_v = v.unsqueeze(1).expand(batch_size, num_obj, num_obj, feat_dim) # batch_size, num_obj(same), num_obj(diff), feat_dim
        coord_weight = self.get_gaussian_weights(coord) # batch, num_obj, num_obj(diff), n_kernels

        slice_idx1 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K, feat_dim) # batch_size, num_obj, K, feat_dim
        slice_idx2 = top_ind.unsqueeze(-1).expand(batch_size, num_obj, K, self.num_kernels) # batch_size, num_obj, K, num_kernels
        sparse_v = torch.gather(conv_v, dim=2, index=slice_idx1)
        sparse_weight = torch.gather(coord_weight, dim=2, index=slice_idx2)
        if weight_adj:
            adj_mat = adj_matrix.unsqueeze(-1)  # batch, num_obj, K(sum=1), 1
            attentive_v = sparse_v * adj_mat # update feature : batch_size, num_obj, K(diff), feat_dim
        else:
            attentive_v = sparse_v       # update feature : batch_size, num_obj(same), K(diff), feat_dim
        
        weighted_neighbourhood = torch.matmul(sparse_weight.transpose(2, 3), attentive_v) # batch, num_obj, n_kernels, feat_dim
        weighted_neighbourhood = [self.conv_weights[i](weighted_neighbourhood[:, :, i, :]) for i in range(self.num_kernels)]  # each: batch, num_obj, feat_dim
        output = torch.cat(weighted_neighbourhood, dim=2)  # batch, num_obj(same), feat_dim

        return output

    def get_gaussian_weights(self, coord):
        """
        Input:
        coord: relative coord  [batch, num_obj, num_obj, 2]  obj to obj relative coord

        Output:
        weights                [batch, num_obj, num_obj, n_kernels)
        """
        batch_size, num_obj, _, _ = coord.shape
        # compute rho weights
        diff = (coord[:, :, :, 0].contiguous().view(-1, 1) - self.mean_rho.view(1, -1))**2  # batch*num_obj*num_obj,  n_kernels
        weights_rho = torch.exp(-0.5 * diff /
                                (1e-14 + self.precision_rho.view(1, -1)**2))  # batch*num_obj*num_obj,  n_kernels

        # compute theta weights
        first_angle = torch.abs(coord[:, :, :, 1].contiguous().view(-1, 1) - self.mean_theta.view(1, -1))
        second_angle = torch.abs(2 * np.pi - first_angle)
        weights_theta = torch.exp(-0.5 * (torch.min(first_angle, second_angle)**2)
                                  / (1e-14 + self.precision_theta.view(1, -1)**2))

        weights = weights_rho * weights_theta
        weights[(weights != weights).detach()] = 0

        # normalise weights
        weights = weights / (torch.sum(weights, dim=1, keepdim=True) + 1e-14) # batch*num_obj*num_obj,  n_kernels (sum=-1)

        return weights.view(batch_size, num_obj, num_obj, self.num_kernels)


# 1. weights Normalized on object dim
# 2. second time still use weight

"""

class GraphConv(nn.Module):
    def __init__(self, v_features, q_features, mid_features, output_features, num_head, sparse_graph, drop=0.0):
        super(GraphConv, self).__init__()
        self.num_head = num_head
        self.norm_term = (mid_features / num_head) ** 0.5
        self.sparse_graph = sparse_graph
        assert(v_features == output_features)
        self.lin_v = FCNet(v_features, mid_features, drop=drop) 
        self.lin_q = FCNet(q_features, mid_features, drop=drop)

        self.lin_pass_v = FCNet(v_features, output_features, drop=drop)
        self.lin_pass_q = FCNet(q_features, output_features, drop=drop)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, v, q, v_mask, sparse_num):
        #v = batch, num_obj, dim
        #q = batch, dim
        #sparse_num: if graph is sparse, how many neighbor will be included
        batch_size, num_obj, _ = v.shape
        # for discrete mask
        att_v = self.lin_v(v)
        att_q = self.lin_q(q)
        v_on_q = att_v * att_q.unsqueeze(1) #batch, num_obj, dim
        # multi-head
        v_on_q_splits = v_on_q.view(batch_size, num_obj, self.num_head, -1).transpose(1,2) # batch, num_head, num_obj, (dim // num_head)
        attention_logits = torch.matmul(v_on_q_splits, v_on_q_splits.transpose(2,3)) / self.norm_term #  batch, num_head, num_obj, num_obj
        attention_logits.masked_fill_(v_mask.unsqueeze(2).unsqueeze(1) @ v_mask.unsqueeze(1).unsqueeze(1) == 0, -float('inf'))
        if self.sparse_graph:
            # select topk neighbors for each object, the attention to rest neighbours will be 0
            attention_logits = attention_logits.view(-1, num_obj)
            _, topk_indices = torch.topk(attention_logits, sparse_num)
            mask = torch.zeros(attention_logits.shape).cuda().scatter_(1, topk_indices, 1)
            attention_logits.masked_fill_(mask==0, -float('inf'))
            attention_logits = attention_logits.view(batch_size, self.num_head, num_obj, num_obj)
        attentions = F.softmax(attention_logits, dim=3)
        print('Sparse Attention Example: ', attentions[1,1,1,:])

        # propagation
        pass_v = self.tanh(self.lin_pass_v(v)).view(batch_size, num_obj, self.num_head, -1) #batch, num_obj, num_head, (dim // num_head)
        gate_q = self.sigmoid(self.lin_pass_q(q)).view(batch_size, self.num_head, -1) #batch, num_head, (dim // num_head)

        pass_v = pass_v.transpose(1,2).unsqueeze(4)  #batch, num_head, num_obj, (dim // num_head), 1
        attentions = attentions.unsqueeze(3) # #batch, num_head, num_obj, 1, num_obj
        update = (pass_v * attentions).sum(4)  #batch, num_head, num_obj, (dim // num_head)

        gated_update = (update * gate_q.unsqueeze(2)).transpose(1,2).contiguous().view(batch_size, num_obj, -1)    #batch, num_obj, dim
        new_v = v + gated_update

        return new_v


class GraphPool(nn.Module):
    def __init__(self, v_features, q_features, mid_features, output_features, num_nodes, drop=0.0):
        super(GraphPool, self).__init__()
        self.lin_assign_v = FCNet(v_features, mid_features, drop=drop)
        self.lin_assign_q = FCNet(q_features, mid_features, drop=drop)
        self.lin_assign = FCNet(mid_features, num_nodes, drop=drop)

        self.lin_v = FCNet(v_features, output_features, activate='relu', drop=drop)


    def forward(self, v, q, v_mask):
        batch_size, _ = v.shape
        # for virtual nodes assignment
        assign_v = self.lin_assign_v(v)
        assign_q = self.lin_assign_q(q)
        v_on_q = assign_v * assign_q.unsqueeze(1) #batch, num_obj, dim

        assign = self.lin_assign(v_on_q) #batch, num_obj, assignment
        assign = F.softmax(assign, dim=2).transpose(1,2)  #batch, assignment, num_obj

        # value
        value_v = self.lin_v(v) * v_mask.unsqueeze(2) #batch, num_obj, dim

        output = assign @ value_v # batch, assignment, dim
        return output.view(batch_size, -1)

"""
