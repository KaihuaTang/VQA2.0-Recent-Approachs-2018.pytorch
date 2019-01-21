import os
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms

import config

def process_answer(answer):
    """
    follow Bilinear Attention Networks 
    and https://github.com/hengyuan-hu/bottom-up-attention-vqa
    """
    answer = answer.float() * 0.3
    answer = torch.clamp(answer, 0, 1)
    return answer

def batch_accuracy(logits, labels):
    """
    follow Bilinear Attention Networks https://github.com/jnhwkim/ban-vqa.git
    """
    logits = torch.max(logits, 1)[1].data # argmax
    one_hots = torch.zeros(*labels.size()).cuda()
    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = (one_hots * labels)
    
    return scores.sum(1)

def calculate_loss(answer, pred, method):
    """
    answer = [batch, 3129]
    pred = [batch, 3129]
    """
    if method == 'binary_cross_entropy_with_logits':
        loss = F.binary_cross_entropy_with_logits(pred, answer) * config.max_answers
    elif method == 'soft_cross_entropy':
        nll = -F.log_softmax(pred, dim=1)
        loss = (nll * answer).sum(dim=1).mean()   # this is worse than binary_cross_entropy_with_logits
    elif method == 'KL_divergence':
        pred = F.softmax(pred, dim=1)
        kl = ((answer / (pred + 1e-12)) + 1e-12).log()
        loss = (kl * answer).sum(1).mean()
    elif method == 'multi_label_soft_margin':
        loss = F.multilabel_soft_margin_loss(pred, answer)
    else:
        print('Error, pls define loss function')
    return loss

def path_for(train=False, val=False, test=False, question=False, answer=False):
    assert train + val + test == 1
    assert question + answer == 1

    if train:
        split = 'train2014'
    elif val:
        split = 'val2014'
    else:
        split = config.test_split

    if question:
        fmt = 'v2_{0}_{1}_{2}_questions.json'
    else:
        if test:
            # just load validation data in the test=answer=True case, will be ignored anyway
            split = 'val2014'
        fmt = 'v2_{1}_{2}_annotations.json'
    s = fmt.format(config.task, config.dataset, split)
    return os.path.join(config.qa_path, s)


def print_lr(optimizer, prefix, epoch):
    all_rl = []
    for p in optimizer.param_groups:
        all_rl.append(p['lr'])
    print('{} E{:03d}:'.format(prefix, epoch), ' Learning Rate: ', set(all_rl))

def set_lr(optimizer, value):
    for p in optimizer.param_groups:
        p['lr'] = value

def decay_lr(optimizer, rate):
    for p in optimizer.param_groups:
        p['lr'] *= rate


def print_grad(named_parameters):
    """
    visualize grad
    """

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)

    print('---Total norm {:.3f} -----------------'.format(total_norm))
    for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
    print('-------------------------------', flush=True)

    return total_norm


class Tracker:
    """ Keep track of results over time, while having access to monitors to display information about them. """
    def __init__(self):
        self.data = {}

    def track(self, name, *monitors):
        """ Track a set of results with given monitors under some name (e.g. 'val_acc').
            When appending to the returned list storage, use the monitors to retrieve useful information.
        """
        l = Tracker.ListStorage(monitors)
        self.data.setdefault(name, []).append(l)
        return l

    def to_dict(self):
        # turn list storages into regular lists
        return {k: list(map(list, v)) for k, v in self.data.items()}


    class ListStorage:
        """ Storage of data points that updates the given monitors """
        def __init__(self, monitors=[]):
            self.data = []
            self.monitors = monitors
            for monitor in self.monitors:
                setattr(self, monitor.name, monitor)

        def append(self, item):
            for monitor in self.monitors:
                monitor.update(item)
            self.data.append(item)

        def __iter__(self):
            return iter(self.data)

    class MeanMonitor:
        """ Take the mean over the given values """
        name = 'mean'

        def __init__(self):
            self.n = 0
            self.total = 0

        def update(self, value):
            self.total += value
            self.n += 1

        @property
        def value(self):
            return self.total / self.n

    class MovingMeanMonitor:
        """ Take an exponentially moving mean over the given values """
        name = 'mean'

        def __init__(self, momentum=0.9):
            self.momentum = momentum
            self.first = True
            self.value = None

        def update(self, value):
            if self.first:
                self.value = value
                self.first = False
            else:
                m = self.momentum
                self.value = m * self.value + (1 - m) * value
