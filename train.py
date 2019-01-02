import sys
import os.path
import argparse
import math
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.nn.utils import clip_grad_norm_
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
from tqdm import tqdm

import config
import data
if config.model_type == 'baseline':
    import baseline_model as model
elif config.model_type == 'inter_intra':
    import inter_intra_model as model
elif config.model_type == 'ban':
    import ban_model as model
elif config.model_type == 'counting':
    import counting_model as model
import utils

def run(net, loader, optimizer, scheduler, tracker, train=False, has_answers=True, prefix='', epoch=0):
    """ Run an epoch over the given loader """
    assert not (train and not has_answers)
    if train:
        net.train()
        tracker_class, tracker_params = tracker.MovingMeanMonitor, {'momentum': 0.99}
    else:
        net.eval()
        tracker_class, tracker_params = tracker.MeanMonitor, {}
        answ = []
        idxs = []
        accs = []

    # set learning rate decay policy
    if epoch < len(config.gradual_warmup_steps) and config.optim_method == 'Adamax':
        utils.set_lr(optimizer, config.gradual_warmup_steps[epoch])
        utils.print_lr(optimizer, prefix, epoch)
    elif (epoch in config.lr_decay_epochs) and train and config.optim_method == 'Adamax':
        utils.decay_lr(optimizer, config.lr_decay_rate)
        utils.print_lr(optimizer, prefix, epoch)
    else:
        utils.print_lr(optimizer, prefix, epoch)

    loader = tqdm(loader, desc='{} E{:03d}'.format(prefix, epoch), ncols=0)
    loss_tracker = tracker.track('{}_loss'.format(prefix), tracker_class(**tracker_params))
    acc_tracker = tracker.track('{}_acc'.format(prefix), tracker_class(**tracker_params))

    for v, q, a, b, idx, v_mask, q_mask, q_len in loader:
        var_params = {
            'requires_grad': False,
        }
        v = Variable(v.cuda(), **var_params)
        q = Variable(q.cuda(), **var_params)
        a = Variable(a.cuda(), **var_params)
        b = Variable(b.cuda(), **var_params)
        q_len = Variable(q_len.cuda(), **var_params)
        v_mask = Variable(v_mask.cuda(), **var_params)
        q_mask = Variable(q_mask.cuda(), **var_params)

        out = net(v, b, q, v_mask, q_mask, q_len)
        if has_answers:
            answer = utils.process_answer(a)
            #nll = -F.log_softmax(out, dim=1)
            #loss = (nll * answer).sum(dim=1).mean()   # this is worse than binary_cross_entropy_with_logits
            loss = F.binary_cross_entropy_with_logits(out, answer) * config.max_answers
            acc = utils.batch_accuracy(out, answer).data.cpu()

        if train:
            optimizer.zero_grad()
            loss.backward()
            # clip gradient
            clip_grad_norm_(net.parameters(), config.clip_value)
            optimizer.step()
            if (config.optim_method == 'Adam'): 
                scheduler.step()
        else:
            # store information about evaluation of this minibatch
            _, answer = out.data.cpu().max(dim=1)
            answ.append(answer.view(-1))
            if has_answers:
                accs.append(acc.view(-1))
            idxs.append(idx.view(-1).clone())

        if has_answers:
            loss_tracker.append(loss.item())
            acc_tracker.append(acc.mean())
            fmt = '{:.4f}'.format
            loader.set_postfix(loss=fmt(loss_tracker.mean.value), acc=fmt(acc_tracker.mean.value))

    if not train:
        answ = list(torch.cat(answ, dim=0))
        if has_answers:
            accs = list(torch.cat(accs, dim=0))
        else:
            accs = []
        idxs = list(torch.cat(idxs, dim=0))
        #print('{} E{:03d}:'.format(prefix, epoch), ' Total num: ', len(accs))
        #print('{} E{:03d}:'.format(prefix, epoch), ' Average Score: ', float(sum(accs) / len(accs)))
        return answ, accs, idxs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('name', nargs='*')
    parser.add_argument('--eval', dest='eval_only', action='store_true')
    parser.add_argument('--test', action='store_true')
    parser.add_argument('--resume', nargs='*')
    parser.add_argument('--describe', type=str, default='describe your setting')
    args = parser.parse_args()

    print('-'*50)
    print(args)
    config.print_param()

    # set mannual seed
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed(config.seed)

    if args.test:
        args.eval_only = True
    src = open(config.model_type+'_model.py').read()
    if args.name:
        name = ' '.join(args.name)
    else:
        from datetime import datetime
        name = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
    target_name = os.path.join('logs', '{}.pth'.format(name))
    if not args.test:
        # target_name won't be used in test mode
        print('will save to {}'.format(target_name))
    if args.resume:
        logs = torch.load(' '.join(args.resume))
        # hacky way to tell the VQA classes that they should use the vocab without passing more params around
        data.preloaded_vocab = logs['vocab']

    cudnn.benchmark = True

    if not args.eval_only:
        train_loader = data.get_loader(train=True)
    if not args.test:
        val_loader = data.get_loader(val=True)
    else:
        val_loader = data.get_loader(test=True)

    net = model.Net(val_loader.dataset.vocab['question'].keys())
    net = nn.DataParallel(net).cuda()
    select_optim = optim.Adamax if (config.optim_method == 'Adamax') else optim.Adam
    optimizer = select_optim([p for p in net.parameters() if p.requires_grad], lr=config.initial_lr)
    scheduler = lr_scheduler.ExponentialLR(optimizer, 0.5**(1 / config.lr_halflife))
    if args.resume:
        net.module.load_state_dict(logs['weights'])

    tracker = utils.Tracker()
    config_as_dict = {k: v for k, v in vars(config).items() if not k.startswith('__')}

    for i in range(config.epochs):
        if not args.eval_only:
            run(net, train_loader, optimizer, scheduler, tracker, train=True, prefix='train', epoch=i)
        r = run(net, val_loader, optimizer, scheduler, tracker, train=False, prefix='val', epoch=i, has_answers=not args.test)
        
        if not args.test:
            results = {
                'name': name,
                'tracker': tracker.to_dict(),
                'config': config_as_dict,
                'weights': net.module.state_dict(),
                'eval': {
                    'answers': r[0],
                    'accuracies': r[1],
                    'idx': r[2],
                },
                'vocab': val_loader.dataset.vocab,
                'src': src,
            }
            torch.save(results, target_name)
        else:
            # in test mode, save a results file in the format accepted by the submission server
            answer_index_to_string = {a:  s for s, a in val_loader.dataset.answer_to_index.items()}
            results = []
            for answer, index in zip(r[0], r[2]):
                answer = answer_index_to_string[answer.item()]
                qid = val_loader.dataset.question_ids[index]
                entry = {
                    'question_id': qid,
                    'answer': answer,
                }
                results.append(entry)
            with open('results.json', 'w') as fd:
                json.dump(results, fd)

        if args.eval_only:
            break

if __name__ == '__main__':
    main()
