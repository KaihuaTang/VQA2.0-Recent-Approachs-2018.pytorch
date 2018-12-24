import json
import sys
import os.path
from collections import defaultdict

import numpy as np
import torch

import utils
import config


q_path = utils.path_for(val=True, question=True)
with open(q_path, 'r') as fd:
    q_json = json.load(fd)
a_path = utils.path_for(val=True, answer=True)
with open(a_path, 'r') as fd:
    a_json = json.load(fd)
with open(os.path.join(config.qa_path, 'v2_mscoco_val2014_complementary_pairs.json')) as fd:
    pairs = json.load(fd)

question_list = q_json['questions']
question_ids = [q['question_id'] for q in question_list]
questions = [q['question'] for q in question_list]
answer_list = a_json['annotations']
categories = [a['answer_type'] for a in answer_list]  # {'yes/no', 'other', 'number'}
accept_condition = {
    'number': (lambda x: id_to_cat[x] == 'number'),
    'count': (lambda x: id_to_question[x].lower().startswith('how many')),
    'all': (lambda x: True),
}

statistics = defaultdict(list)
for path in sys.argv[1:]:
    log = torch.load(path)
    ans = log['eval']
    d = [(acc, ans) for (acc, ans, _) in sorted(zip(ans['accuracies'], ans['answers'], ans['idx']), key=lambda x: x[-1])]
    accs = map(lambda x: x[0], d)
    id_to_cat = dict(zip(question_ids, categories))
    id_to_acc = dict(zip(question_ids, accs))
    id_to_question = dict(zip(question_ids, questions))

    for name, f in accept_condition.items():
        for on_pairs in [False, True]:
            acc = []
            if on_pairs:
                for a, b in pairs:
                    if not (f(a) and f(b)):
                        continue
                    if id_to_acc[a] == id_to_acc[b] == 1:
                        acc.append(1)
                    else:
                        acc.append(0)
            else:
                for x in question_ids:
                    if not f(x):
                        continue
                    acc.append(id_to_acc[x])
            acc = np.mean(acc)
            statistics[name, 'pair' if on_pairs else 'single'].append(acc)

for (name, pairness), accs in statistics.items():
    mean = np.mean(accs)
    std = np.std(accs, ddof=1)
    print('{} ({})\t: {:.2f}% +- {}'.format(name, pairness, 100 * mean, 100 * std))
