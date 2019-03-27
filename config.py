# paths
qa_path = 'data'  # directory containing the question and annotation jsons
bottom_up_trainval_path = 'data/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = 'data/test2015'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_test_path = 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to
glove_index = 'data/dictionary.pkl'
result_json_path = 'results.json'  # the path to save the test json that can be uploaded to vqa2.0 online evaluation server

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # always 'test2015' since from 2018, vqa online evaluation server requires to upload entire test2015 result even for test-dev split

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

###################################################################
#              Default Setting for All Model
###################################################################
# training config
epochs = 100
batch_size = 256
initial_lr = 1e-3
lr_decay_step = 2
lr_decay_rate = 0.25
lr_halflife = 50000 # for scheduler (counting)
data_workers = 4
max_answers = 3129
max_q_length = 666 # question_length = min(max_q_length, max_length_in_dataset)
clip_value = 0.25
v_feat_norm = False # Only useful in learning to count
print_gradient = False
normalize_box = False
seed = 5225
weight_decay = 0.0

model_type = 'baseline'           # "Bottom-up top-down"
#model_type = 'inter_intra'       # "Intra- and Inter-modality Attention" 
#model_type = 'ban'               # "Bilinear Attention Network"
#model_type = 'counting'          # "Learning to count objects"
#model_type = 'graph'             # "Learning Conditioned Graph Structures for Interpretable Visual Question Answering"

optim_method = 'Adamax'           # used in "Bottom-up top-down", "Bilinear Attention Network", "Intra- and Inter-modality Attention" 
#optim_method = 'Adam'            # used in "Learning to count objects", set initial_lr to 1.5e-3

schedule_method = 'warm_up'
#schedule_method = 'batch_decay'

loss_method = 'binary_cross_entropy_with_logits'
#loss_method = 'soft_cross_entropy'
#loss_method = 'KL_divergence'
#loss_method = 'multi_label_soft_margin'

gradual_warmup_steps = [1.0 * initial_lr, 1.0 * initial_lr, 2.0 * initial_lr, 2.0 * initial_lr]
lr_decay_epochs = range(10, 100, lr_decay_step)

###################################################################
#              Detailed Setting for Each Model
###################################################################

# "Bottom-up top-down"
# baseline Setting
if model_type == 'baseline':
    loss_method = 'binary_cross_entropy_with_logits'
    gradual_warmup_steps = [0.5 * initial_lr, 1.0 * initial_lr, 1.5 * initial_lr, 2.0 * initial_lr]

# "Intra- and Inter-modality Attention" 
# inter_intra setting
elif model_type == 'inter_intra':
    lr_decay_step = 10
    max_q_length = 14
    loss_method = 'binary_cross_entropy_with_logits'
    gradual_warmup_steps = [1.0 * initial_lr, 1.0 * initial_lr, 2.0 * initial_lr, 2.0 * initial_lr]

# "Bilinear Attention Network"
# ban setting
elif model_type == 'ban':
    batch_size = 128
    lr_decay_step = 2
    max_q_length = 14
    loss_method = 'binary_cross_entropy_with_logits'
    gradual_warmup_steps = [0.5 * initial_lr, 1.0 * initial_lr, 1.5 * initial_lr, 2.0 * initial_lr]

# "Learning to count objects"
# counting setting
elif model_type == 'counting':
    optim_method = 'Adam'
    schedule_method = 'batch_decay'
    v_feat_norm = True
    loss_method = 'soft_cross_entropy'

# "Learning Conditioned Graph Structures for Interpretable Visual Question Answering"
# graph setting
elif model_type == 'graph':
    initial_lr = 1e-4
    lr_decay_step = 10
    lr_decay_rate = 0.5
    normalize_box = True
    loss_method = 'multi_label_soft_margin'
    gradual_warmup_steps = [1.0 * initial_lr, 1.0 * initial_lr, 2.0 * initial_lr, 2.0 * initial_lr]
    lr_decay_epochs = range(30, 100, lr_decay_step)
    

def print_param():
    print('--------------------------------------------------')
    print('Num obj: ', output_size)
    print('Num epochs: ', epochs)
    print('Batch size: ', batch_size)
    print('Model type: ', model_type)
    print('Optimization Method: ', optim_method)
    print('Schedule Method: ', schedule_method)
    print('Loss Method: ', loss_method)
    print('Clip Value: ', clip_value)
    print('Init LR: ', initial_lr)
    print('LR decay step: ', lr_decay_step)
    print('LR decay rate: ', lr_decay_rate)
    print('LR half life: ', lr_halflife)
    print('Normalize visual feature: ', v_feat_norm)
    print('Print Gradient: ', print_gradient)
    print('Normalize Box Size: ', normalize_box)
    print('Max answer choice: ', max_answers)
    print('Manually set max question lenght: ', max_q_length)
    print('Random Seed: ', seed)
    print('gradual_warmup_steps: ', gradual_warmup_steps)
    print('Weight Decay: ', weight_decay)
    print('--------------------------------------------------')
    