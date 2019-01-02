# paths
qa_path = 'data'  # directory containing the question and annotation jsons
bottom_up_trainval_path = 'data/trainval'  # directory containing the .tsv file(s) with bottom up features
bottom_up_test_path = 'data/test2015'  # directory containing the .tsv file(s) with bottom up features
preprocessed_trainval_path = 'genome-trainval.h5'  # path where preprocessed features from the trainval split are saved to and loaded from
preprocessed_test_path = 'genome-test.h5'  # path where preprocessed features from the test split are saved to and loaded from
vocabulary_path = 'vocab.json'  # path where the used vocabularies for question and answers are saved to

task = 'OpenEnded'
dataset = 'mscoco'

test_split = 'test2015'  # either 'test-dev2015' or 'test2015'

# preprocess config
output_size = 100  # max number of object proposals per image
output_features = 2048  # number of features in each object proposal

# training config
epochs = 100
batch_size = 256
initial_lr = 1e-3
lr_decay_step = 5
lr_decay_rate = 0.25
lr_halflife = 50000 # for scheduler
data_workers = 4
max_answers = 3129
max_q_length = 999 # question_length = min(max_q_length, max_length_in_dataset)
clip_value = 0.25
v_feat_norm = False
seed = 5225

model_type = 'baseline'           # "Bottom-up top-down"
#model_type = 'inter_intra'       # "Intra- and Inter-modality Attention" 
#model_type = 'ban'               # "Bilinear Attention Network"
#model_type = 'counting'

optim_method = 'Adamax'           # used in "Bottom-up top-down", "Bilinear Attention Network", "Intra- and Inter-modality Attention" 
#optim_method = 'Adam'            # used in "Learning to count objects", set initial_lr to 1.5e-3

gradual_warmup_steps = [0.5 * initial_lr, 1.0 * initial_lr, 1.5 * initial_lr, 2.0 * initial_lr]
lr_decay_epochs = range(10, 40, lr_decay_step)

def print_param():
    print('Num obj: ', output_size)
    print('Num epochs: ', epochs)
    print('Batch size: ', batch_size)
    print('Model type: ', model_type)
    print('Optimization Method: ', optim_method)
    print('Clip Value: ', clip_value)
    print('Init LR: ', initial_lr)
    print('LR decay step: ', lr_decay_step)
    print('LR decay rate: ', lr_decay_rate)
    print('LR half life: ', lr_halflife)
    print('Normalize visual feature: ', v_feat_norm)
    print('Max answer choice: ', max_answers)
    print('Manually set max question lenght: ', max_q_length)
    print('Random Seed: ', seed)
    