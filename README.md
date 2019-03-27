# Several Recent Approaches (2018) on VQA v2

The project is based on [Cyanogenoid/vqa-counting][0]. Most of the current VQA2.0 projects are based on [https://github.com/hengyuan-hu/bottom-up-attention-vqa][1], while I personally prefer the Cyanogenoid's framework, because it's very clean and clear. So I reimplement several recent approaches including :
- [bottom-up top-down][2], 
- [bilinear attention network][3], 
- [Intra- and Inter-modality Attention][4] 
- [learning to count][5]
- [Learning Conditioned Graph Structures][6]

One of the benefit of our framework is that you can easily add counting module into your own model, which is proved to be effictive in imporving counting questions without harm the performance of your own model. 

## Dependencies

- Python 3.6
  - torch > 0.4
  - torchvision 0.2
  - h5py 2.7
  - tqdm 4.19

## Prepare dataset (Follow[Cyanogenoid/vqa-counting][0])
- In the `data` directory, execute `./download.sh` to download VQA v2.
  - For experimenting, using 36 fixed proposals is faster, at the expense of a bit of accuracy. Uncomment the relevant lines in `download.sh` and change the paths in `config.py` accordingly. Don't forget to set `output_size` in there to 36 to actually get the speed-up.
- Prepare the data by running
```
python preprocess-images.py
python preprocess-vocab.py
```
This creates an `h5py` database (95 GiB) containing the object proposal features and a vocabulary for questions and answers at the locations specified in `config.py`.

## How to Train

All the models are named as XXX_model.py, and most of the parameters is under config.py. To change the model, simply change model_type in config.py. Then train your model with:
```
python train.py [optional-name]
```
- To evaluate accuracy (VQA accuracy and balanced pair accuracy) in various categories, you can run
```
python eval-acc.py <path to .pth log> [<more paths to .pth logs> ...]
``` 

## Support training whole trainval split and generate result.json file for you to upload to the vqa2.0 online evaluation server

- First, I merge the question and annotation json for train and validation splits. You can download [trainval_annotation.json][7] and [trainval_question.json][8] from the links, and put them into ./data/ directory
- To train your model using the entire train & val sets, simply type the --trainval option during your training
```
python train.py --trainval
```
- To generate result.json file for you to upload to the vqa2.0 online evaluation server, you need to resume from the previous model trained from trainval split and select test. The generated result.json will be put into config.result_json_path
```
python train.py --test --resume=./logs/YOUR_MODEL.pth
```
- One More Thing: note that most of the methods require different learning rates when they train through the entire train&val splits. Usually, it's small than the learning rate that used to train on single train split.

## Model Details

Note that I didn't implement tfidf embedding of BAN model (though the current model has competitive/almost the same performance even without tfidf), only Glove Embedding is provided. About Intra- and Inter-modality Attention, Although I implemented all the details provided by the paper, it still seems not as good as the paper reported, even after I discussed with auther and made some modifications.

To Train [Counting Model][5]

Set following parameters in config.py:
```
model_type = 'counting'
```

To Train [Bottom-up Top-down][2]
```
model_type = 'baseline' 
```

To Train [Bilinear attention network][3]
```
model_type = 'ban' 
```
Note that BAN is very Memory Comsuming, so please ensure you got enough GPUs and run main.py with CUDA_VISIBLE_DEVICES=0,1,2,3

To Train [Intra- and Inter-modality Attention][4]
```
model_type = 'inter_intra' 
```
You may need to change the learning rate decay strategy as well from gradual_warmup_steps and lr_decay_epochs in config.py 

To Train [Learning Conditioned Graph Structures][6]
```
model_type = 'graph' 
```
Though this method seem less competitive. 

[0]: https://github.com/Cyanogenoid/vqa-counting
[1]: https://github.com/hengyuan-hu/bottom-up-attention-vqa
[2]: https://arxiv.org/abs/1707.07998
[3]: https://arxiv.org/abs/1805.07932
[4]: https://arxiv.org/abs/1812.05252
[5]: https://arxiv.org/abs/1802.05766
[6]: https://arxiv.org/abs/1806.07243
[7]: https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21768236&authkey=AHGPar-chbF0PuI
[8]: https://onedrive.live.com/embed?cid=22376FFAD72C4B64&resid=22376FFAD72C4B64%21768235&authkey=AJTII83FKtUN258
