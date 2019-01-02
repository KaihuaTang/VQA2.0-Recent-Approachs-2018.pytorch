# Counting component on VQA v2

This directory contains code for training and evaluating a model using the counting component on the VQA v2 dataset.
The code is loosely based on [this implementation][0].
You can find additional resources for download (i.e. poster of the paper, pre-trained weights, results.json file) in [this release](https://github.com/Cyanogenoid/vqa-counting/releases/tag/resources).
The results have been [independently reproduced][5] by Shagun Sodhani and Vardaan Pahuja.

## Instructions

- In the `data` directory, execute `./download.sh` to download VQA v2 [questions, answers, and bottom-up features][4].
  - For experimenting, using 36 fixed proposals is faster, at the expense of a bit of accuracy. Uncomment the relevant lines in `download.sh` and change the paths in `config.py` accordingly. Don't forget to set `output_size` in there to 36 to actually get the speed-up.
- Prepare the data by running
```
python preprocess-images.py
python preprocess-vocab.py
```
This creates an `h5py` database (95 GiB) containing the object proposal features and a vocabulary for questions and answers at the locations specified in `config.py`.
- Train the model in `model.py` with:
```
python train.py [optional-name]
```
This will alternate between one epoch of training on the train split and one epoch of validation on the validation split while printing the current training progress to stdout and saving logs in the `logs` directory.
The logs contain the name of the model, training statistics, contents of `config.py`,  model weights, evaluation information (per-question answer and accuracy), and question and answer vocabularies.
- To view training progression of a model that is currently or has finished training.
```
python view-log.py <path to .pth log>
```

- To evaluate accuracy (VQA accuracy and balanced pair accuracy; see paper for details) in various categories, you can run
```
python eval-acc.py <path to .pth log> [<more paths to .pth logs> ...]
```
If you pass in multiple paths as arguments, this gives you standard deviations as well.
To customise what categories are shown, you can modify the "accept conditions" for categories in `eval-acc.py`.

## Other things you can do
- If you want to evaluate on the official test server, run
```
python preprocess-images.py --test
```
to create the feature database for the test split, then `train.py` with `--test` and `--resume` arguments.
- `train.py` supports some more arguments:
  - `--resume <path to .pth log>` starts the training procedure with the weights initialised from the weights stored in the given log file.
  - `--eval` does not train a model, but only evaluates on the validation split. Probably only useful with `--resume`.
  - `--test` does the same as `--eval`, but uses the test split specified in `config.py` instead and outputs a `results.json` file ready to be uploaded to the test server. Also probably only useful with `--resume`.
- Training on both the `train` and `val` splits for evaluation on the test server is now supported in this.
Switch to the trainval branch of this repository, run `preprocess-vocab.py` to have the vocabulary depend on training and validation data and train a model with `train.py`.
Evaluation on test-dev and test works as before with `train.py`.
- The baseline model is obtained by commenting out the line in the classifier that merges the count features back in.
The NMS baseline model is obtained by copying the [NMS implementation from here][1] into this directory and [building it][4], removing the `forward` function from the `Net` class in `model.py` and putting [these two functions][2] there instead.
It is probably a good idea to do these in different branches so that you can switch between the different models easily; that is what I am doing at least.

## Dependencies

This code was confirmed to run with the following environment:

- Python 3.6.2
  - torch 0.4.0
  - torchvision 0.2
  - h5py 2.7.0
  - tqdm 4.19.2


[0]: https://github.com/Cyanogenoid/pytorch-vqa
[1]: https://github.com/ruotianluo/pytorch-faster-rcnn/tree/master/lib/nms
[2]: https://gist.github.com/anonymous/2701c0964712e0a7fcce64ea752e391a
[4]: https://github.com/ruotianluo/pytorch-faster-rcnn#installation
[5]: https://arxiv.org/abs/1805.08174
