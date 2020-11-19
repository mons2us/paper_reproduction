## Overview
This is an implementation of AnoGAN using pytorch,<br>
as a reproduction of the paper \<
Unsupervised Anomaly Detection with Generative Adversarial Networks to Guide Marker Discovery\>, Thomas Schlegl et al., 2017. [paper link](https://arxiv.org/abs/1703.05921)<br><br>
To read my review on the paper(_written in Korean_), please follow the [link](https://mons2us.github.io/paper-reproduction/deeplearning/2020/11/14/anogan.html) to my blog.

## Installation
```bash
$ git clone https://github.com/mons2us/paper_reproduction.git
$ cd {dir}/AnoGAN
# This implementation was made under python 3.7
```

## Usage
```bash
usage: main.py [-h] [--mode MODE] [--data_type DATA_TYPE] [--batch_size N] [--train_label N]
               [--test_label N] [--epochs N] [--use_cuda] [--log_interval N]
               [--model_pth MODEL_PTH] [--plot_pth PLOT_PTH]

AnoGAN Implementation

optional arguments:
  -h, --help            show this help message and exit
  --mode MODE
  --data_type DATA_TYPE
                        Data type to train or test: CIFAR10 available
  --batch_size N        Batch size to be used for training (default: 128)
  --train_label N       Label to be used in training DCGAN (considered as normal data)
  --test_label N        Normal label and anomaly label
  --epochs N            Number of epochs to be used for training (default: 10)
  --use_cuda            Whether to use cuda in training. If you don't want to use cuda, set this
                        to False
  --log_interval N      Logging interval in training (default: 10)
  --model_pth MODEL_PTH
                        Path for the model to be saved or loaded from. Default is ./model; If
                        using svm loss, model will automatically be saved in ./model with name:
                        model_cnn_svm.pkl
  --plot_pth PLOT_PTH   Path for the result plot to be saved at. Default is ./plot
```

For training DCGAN,
```bash
$ python main.py --mode=train \
                 --epochs=30 \
                 --batch_size=512 \
                 --train_label=0 \
                 --test_label="0,7"
```

For performing anomaly detection,
```bash
$ python main.py --mode=anogan \
                 --train_label=0 \
                 --test_label="0,7"
```

