# Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Getting started

#### Prerequisites
* This repository is implemented and verified on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.7
```bash
$ conda create -n model_compression python=3.7
$ conda activate model_compression
```

#### Installation

0. Clone this repository.
```bash
$ git clone https://github.com/j-marple-dev/model_compression.git
$ cd model_compression
```

1. Install PyTorch 1.5.1 and Torchvision 0.6.1 (See the following official instruction).
```bash
$ conda install pytorch==1.5.1 torchvision==0.6.1 -c pytorch
```

2. Install `progressbar2`

```bash
$ conda install -c conda-forge progressbar2
```

###### For users
Install packages required to execute the code. Just type:
```bash
$ make dep
```

###### For developers

If you want to modify code you should configure formatting and linting settings. It automatically runs formatting and linting when you commit the code. Just type:
```bash
$ make dev
```

After having done `make dev`, you can validate the code by the following commands.
```bash
$ make format  # for formatting
$ make test  # for linting
```

## Usages

#### Run training
Training the model. Trainer supports the following options:
  - batch size, epoch numbers, seed
  - SGD: momentum, weight decay, initial learning rate, nesterov
  - Image Augmentation: [Autoaugment](https://arxiv.org/pdf/1805.09501.pdf), [Randaugment](https://arxiv.org/pdf/1909.13719.pdf), [CutMix](https://arxiv.org/pdf/1905.04899.pdf)
  - Loss: Cross Entropy + [Label Smoothing](https://arxiv.org/pdf/1906.02629.pdf), [Hinton Knowledge Distillation Loss](https://arxiv.org/pdf/1503.02531.pdf)
  - Learning Rate Scheduler: [Cosine Annealing](https://arxiv.org/abs/1608.03983) with Initial Warmups

```bash
$ python train.py --help
usage: train.py [-h] [--multi-gpu] [--gpu GPU] [--finetune FINETUNE]
                [--resume RESUME] [--half] [--wlog] [--config CONFIG]

Model trainer.

optional arguments:
  -h, --help           show this help message and exit
  --multi-gpu          Multi-GPU use
  --gpu GPU            GPU id to use
  --finetune FINETUNE  Model path to finetune (.pth.tar)
  --resume RESUME      Input log directory name to resume in save/checkpoint
  --half               Use half precision
  --wlog               Turns on wandb logging
  --config CONFIG      Configuration path (.py)

$ python train.py --config config_path  # how to run
```

#### Run pruning
Pruning makes a model sparse. Pruner supports the following methods:

1. Unstructured Pruning
  - [Lottery Ticket Hypothesis (LTH)](https://arxiv.org/pdf/1803.03635.pdf)
  - [LTH with weight rewinding](https://arxiv.org/pdf/1903.01611.pdf)
  - [LTH with learning rate rewinding](https://arxiv.org/pdf/2003.02389.pdf)

2. Structured (Channel-wise) Pruning
  - [Network Sliming](https://arxiv.org/pdf/1708.06519.pdf)
  - [Magnitude based channel-wise pruning](http://pages.cs.wisc.edu/~kadav/app/papers/pruning.pdf)
  - Slim-Magnitude channel-wise pruning (combination of above two methods)

Usually, unstructured pruning gives more sparsity, but it doesn't support shrinking.

```bash
$ python prune.py --help
usage: prune.py [-h] [--multi-gpu] [--gpu GPU] [--resume RESUME] [--wlog]
                [--config CONFIG]

Model pruner.

optional arguments:
  -h, --help       show this help message and exit
  --multi-gpu      Multi-GPU use
  --gpu GPU        GPU id to use
  --resume RESUME  Input checkpoint directory name
  --wlog           Turns on wandb logging
  --config CONFIG  Configuration path

usage: prune.py [-h] [--gpu GPU] [--resume RESUME] [--wlog] [--config CONFIG]

$ python prune.py --config config_path  # how to run
```

#### Run shrinking
Shrinking reshapes a pruned model and reduce its size.

```bash
$ python shrink.py --help
usage: shrink.py [-h] [--gpu GPU] [--checkpoint CHECKPOINT] [--config CONFIG]

Model shrinker.

optional arguments:
  -h, --help            show this help message and exit
  --gpu GPU             GPU id to use
  --checkpoint CHECKPOINT
                        input checkpoint path to quantize
  --config CONFIG       Pruning configuration path

$ python shrink.py --config config_path --checkpoint checkpoint_path  # how to run
```


##### Important Notes:
Shrinker is now experimental. It only supports:
  - networks that consist of conv-bn-activation sequence
  - network blocks that has channel concatenation followed by skip connections (e.g. DenseNet)
  - networks that have only one last fully-connected layer

On the other hads, it doesn't support:
  - network blocks that has element-wise sum followed by skip connections (e.g. ResNet, MixNet)
  - networks that have multiple fully-connected layers
  - Quantization after shrinking


#### Run quantization
It conducts one of 8-bit quantization methods:
  - post-training static quantization
  - Quantization-Aware Training

```bash
$ python quantize.py --help
usage: quantize.py [-h] [--resume RESUME] [--wlog] [--config CONFIG]
                   [--checkpoint CHECKPOINT]

Model quantizer.

optional arguments:
  -h, --help            show this help message and exit
  --resume RESUME       Input log directory name to resume
  --wlog                Turns on wandb logging
  --static              Post-training static quantization
  --config CONFIG       Configuration path
  --checkpoint CHECKPOINT
                        Input checkpoint path to quantize

$ python quantize.py --config config_path -checkpoint checkpoint_path
```

## Experiment Results

#### Unstructured Pruning
TODO

#### Structured Pruning
TODO

#### Shrinking after Structured Pruning
TODO

#### Quantization

###### Post-training Static Quantization
TODO

###### Quantization-Aware Training
TODO
