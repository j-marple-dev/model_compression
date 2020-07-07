# Model Compression

[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=shield)](https://circleci.com/gh/Curt-Park/model_compression)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Getting started

#### Prerequisites
* This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.7
```bash
$ conda create -n model_compression python=3.7
$ conda activate model_compression
```

#### Installation
First, clone the repository.
```bash
$ git clone https://github.com/Curt-Park/model_compression.git
$ cd model_compression
```

###### Prerequisites

1. Install PyTorch 1.5.0 and Torchvision 0.6.0 (See the following official instruction).

https://pytorch.org/get-started/previous-versions/

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
```bash
$ python train.py --config config_path
$ python train.py --help
usage: train.py [-h] [--gpu GPU] [--resume RESUME] [--wlog] [--config CONFIG]

Model trainer.

optional arguments:
  -h, --help       show this help message and exit
  --gpu GPU        GPU id to use
  --resume RESUME  Input log directory name to resume
  --wlog           Turns on wandb logging
  --config CONFIG  Configuration path
```

#### Run pruning
```bash
$ python prune.py --config config_path 
$ python prune.py --help
usage: prune.py [-h] [--gpu GPU] [--resume RESUME] [--wlog] [--config CONFIG]

Model pruner.

optional arguments:
  -h, --help       show this help message and exit
  --gpu GPU        GPU id to use
  --resume RESUME  Input log directory name to resume
  --wlog           Turns on wandb logging
  --config CONFIG  Configuration path
```

#### Run quantization
```bash
$ python quantize.py --config config_path -checkpoint checkpoint_path
$ python quantize.py --help
usage: quantize.py [-h] [--resume RESUME] [--wlog] [--config CONFIG]
                   [--checkpoint CHECKPOINT]

Model quantizer.

optional arguments:
  -h, --help            show this help message and exit
  --resume RESUME       Input log directory name to resume
  --wlog                Turns on wandb logging
  --config CONFIG       Configuration path
  --checkpoint CHECKPOINT
                        Input checkpoint path to quantize
```
