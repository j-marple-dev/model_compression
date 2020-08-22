# Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Contents

* [Getting started](https://github.com/j-marple-dev/model_compression#getting-started)
* [Usages](https://github.com/j-marple-dev/model_compression#usages)
* [Experimental Results](https://github.com/j-marple-dev/model_compression#experimental-results)
* [Class Diagram](https://github.com/j-marple-dev/model_compression#class-diagram)
* [References](https://github.com/j-marple-dev/model_compression#references)

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
  - Basic Settings: batch size, epoch numbers, seed
  - Stochastic Gradient Decent: momentum, weight decay, initial learning rate, nesterov momentum
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

$ python train.py --config path_to_config.py  # basic run
$ python train.py --config path_to_config.py  --gpu 1 --resume checkpoint_dir_name # resume training on gpu 1
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

$ python prune.py --config path_to_config.py  # basic run
$ python prune.py --config path_to_config.py --multi-gpu --wlog  # run on multi-gpu with wandb logging
```

#### Run shrinking (Experimental)
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

$ python shrink.py --config path_to_config.py --checkpoint path_to_checkpoint.pth.tar  # basic run
```


##### Important Notes:
Shrinker is now experimental. It only supports:
  - channel-wise prunned models
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

$ python quantize.py --config path_to_config.py --checkpoint path_to_checkpoint.pth.tar  # basic qat run
$ python quantize.py --config path_to_config.py --checkpoint path_to_checkpoint.pth.tar --static  # basic static quantization run
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


## Class Diagram

<img width="671" alt="class_diagram" src="https://user-images.githubusercontent.com/14961526/90956407-5ab73a80-e4c1-11ea-8d71-f78b8b997a01.png">


## References

#### Papers

###### Architecture / Training
* [MixConv: Mixed Depthwise Convolutional Kernels](https://arxiv.org/pdf/1907.09595.pdf)
* [Densely Connected Convolutional Networks](https://arxiv.org/pdf/1608.06993.pdf)
* [Memory-Efficient Implementation of DenseNets](https://arxiv.org/pdf/1707.06990.pdf)
* [When Does Label Smoothing Help?](https://arxiv.org/pdf/1906.02629.pdf)
* [SGDR: Stochastic Gradient Descent with Warm Restarts](https://arxiv.org/pdf/1608.03983.pdf)

###### Augmentation
* [Improved Regularization of Convolutional Neural Networks with Cutout](https://arxiv.org/pdf/1708.04552.pdf)
* [AutoAugment: Learning Augmentation Strategies from Data](https://arxiv.org/pdf/1805.09501.pdf)
* [RandAugment: Practical automated data augmentation with a reduced search space](https://arxiv.org/pdf/1909.13719.pdf)
* [CutMix: Regularization Strategy to Train Strong Classifiers with Localizable Features](https://arxiv.org/pdf/1905.04899.pdf)

###### Pruning
* [WHAT IS THE STATE OF NEURAL NETWORK PRUNING?](https://arxiv.org/pdf/2003.03033.pdf)
* [THE LOTTERY TICKET HYPOTHESIS: FINDING SPARSE, TRAINABLE NEURAL NETWORKS](https://arxiv.org/pdf/1803.03635.pdf)
* [Stabilizing the Lottery Ticket Hypothesis](https://arxiv.org/pdf/1903.01611.pdf)
* [COMPARING REWINDING AND FINE-TUNING IN NEURAL NETWORK PRUNING](https://arxiv.org/pdf/2003.02389.pdf)
* [Learning Efficient Convolutional Networks through Network Slimming](https://arxiv.org/pdf/1708.06519.pdf)
* [PRUNING FILTERS FOR EFFICIENT CONVNETS](http://pages.cs.wisc.edu/~kadav/app/papers/pruning.pdf)

###### Knowledge Distillation
* [The State Of Knowledge Distillation For Classification Tasks](https://arxiv.org/pdf/1912.10850.pdf)
* [Distilling the Knowledge in a Neural Network](https://arxiv.org/pdf/1503.02531.pdf)

###### Quantization
* [Quantizing deep convolutional networks for efficient inference: A whitepaper](https://arxiv.org/pdf/1806.08342.pdf)


#### Implementations / Tutorials

###### Competition
* https://github.com/wps712/MicroNetChallenge/tree/cifar100
* https://github.com/Kthyeon/micronet_neurips_challenge

###### Architecture / Training
* https://github.com/rwightman/pytorch-image-models
* https://github.com/bearpaw/pytorch-classification
* https://github.com/gpleiss/efficient_densenet_pytorch
* https://github.com/leaderj1001/Mixed-Depthwise-Convolutional-Kernels

###### Augmentation
* https://github.com/kakaobrain/fast-autoaugment/
* https://github.com/DeepVoltaire/AutoAugment
* https://github.com/clovaai/CutMix-PyTorch

###### Pruning
* https://github.com/facebookresearch/open_lth
* https://github.com/lottery-ticket/rewinding-iclr20-public
* https://pytorch.org/tutorials/intermediate/pruning_tutorial.html

###### Knowledge Distillation
* https://github.com/karanchahal/distiller

###### Quantization
* https://pytorch.org/docs/stable/quantization.html
* https://pytorch.org/tutorials/advanced/static_quantization_tutorial.html
* https://github.com/pytorch/vision/tree/master/torchvision/models/quantization
