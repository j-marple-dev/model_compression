# Model Compression

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
<!-- ALL-CONTRIBUTORS-BADGE:START - Do not remove or modify this section -->
[![All Contributors](https://img.shields.io/badge/all_contributors-4-green.svg?style=flat-square)](#contributors-)
<!-- ALL-CONTRIBUTORS-BADGE:END -->

## Contents

* [Getting started](https://github.com/j-marple-dev/model_compression#getting-started)
  * [Prerequisites](https://github.com/j-marple-dev/model_compression#prerequisites)
  * [Installation](https://github.com/j-marple-dev/model_compression#installation)
  * [Docker](https://github.com/j-marple-dev/model_compression#docker)

* [Usages](https://github.com/j-marple-dev/model_compression#usages)

  * [Run Training](https://github.com/j-marple-dev/model_compression#run-training)
  * [Configurations for training](https://github.com/j-marple-dev/model_compression#configurations-for-training)
  * [Run Pruning](https://github.com/j-marple-dev/model_compression#run-pruning)
  * [Configurations for pruning](https://github.com/j-marple-dev/model_compression#configurations-for-pruning)
  * [Run Shrinking](https://github.com/j-marple-dev/model_compression#run-shrinking-experimental)
  * [Run Quantization](https://github.com/j-marple-dev/model_compression#run-quantization)
* [Experimental Results](https://github.com/j-marple-dev/model_compression#experimental-results)
  * [Unstructured Pruning](https://github.com/j-marple-dev/model_compression#unstructured-pruning-lth-vs-weight-rewinding-vs-lr-rewinding)
  * [Structured Pruning](https://github.com/j-marple-dev/model_compression#structured-pruning-slim-vs-l2mag-vs-l2magslim)
  * [Shrinking after Structured Pruning](https://github.com/j-marple-dev/model_compression#shrinking-after-structured-pruning)
  * [Quantization](https://github.com/j-marple-dev/model_compression#quantization)
* [Class Diagram](https://github.com/j-marple-dev/model_compression#class-diagram)
* [References](https://github.com/j-marple-dev/model_compression#references)
* [Contributors](https://github.com/j-marple-dev/model_compression#contributors)

## Getting started

#### Prerequisites
* This repository is implemented and verified on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.7

#### Installation

0. Clone this repository.
```bash
$ git clone https://github.com/j-marple-dev/model_compression.git
$ cd model_compression
```

1. Create virtual environment
```bash
$ conda env create -f environment.yml 
$ conda activate model_compression
```
or
```bash
$ make install 
$ conda activate model_compression
```

2. (Optional for contributors) Install CI environment
```bash
$ conda activate model_compression
$ make dev
```

3. (Optional for nvidia gpu) Install cudatoolkit.
```bash
$ conda activate model_compression
$ conda install -c pytorch cudatooolkit=${cuda_version}
```

After environment setup, you can validate the code by the following commands.
```bash
$ make format  # for formatting
$ make test  # for linting
```

#### Docker

0. Clone this repository.
```bash
$ git clone https://github.com/j-marple-dev/model_compression.git
$ cd model_compression
```

1. Make sure you have installed [Docker Engine](https://www.docker.com/) and [nvidia-docker](https://github.com/NVIDIA/nvidia-docker).

2. Run the docker image.
```bash
$ docker run -it --gpus all --ipc=host -v $PWD:/app/model_compression jmarpledev/model_compression:latest /bin/bash
$ cd model_compression
```

## Usages

#### Run training
Training the model. Trainer supports the following options:
  - Basic Settings: batch size, epoch numbers, seed
  - Stochastic Gradient Descent: momentum, weight decay, initial learning rate, nesterov momentum
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

#### Configurations for training
Following options are available:
   - Basic Settings: BATCH_SIZE, EPOCHS, SEED, MODEL_NAME(src/models), MODEL_PARAMS, DATASET
   - Stochatic Gradient descent: MOMENTUM, WEIGHT_DECAY, LR
   - Image Augmentation: AUG_TRAIN(src/augmentation/policies.py), AUG_TRAIN_PARAMS, AUG_TEST(src/augmentation/policies.py), CUTMIX
   - Loss: CRITERION(src/criterions.py), CRITERION_PARAMS
   - Learning Rate Scheduler: LR_SCHEDULER(src/lr_schedulers.py), LR_SCHEDULER_PARAMS

```python
# Example of train config(config/train/cifar/densenet_121.py)
import os

config = {
    "SEED": 777,
    "AUG_TRAIN": "randaugment_train_cifar100_224",
    "AUG_TRAIN_PARAMS": dict(n_select=2, level=None),
    "AUG_TEST": "simple_augment_test_cifar100_224",
    "CUTMIX": dict(beta=1, prob=0.5),
    "DATASET": "CIFAR100",
    "MODEL_NAME": "densenet",
    "MODEL_PARAMS": dict(
        num_classes=100,
        inplanes=24,
        growthRate=32,
        compressionRate=2,
        block_configs=(6, 12, 24, 16),
        small_input=False,
        efficient=False,
    ),
    "CRITERION": "CrossEntropy", # CrossEntropy, HintonKLD
    "CRITERION_PARAMS": dict(num_classes=100, label_smoothing=0.1),
    "LR_SCHEDULER": "WarmupCosineLR", # WarmupCosineLR, Identity, MultiStepLR
    "LR_SCHEDULER_PARAMS": dict(
        warmup_epochs=5, start_lr=1e-3, min_lr=1e-5, n_rewinding=1
    ),
    "BATCH_SIZE": 128,
    "LR": 0.1,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 1e-4,
    "NESTEROV": True,
    "EPOCHS": 300,
    "N_WORKERS": os.cpu_count(),
}
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

#### Configurations for pruning
Pruning configuration extends training configuration (recommended) with following options:
  - Basic Training Settings: TRAIN_CONFIG
  - Pruning Settings: N_PRUNING_ITER, PRUNE_METHOD(src/runner/pruner.py), PRUNE_PARAMS

```python
# Example of prune config(config/prune/cifar100/densenet_small_l2mag.py)
from config.train.cifar100 import densenet_small

train_config = densenet_small.config
config = {
    "TRAIN_CONFIG": train_config,
    "N_PRUNING_ITER": 15,
    "PRUNE_METHOD": "Magnitude", # LotteryTicketHypothesis, Magnitude, NetworkSlimming, SlimMagnitude
    "PRUNE_PARAMS": dict(
        PRUNE_AMOUNT=0.2,  # it iteratively prunes 20% of the network parameters at the end of trainings
        NORM=2,
        STORE_PARAM_BEFORE=train_config["EPOCHS"],  # used for weight initialization at every pruning iteration
        TRAIN_START_FROM=0,  # training starts from this epoch
        PRUNE_AT_BEST=False,  # if True, it prunes parameters at the trained network which achieves the best accuracy
                              # otherwise, it prunes the network at the end of training
    ),
}
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

[WANDB Log](https://app.wandb.ai/j-marple/model_compression/reports/Structured-Pruning-Unstructured-Pruning--VmlldzoyMTA4MDA)

#### Unstructured Pruning (LTH vs Weight Rewinding vs LR Rewinding)
<img width="1079" alt="Screen Shot 2020-08-24 at 1 00 31 AM" src="https://user-images.githubusercontent.com/14961526/90982941-962b3500-e5a5-11ea-922f-f792a2192c2e.png">


#### Structured Pruning (Slim vs L2Mag vs L2MagSlim)
<img width="1078" alt="Screen Shot 2020-08-26 at 11 05 22 PM" src="https://user-images.githubusercontent.com/14961526/91313805-c6b2df00-e7f0-11ea-8e16-886c3a594247.png">

#### Shrinking after Structured Pruning

###### Densenet (L=100, k=12) pruned by 19.66% (Slim & CIFAR100)
![parameters](https://user-images.githubusercontent.com/14961526/91256808-43b76780-e7a3-11ea-965d-1543385e400a.png)

* Accuracy: 80.37%
* Parameters: 0.78M -> 0.51M
* Model Size: 6.48Mb -> 4.14Mb

```bash
$ python shrink.py --config config/prune/cifar100/densenet_small_slim.py --checkpoint path_to_checkpoint.pth.tar

2020-08-26 13:50:38,442 - trainer.py:71 - INFO - Created a model densenet with 0.78M params
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
2020-08-26 13:50:42,719 - shrinker.py:104 - INFO - Acc: 80.37, Size: 6.476016 MB, Sparsity: 19.66 %
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
2020-08-26 13:50:45,781 - shrinker.py:118 - INFO - Acc: 80.37, Size: 4.141713 MB, Params: 0.51 M
```

###### Densenet (L=100, k=12) pruned by 35.57% (Network Slimming & CIFAR100)
![parameters](https://user-images.githubusercontent.com/14961526/91256890-81b48b80-e7a3-11ea-812b-d806e4afab34.png)

* Accuracy: 79.07%
* Parameters: 0.78M -> 0.35M
* Model Size: 6.48Mb -> 2.85Mb

```bash
$ python shrink.py --config config/prune/cifar100/densenet_small_slim.py --checkpoint path_to_checkpoint.pth.tar

2020-08-26 13:52:58,946 - trainer.py:71 - INFO - Created a model densenet with 0.78M params
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
2020-08-26 13:53:03,100 - shrinker.py:104 - INFO - Acc: 79.07, Size: 6.476016 MB, Sparsity: 35.57 %
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:00:02 Time:  0:00:02
2020-08-26 13:53:06,114 - shrinker.py:118 - INFO - Acc: 79.07, Size: 2.851149 MB, Params: 0.35 M
```


#### Quantization

###### Post-training Static Quantization

```bash
$ python quantize.py --config config/quantize/cifar100/densenet_small.py --checkpoint save/test/densenet_small/296_81_20.pth.tar --static --check-acc

2020-08-26 13:57:02,595 - trainer.py:71 - INFO - Created a model quant_densenet with 0.78M params
2020-08-26 13:57:05,275 - quantizer.py:87 - INFO - Acc: 81.2 %  Size: 3.286695 MB
2020-08-26 13:57:05,344 - quantizer.py:95 - INFO - Post Training Static Quantization: Run calibration
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:40 Time:  0:02:40
2020-08-26 13:59:47,555 - quantizer.py:117 - INFO - Acc: 81.03 %  Size: 0.974913 MB
```

###### Quantization-Aware Training

```bash
$ python quantize.py --config config/quantize/cifar100/densenet_small.py --checkpoint path_to_checkpoint.pth.tar --check-acc

2020-08-26 14:06:46,855 - trainer.py:71 - INFO - Created a model quant_densenet with 0.78M params
2020-08-26 14:06:49,506 - quantizer.py:87 - INFO - Acc: 81.2 %  Size: 3.286695 MB
2020-08-26 14:06:49,613 - quantizer.py:99 - INFO - Quantization Aware Training: Run training
2020-08-26 14:46:51,857 - trainer.py:209 - INFO - Epoch: [0 | 4]        train/lr: 0.0001        train/loss: 1.984219    test/loss: 1.436638     test/model_acc: 80.96%    test/best_acc: 80.96%
[Train] 100% (782 of 782) |########################################################################################| Elapsed Time: 0:38:09 Time:  0:38:09
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:40 Time:  0:02:40
2020-08-26 15:27:43,919 - trainer.py:209 - INFO - Epoch: [1 | 4]        train/lr: 9e-05 train/loss: 1.989543    test/loss: 1.435748     test/model_acc: 80.87%    test/best_acc: 80.96%
[Train] 100% (782 of 782) |########################################################################################| Elapsed Time: 0:38:10 Time:  0:38:10
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:36 Time:  0:02:36
2020-08-26 16:08:32,883 - trainer.py:209 - INFO - Epoch: [2 | 4]        train/lr: 6.5e-05       train/loss: 1.984149    test/loss: 1.436074     test/model_acc: 80.82%    test/best_acc: 80.96%
[Train] 100% (782 of 782) |########################################################################################| Elapsed Time: 0:38:14 Time:  0:38:14
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:39 Time:  0:02:39
2020-08-26 16:49:28,848 - trainer.py:209 - INFO - Epoch: [3 | 4]        train/lr: 3.5e-05       train/loss: 1.984537    test/loss: 1.43442      test/model_acc: 81.01%    test/best_acc: 81.01%
[Train] 100% (782 of 782) |########################################################################################| Elapsed Time: 0:38:19 Time:  0:38:19
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:42 Time:  0:02:42
2020-08-26 17:30:32,187 - trainer.py:209 - INFO - Epoch: [4 | 4]        train/lr: 1e-05 train/loss: 1.990936    test/loss: 1.435393     test/model_acc: 80.92%    test/best_acc: 81.01%
[Test]  100% (157 of 157) |#########################################################################################| Elapsed Time: 0:02:37 Time:  0:02:37
2020-08-26 17:33:10,689 - quantizer.py:117 - INFO - Acc: 81.01 %        Size: 0.974913 MB
```

## Class Diagram

<img width="671" alt="class_diagram" src="https://user-images.githubusercontent.com/25141842/90982522-80684080-e5a2-11ea-812c-a2395caf9826.png">


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

## Contributors

Thanks goes to these wonderful people ([emoji key](https://allcontributors.org/docs/en/emoji-key)):

<!-- ALL-CONTRIBUTORS-LIST:START - Do not remove or modify this section -->
<!-- prettier-ignore-start -->
<!-- markdownlint-disable -->
<table>
  <tr>
    <td align="center"><a href="https://github.com/Curt-Park"><img src="https://avatars3.githubusercontent.com/u/14961526?v=4" width="100px;" alt=""/><br /><sub><b>Jinwoo Park (Curt)</b></sub></a><br /><a href="https://github.com/j-marple-dev/model_compression/commits?author=Curt-Park" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/Hoonyyhoon"><img src="https://avatars0.githubusercontent.com/u/25141842?v=4" width="100px;" alt=""/><br /><sub><b>Junghoon Kim</b></sub></a><br /><a href="https://github.com/j-marple-dev/model_compression/commits?author=Hoonyyhoon" title="Code">💻</a></td>
    <td align="center"><a href="https://github.com/HSShin0"><img src="https://avatars0.githubusercontent.com/u/44793742?v=4" width="100px;" alt=""/><br /><sub><b>Hyungseok Shin</b></sub></a><br /><a href="https://github.com/j-marple-dev/model_compression/commits?author=HSShin0" title="Code">💻</a></td>
    <td align="center"><a href="https://www.linkedin.com/in/juhee-lee-393342126/"><img src="https://avatars0.githubusercontent.com/u/18753708?v=4" width="100px;" alt=""/><br /><sub><b>Juhee Lee</b></sub></a><br /><a href="https://github.com/j-marple-dev/model_compression/commits?author=Ingenjoy" title="Code">💻</a></td>
  </tr>
</table>

<!-- markdownlint-enable -->
<!-- prettier-ignore-end -->
<!-- ALL-CONTRIBUTORS-LIST:END -->

This project follows the [all-contributors](https://github.com/all-contributors/all-contributors) specification. Contributions of any kind welcome!
