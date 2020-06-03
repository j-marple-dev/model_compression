# Model Compression

[![CircleCI](https://circleci.com/gh/circleci/circleci-docs.svg?style=shield)](https://circleci.com/gh/Curt-Park/model_compression)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

## Getting started

#### Prerequisites
* This repository is tested on [Anaconda](https://www.anaconda.com/distribution/) virtual environment with python 3.7
```
$ conda create -n model_compression python=3.7
$ conda activate model_compression
```

#### Installation
First, clone the repository.
```
git clone https://github.com/Curt-Park/model_compression.git
cd model_compression
```

###### For users
Install packages required to execute the code. Just type:
```
make dep
```

###### For developers
If you want to modify code you should configure formatting and linting settings. It automatically runs formatting and linting when you commit the code. Just type:
```
make dev
```

After having done `make dev`, you can validate the code by the following commands.
```
make format  # for formatting
make test  # for linting
```
