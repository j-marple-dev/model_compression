FROM nvcr.io/nvidia/tensorrt:21.05-py3

LABEL maintainer="Jongkuk Lim <limjk@jmarple.ai>"

ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=Asia/Seoul

ARG	UID=1000
ARG	GID=1000
RUN	groupadd -g $GID -o user && useradd -m -u $UID -g $GID -o -s /bin/bash user

RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
RUN apt-get update && apt-get install -y sudo dialog apt-utils tzdata
RUN	echo "%sudo	ALL=(ALL) NOPASSWD: ALL" >> /etc/sudoers && echo "user:user" | chpasswd && adduser user sudo

WORKDIR	/home/user
USER	user

# Install Display dependencies
RUN sudo apt-get update && sudo apt-get install -y libgl1-mesa-dev && sudo apt-get -y install jq

# Install pip3 and C++ linter
RUN sudo apt-get install -y clang-format cppcheck
RUN curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && python3 get-pip.py --force-reinstall && python3 -m pip install --upgrade pip
RUN python3 -m pip install wheel cpplint

# Install doxygen for C++ documentation
RUN sudo apt-get update && sudo apt-get install -y flex bison && sudo apt-get autoremove -y
RUN git clone -b Release_1_9_2 https://github.com/doxygen/doxygen.git \
    && cd doxygen \
    && mkdir build \
    && cd build \
    && cmake -G "Unix Makefiles" .. \
    && make -j `cat /proc/cpuinfo | grep cores | wc -l` \
    && sudo make install

# Install PyTorch CUDA 11.1
RUN python3 -m pip install torch==1.9.1+cu111 torchvision==0.10.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html

# Install other development dependencies
COPY ./requirements-dev.txt ./
RUN python3 -m pip install -r requirements-dev.txt
RUN rm requirements-dev.txt

# Download libtorch
RUN wget -q https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.9.1%2Bcu111.zip \
    && unzip libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111.zip \
    && mkdir libs \
    && mv libtorch libs/libtorch \
    && rm libtorch-cxx11-abi-shared-with-deps-1.9.1+cu111.zip

# Install cmake 3.21.0 version.
RUN wget -q https://github.com/Kitware/CMake/releases/download/v3.21.0/cmake-3.21.0-linux-x86_64.tar.gz \
    && tar -xzvf cmake-3.21.0-linux-x86_64.tar.gz \
    && sudo ln -s /home/user/cmake-3.21.0-linux-x86_64/bin/cmake /usr/bin/cmake \
    && sudo ln -s /home/user/root/cmake-3.21.0-linux-x86_64/bin/ctest /usr/bin/ctest \
    && sudo ln -s /home/user/root/cmake-3.21.0-linux-x86_64/bin/cpack /usr/bin/cpack \
    && rm cmake-3.21.0-linux-x86_64.tar.gz

# Terminal environment
RUN git clone https://github.com/JeiKeiLim/my_term.git \
    && cd my_term \
    && ./run.sh

# Fix error messages with vim plugins
RUN cd /home/user/.vim_runtime/sources_non_forked && rm -rf tlib vim-fugitive && git clone https://github.com/tomtom/tlib_vim.git tlib && git clone https://github.com/tpope/vim-fugitive.git 

# Install vim 8.2 with YCM
RUN sudo apt-get install -y software-properties-common \
    && sudo add-apt-repository ppa:jonathonf/vim \
    && sudo add-apt-repository ppa:ubuntu-toolchain-r/test \
    && sudo apt-get update \
    && sudo apt-get install -y vim g++-8 libstdc++6

RUN cd /home/user/.vim_runtime/my_plugins \
    && git clone --recursive https://github.com/ycm-core/YouCompleteMe.git \
    && cd YouCompleteMe \
    && CC=gcc-8 CXX=g++-8 python3 install.py --clangd-completer

# Install DALI
RUN python3 -m pip install --extra-index-url https://developer.download.nvidia.com/compute/redist nvidia-dali-cuda110 

# Add PATH
RUN echo "export PATH=/home/user/.local/bin:\$PATH" >> /home/user/.bashrc
RUN echo "export LC_ALL=C.UTF-8 && export LANG=C.UTF-8" >> /home/user/.bashrc
