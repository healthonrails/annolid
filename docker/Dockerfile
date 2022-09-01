# Modified from:
# 1. https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
# 2. https://github.com/wkentaro/labelme/blob/main/docker/Dockerfile

FROM nvidia/cuda:11.1.1-cudnn8-devel-ubuntu20.04
LABEL maintainer "Chen Yang <healthonrails@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get update -qq && \
  apt-get install -qq -y \
    git \
    wget \
    ca-certificates \
    python3 \
    python3-pip \
    python3-matplotlib \
    python3-pyqt5 \
    python3-opencv \
    python3-dev \
    sudo \
    ninja-build\
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel
RUN ln -sv /usr/bin/python3 /usr/bin/python

# create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system  --uid ${USER_ID} annoliduser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER annoliduser
WORKDIR /home/annoliduser

ENV PATH="/home/annoliduser/.local/bin:${PATH}"
RUN wget https://bootstrap.pypa.io/pip/3.6/get-pip.py && \
	python3 get-pip.py --user && \
	rm get-pip.py

# install dependencies
RUN pip install --user tensorboard cmake
RUN pip install --user torch==1.9 torchvision==0.10 -f https://download.pytorch.org/whl/cu111/torch_stable.html

RUN pip install --user 'git+https://github.com/facebookresearch/fvcore'
# install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
# set FORCE_CUDA because during `docker build` cuda is not accessible
ENV FORCE_CUDA="1"
# This will by default build detectron2 for all common cuda architectures and take a lot more time,
# because inside `docker build`, there is no way to tell which architecture will be used.
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

RUN git clone --recurse-submodules https://github.com/healthonrails/annolid.git
# RUN git clone --recurse-submodules -b docker_bug --single-branch https://github.com/jeremyforest/annolid.git
RUN pip install --user -e annolid

# Set a fixed model cache directory.
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/annoliduser/annolid

ENV LANG en-US

ENTRYPOINT [ "annolid" ]
