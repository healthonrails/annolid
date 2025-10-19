# Modified from:
# 1. https://github.com/facebookresearch/detectron2/blob/main/docker/Dockerfile
# 2. https://github.com/wkentaro/labelme/blob/main/docker/Dockerfile

# Use the official PyTorch image as the base
FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-devel
LABEL maintainer "Chen Yang <healthonrails@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update -qq && \
    apt-get install -qq -y \
    git \
    wget \
    ca-certificates \
    python3-matplotlib \
    python3-pyqt5 \
    python3-opencv \
    sudo \
    ninja-build \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user
ARG USER_ID=1000
RUN useradd -m --no-log-init --system --uid ${USER_ID} annoliduser -g sudo
RUN echo '%sudo ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers
USER annoliduser
WORKDIR /home/annoliduser

# Set the path for user-installed packages
ENV PATH="/home/annoliduser/.local/bin:${PATH}"

# Install python dependencies
RUN pip install --user tensorboard cmake 'git+https://github.com/facebookresearch/fvcore'

# Clone and install detectron2
RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo
ENV FORCE_CUDA="1"
ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"

RUN pip install --user -e detectron2_repo

# Clone and install annolid
RUN git clone --recurse-submodules https://github.com/healthonrails/annolid.git
RUN pip install --user -e annolid
RUN pip install --user numpy==1.24.4
RUN rm  -rf  /home/annoliduser/.local/lib/python3.11/site-packages/cv2/qt/plugins/*
# Set a fixed model cache directory
ENV FVCORE_CACHE="/tmp"
WORKDIR /home/annoliduser/annolid

# Set language environment
ENV LANG en-US

# Entry point for annolid
ENTRYPOINT [ "annolid" ]
