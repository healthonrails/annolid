FROM ubuntu:bionic
LABEL maintainer "Chen Yang <healthonrails@gmail.com>"

ENV DEBIAN_FRONTEND=noninteractive

RUN \
  apt-get update -qq && \
  apt-get install -qq -y \
    git \
    python3 \
    python3-pip \
    python3-matplotlib \
    python3-pyqt5 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install -U pip setuptools wheel

RUN python3 -m pip install -v git+https://github.com/healthonrails/annolid.git

RUN mkdir /root/workdir

ENV LANG en-US

WORKDIR /root/workdir
ENTRYPOINT [ "annolid" ]