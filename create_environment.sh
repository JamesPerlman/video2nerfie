#!/bin/bash

apt-get update
apt-get install -y \
    build-essential \
    cmake \
    ffmpeg \
    gcc \
    git \
    libboost-program-options-dev \
    libboost-filesystem-dev \
    libboost-graph-dev \
    libboost-system-dev \
    libboost-test-dev \
    libeigen3-dev \
    libopenexr-dev \
    libsuitesparse-dev \
    libfreeimage-dev \
    libgoogle-glog-dev \
    libgflags-dev \
    libglew-dev \
    libqt5opengl5-dev \
    libcgal-dev \
    openexr \
    python \
    qtbase5-dev

conda create -n video2nerfie python=3.8
source activate video2nerfie

pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install -r requirements.txt

# export PYTHONPATH="${PYTHONPATH}:/usr/local/video2nerfie"
