# syntax=docker/dockerfile:1

FROM continuumio/miniconda3:4.10.3p0-alpine AS conda_env

ENV DEBIAN_FRONTEND noninteractive

COPY requirements.txt .

RUN apk add curl git && \
    conda install conda-pack && \
    curl https://bootstrap.pypa.io/get-pip.py | python && \
    conda create -n video2nerfie python=3.8

SHELL ["conda", "run", "-n", "video2nerfie", "/bin/bash", "-c"]

RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    pip install -r requirements.txt && \
    mkdir /env && \
    conda-pack -n video2nerfie -o /env/video2nerfie.tar.gz

# Set up CUDA build machine

FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu18.04 AS cuda_env

SHELL ["/bin/bash", "-c"]

COPY --from=conda_env /env/video2nerfie.tar.gz .

ENV DEBIAN_FRONTEND noninteractive
ENV HOME_PATH=/usr/local
ENV PROJ_PATH=${HOME_PATH}/video2nerfie

# Set up conda env

RUN mkdir /env && \
    tar -xzf video2nerfie.tar.gz -C /env && \
    rm video2nerfie.tar.gz && \
    source /env/bin/activate

# Install dependencies

RUN apt-get update && \
    apt-get install -y \
        colmap \
        ffmpeg \
        git \
        libopenexr-dev \
        openexr \
    && \
    git clone https://github.com/JamesPerlman/video2nerfie.git ${PROJ_PATH}

WORKDIR ${PROJ_PATH}


