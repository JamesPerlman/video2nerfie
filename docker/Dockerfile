# syntax=docker/dockerfile:1

FROM continuumio/miniconda3:4.10.3p0-alpine AS conda_env

ENV DEBIAN_FRONTEND noninteractive

COPY requirements.txt .

RUN apk add curl git && \
    conda install conda-pack && \
    curl https://bootstrap.pypa.io/get-pip.py | python && \
    conda create -n video2nerfie python=3.8

RUN echo "conda activate video2nerfie" >> ~/.bashrc
SHELL ["/bin/bash", "--login", "-c"]

RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html
RUN pip install -r requirements.txt
RUN pip install pandas tensorflow-graphics
RUN mkdir /env
RUN conda-pack -n video2nerfie -o /env/video2nerfie.tar.gz

# Set up CUDA build machine

FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04 AS cuda_env

SHELL ["/bin/bash", "-c"]

COPY --from=conda_env /env/video2nerfie.tar.gz .

ENV DEBIAN_FRONTEND noninteractive
ENV HOME_PATH=/usr/local
ENV PROJ_PATH=${HOME_PATH}/video2nerfie

# Set up conda env

RUN mkdir /env && \
    tar -xzf video2nerfie.tar.gz -C /env && \
    rm video2nerfie.tar.gz && \
    echo "source /env/bin/activate" >> ~/.bashrc && \
    echo "export PYTHONPATH=\"${PYTHONPATH}:/usr/local/video2nerfie\"" >> ~/.bashrc

# Install dependencies

RUN apt-get update && \
    apt-get install -y \
        build-essential \
        cmake \
        ffmpeg \
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
        qtbase5-dev \
    && \
    git clone https://github.com/JamesPerlman/video2nerfie.git ${PROJ_PATH}

# Install ceres-solver
RUN git clone https://github.com/ceres-solver/ceres-solver.git -b 2.0.0 && \
    cd ceres-solver && \
    mkdir build && \
    cd build && \
    cmake .. -DBUILD_TESTING=OFF -DBUILD_EXAMPLES=OFF && \
    make -j8 && \
    make install

# Install colmap
RUN git clone https://github.com/colmap/colmap.git -b 3.7 && \
    cd colmap && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make -j8 && \
    make install

WORKDIR ${PROJ_PATH}
