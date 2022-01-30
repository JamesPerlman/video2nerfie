# syntax=docker/dockerfile:1

FROM continuumio/miniconda3:4.10.3p0-alpine AS miniconda

FROM nvidia/cuda:11.5.1-cudnn8-devel-ubuntu20.04 AS main

# Set up CUDA build machine
ENV DEBIAN_FRONTEND noninteractive
ENV HOME_PATH=/usr/local
ENV PROJ_PATH=${HOME_PATH}/video2nerfie

# We just need conda from the alpine miniconda build

COPY --from=miniconda /opt/conda /opt/conda

RUN echo "export PATH=${PATH}:/opt/conda/bin" >> ~/.bashrc

# Install apt dependencies first

RUN apt-get update && \
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
        qtbase5-dev \
    && \
    git clone https://github.com/JamesPerlman/video2nerfie.git ${PROJ_PATH}

WORKDIR ${PROJ_PATH}

# Create conda env

SHELL ["/bin/bash", "--login", "-c"]

RUN conda create -n video2nerfie python=3.8 && \
    echo "conda activate video2nerfie" >> ~/.bashrc

RUN pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html && \
    pip install -r requirements.txt && \
    pip install pandas tensorflow-graphics && \
    echo "export PYTHONPATH=\"${PYTHONPATH}:/usr/local/video2nerfie\"" >> ~/.bashrc

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
