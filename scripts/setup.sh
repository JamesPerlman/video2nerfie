#!/bin/bash

apt-get update && apt-get install -y wget rsync git nano

# install conda
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p /opt/conda
conda init bash

source ~/.bashrc

# clone repo
git clone https://github.com/JamesPerlman/video2nerfie
cd video2nerfie

# handle dependencies
./scripts/install_dependencies.sh
./scripts/install_colmap.sh
./scripts/create_environment.sh

# necessary env vars
echo "export XLA_PYTHON_CLIENT_PREALLOCATE=false" >> ~/.bashrc
echo "export XLA_PYTHON_CLIENT_ALLOCATOR=platform" >> ~/.bashrc
echo "export TF_GPU_ALLOCATOR=cuda_malloc_async" >> ~/.bashrc
echo "export PYTHONPATH=$PYTHONPATH:~/video2nerfie" >> ~/.bashrc

source ~/.bashrc

cd ~/video2nerfie
