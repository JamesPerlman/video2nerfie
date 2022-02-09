
conda create -n video2nerfie python=3.8
source activate video2nerfie

pip install -y https://storage.googleapis.com/jax-releases/cuda11/jaxlib-0.1.75+cuda11.cudnn82-cp38-none-manylinux2010_x86_64.whl

pip install -r requirements.txt
