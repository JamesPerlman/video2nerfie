
conda create -n video2nerfie python=3.8
source activate video2nerfie

pip install jax[cuda11_cudnn82] -f https://storage.googleapis.com/jax-releases/jax_releases.html

pip install -r requirements.txt
