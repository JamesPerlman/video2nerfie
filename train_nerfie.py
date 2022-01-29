##############################################
# Most of this code is from the nerfies repo #
# https://github.com/google/nerfies.git      #
##############################################

# python video2nerfie/train_nerfie.py -i "content/hoksmac/nerfies_dataset/" -o "content/hoksmac/nerfies_train"
###################################
# GET ARGS                        #
###################################

from argparse import ArgumentParser
from pathlib import Path

parser = ArgumentParser("Train Nerfie")
parser.add_argument("-i", "--input", help="Path to nerfies dataset", type=str, required=True)
parser.add_argument("-o", "--output", help="Path to trained output data", type=str, required=True)

args = parser.parse_args()

data_dir = Path(args.input)
train_dir = Path(args.output)
train_dir.mkdir(exist_ok=True)

###################################
# ENVIRONMENT SETUP               #
###################################

import jax
from jax.config import config as jax_config
import jax.numpy as jnp
from jax import grad, jit, vmap
from jax import random

import flax
import flax.linen as nn
from flax import jax_utils
from flax import optim
from flax.metrics import tensorboard
from flax.training import checkpoints
jax_config.enable_omnistaging() # Linen requires enabling omnistaging

from absl import logging
from io import BytesIO
import random as pyrandom
import numpy as np
import PIL
import IPython

runtime_type = 'gpu'  # @param ['gpu', 'tpu']
if runtime_type == 'tpu':
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()

print('Detected Devices:', jax.devices())

# Monkey patch logging.
def myprint(msg, *args, **kwargs):
    print(msg % args)

logging.info = myprint 
logging.warn = myprint
logging.error = myprint


###################################
# CONFIG                          #
###################################

from pathlib import Path
from pprint import pprint
import gin

from nerfies import configs


# @markdown Training configuration.
max_steps = 100000  # @param {type: 'number'}
batch_size = 4096  # @param {type: 'number'}
image_scale = 1  # @param {type: 'number'}

# @markdown Model configuration.
use_viewdirs = True  #@param {type: 'boolean'}
use_appearance_metadata = True  #@param {type: 'boolean'}
warp_field_type = 'se3'  #@param['se3', 'translation']
num_warp_freqs = 8  #@param{type:'number'}
num_coarse_samples = 64  # @param {type: 'number'}
num_fine_samples = 64  # @param {type: 'number'}

checkpoint_dir = Path(train_dir, 'checkpoints')
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_str = f"""
ExperimentConfig.image_scale = {image_scale}
ExperimentConfig.datasource_spec = {{
    'type': 'nerfies',
    'data_dir': '{data_dir}',
    'camera_type': 'json',
}}

ModelConfig.use_warp = True
ModelConfig.use_viewdirs = {int(use_viewdirs)}
ModelConfig.use_appearance_metadata = {int(use_appearance_metadata)}
ModelConfig.warp_field_type = 'se3'
ModelConfig.num_warp_freqs = {num_warp_freqs}
ModelConfig.num_coarse_samples = {num_coarse_samples}
ModelConfig.num_fine_samples = {num_fine_samples}

TrainConfig.max_steps = {max_steps}
TrainConfig.batch_size = {batch_size}
TrainConfig.print_every = 100
TrainConfig.use_elastic_loss = False
TrainConfig.use_background_loss = False
TrainConfig.warp_alpha_schedule = {{
    'type': 'linear',
    'initial_value': 0.0,
    'final_value': {num_warp_freqs},
    'num_steps': {int(max_steps*0.8)},
}}
"""

gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
  logging.info('Saving config to %s', config_path)
  f.write(config_str)

exp_config = configs.ExperimentConfig()
model_config = configs.ModelConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()

###################################
# CREATE DATASOURCE               #
###################################

from nerfies import datasets
from nerfies import image_utils

datasource = datasets.from_config(
  exp_config.datasource_spec,
  image_scale=exp_config.image_scale,
  use_appearance_id=model_config.use_appearance_metadata,
  use_camera_id=model_config.use_camera_metadata,
  use_warp_id=model_config.use_warp,
  random_seed=exp_config.random_seed
)

# show_image(datasource.load_rgb(datasource.train_ids[0]))

###################################
# CREATE TRAINING ITERATORS       #
###################################

devices = jax.local_devices()

train_iter = datasource.create_iterator(
    datasource.train_ids,
    batch_size=train_config.batch_size,
    flatten=True,
    shuffle=True,
    prefetch_size=3,
    devices=devices
)

def shuffled(l):
  import random as r
  import copy
  l = copy.copy(l)
  r.shuffle(l)
  return l

train_eval_iter = datasource.create_iterator(
    shuffled(datasource.train_ids),
    batch_size=0,
    devices=devices
)

val_eval_iter = datasource.create_iterator(
    shuffled(datasource.val_ids),
    batch_size=0,
    devices=devices
)

###################################
# TRAINING - INITIALIZE MODEL     #
###################################

# Defines the model and initializes its parameters.

from flax.training import checkpoints
from nerfies import models
from nerfies import model_utils
from nerfies import schedules
from nerfies import training

# @markdown Restore a checkpoint if one exists.
restore_checkpoint = False  # @param{type:'boolean'}


rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.process_index())
devices_to_use = jax.devices()

learning_rate_sched = schedules.from_config(train_config.lr_schedule)
warp_alpha_sched = schedules.from_config(train_config.warp_alpha_schedule)
elastic_loss_weight_sched = schedules.from_config(
    train_config.elastic_loss_weight_schedule)

rng, key = random.split(rng)
params = {}
model, params['model'] = models.construct_nerf(
    key,
    model_config,
    batch_size=train_config.batch_size,
    appearance_ids=datasource.appearance_ids,
    camera_ids=datasource.camera_ids,
    warp_ids=datasource.warp_ids,
    near=datasource.near,
    far=datasource.far,
    use_warp_jacobian=train_config.use_elastic_loss,
    use_weights=train_config.use_elastic_loss
)

optimizer_def = optim.Adam(learning_rate_sched(0))
optimizer = optimizer_def.create(params)

state = model_utils.TrainState(
    optimizer=optimizer,
    warp_alpha=warp_alpha_sched(0)
)

scalar_params = training.ScalarParams(
    learning_rate=learning_rate_sched(0),
    elastic_loss_weight=elastic_loss_weight_sched(0),
    warp_reg_loss_weight=train_config.warp_reg_loss_weight,
    warp_reg_loss_alpha=train_config.warp_reg_loss_alpha,
    warp_reg_loss_scale=train_config.warp_reg_loss_scale,
    background_loss_weight=train_config.background_loss_weight
)

if restore_checkpoint:
  logging.info('Restoring checkpoint from %s', checkpoint_dir)
  state = checkpoints.restore_checkpoint(checkpoint_dir, state)

step = state.optimizer.state.step + 1
state = jax_utils.replicate(state, devices=devices)

del params


###################################
# DEFINE PMAPPED FUNCTIONS        #
###################################

# This parallelizes the training and evaluation step functions using `jax.pmap`.

import functools
from nerfies import evaluation


def _model_fn(key_0, key_1, params, rays_dict, warp_extra):
  out = model.apply({'params': params},
                    rays_dict,
                    warp_extra=warp_extra,
                    rngs={
                        'coarse': key_0,
                        'fine': key_1
                    },
                    mutable=False)
  return jax.lax.all_gather(out, axis_name='batch')

pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices_to_use,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

render_fn = functools.partial(evaluation.render_image,
                              model_fn=pmodel_fn,
                              device_count=len(devices),
                              chunk=eval_config.chunk)
train_step = functools.partial(
    training.train_step,
    model,
    elastic_reduce_method=train_config.elastic_reduce_method,
    elastic_loss_type=train_config.elastic_loss_type,
    use_elastic_loss=train_config.use_elastic_loss,
    use_background_loss=train_config.use_background_loss,
    use_warp_reg_loss=train_config.use_warp_reg_loss,
)
ptrain_step = jax.pmap(
    train_step,
    axis_name='batch',
    devices=devices,
    # rng_key, state, batch, scalar_params.
    in_axes=(0, 0, 0, None),
    # Treat use_elastic_loss as compile-time static.
    donate_argnums=(2,),  # Donate the 'batch' argument.
)

###################################
# TRAIN A NERFIE!                 #
###################################
# @markdown This runs the training loop!

import mediapy
from nerfies import utils
from nerfies import visualization as viz


print_every_n_iterations = 100  # @param{type:'number'}
visualize_results_every_n_iterations = 500  # @param{type:'number'}
save_checkpoint_every_n_iterations = 1000  # @param{type:'number'}


logging.info('Starting training')
rng = rng + jax.process_index()  # Make random seed separate across hosts.
keys = random.split(rng, len(devices))
time_tracker = utils.TimeTracker()
time_tracker.tic('data', 'total')

for step, batch in zip(range(step, train_config.max_steps + 1), train_iter):
  time_tracker.toc('data')
  scalar_params = scalar_params.replace(
      learning_rate=learning_rate_sched(step),
      elastic_loss_weight=elastic_loss_weight_sched(step))
  warp_alpha = jax_utils.replicate(warp_alpha_sched(step), devices)
  state = state.replace(warp_alpha=warp_alpha)

  with time_tracker.record_time('train_step'):
    state, stats, keys = ptrain_step(keys, state, batch, scalar_params)
    time_tracker.toc('total')

  if step % print_every_n_iterations == 0:
    logging.info(
        'step=%d, warp_alpha=%.04f, %s',
        step, warp_alpha_sched(step), time_tracker.summary_str('last'))
    coarse_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['coarse'].items()])
    fine_metrics_str = ', '.join(
        [f'{k}={v.mean():.04f}' for k, v in stats['fine'].items()])
    logging.info('\tcoarse metrics: %s', coarse_metrics_str)
    if 'fine' in stats:
      logging.info('\tfine metrics: %s', fine_metrics_str)
  
  if step % visualize_results_every_n_iterations == 0:
    print(f'[step={step}] Training set visualization')
    eval_batch = next(train_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                        titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

    print(f'[step={step}] Validation set visualization')
    eval_batch = next(val_eval_iter)
    render = render_fn(state, eval_batch, rng=rng)
    rgb = render['rgb']
    acc = render['acc']
    depth_exp = render['depth']
    depth_med = render['med_depth']
    rgb_target = eval_batch['rgb']
    depth_med_viz = viz.colorize(depth_med, cmin=datasource.near, cmax=datasource.far)
    mediapy.show_images([rgb_target, rgb, depth_med_viz],
                       titles=['GT RGB', 'Pred RGB', 'Pred Depth'])

  if step % save_checkpoint_every_n_iterations == 0:
    training.save_checkpoint(checkpoint_dir, state)

  time_tracker.tic('data', 'total')

print("BOOPS")