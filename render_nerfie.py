# ARGS
from nerfies import utils
from nerfies import visualization as viz
from nerfies import evaluation
import functools
from nerfies import training
from nerfies import schedules
from nerfies import model_utils
from nerfies import models
from nerfies import image_utils
from nerfies import datasets
from nerfies import configs
import gin
import mediapy
import imageio
import tempfile
import numpy as np
import random as pyrandom
from io import BytesIO
from absl import logging
from flax.training import checkpoints
from flax.metrics import tensorboard
from flax import optim
from flax import jax_utils
import flax.linen as nn
import flax
from jax import random
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.config import config as jax_config
import jax
from pathlib import Path

project_dir = Path('/mnt/e/2022/NeRF/Nerfies/cortines')
train_dir = project_dir / 'nerfie_trained'
data_dir = project_dir / 'nerfie_dataset'
camera_dir = data_dir / 'camera-paths/orbit-mild'

render_dir = project_dir / 'render-default-nerfies-cameras'

# @title Configure notebook runtime
# @markdown If you would like to use a GPU runtime instead, change the runtime type by going to `Runtime > Change runtime type`.
# @markdown You will have to use a smaller batch size on GPU.
runtime_type = 'gpu'  # @param ['gpu', 'tpu']
if runtime_type == 'tpu':
    import jax.tools.colab_tpu
    jax.tools.colab_tpu.setup_tpu()

print('Detected Devices:', jax.devices())

# @title Define imports and utility functions.


# Monkey patch logging.

def myprint(msg, *args, **kwargs):
    print(msg % args)


logging.info = myprint
logging.warn = myprint
logging.error = myprint

# @title Model and dataset configuration
# @markdown Change the directories to where you saved your capture and experiment.


checkpoint_dir = train_dir / 'checkpoints'
checkpoint_dir.mkdir(exist_ok=True, parents=True)

config_path = train_dir / 'config.gin'
with open(config_path, 'r') as f:
    logging.info('Loading config from %s', config_path)
    config_str = f.read()
gin.parse_config(config_str)

config_path = Path(train_dir, 'config.gin')
with open(config_path, 'w') as f:
    logging.info('Saving config to %s', config_path)
    f.write(config_str)

exp_config = configs.ExperimentConfig()
model_config = configs.ModelConfig()
train_config = configs.TrainConfig()
eval_config = configs.EvalConfig()


# @title Create datasource and show an example.


datasource = datasets.from_config(
    exp_config.datasource_spec,
    image_scale=exp_config.image_scale,
    use_appearance_id=model_config.use_appearance_metadata,
    use_camera_id=model_config.use_camera_metadata,
    use_warp_id=model_config.use_warp,
    random_seed=exp_config.random_seed,
)

# @title Initialize model
# @markdown Defines the model and initializes its parameters.


rng = random.PRNGKey(exp_config.random_seed)
np.random.seed(exp_config.random_seed + jax.process_index())
devices = jax.devices()

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
    use_weights=train_config.use_elastic_loss,
)

optimizer_def = optim.Adam(learning_rate_sched(0))
optimizer = optimizer_def.create(params)

state = model_utils.TrainState(
    optimizer=optimizer,
    warp_alpha=warp_alpha_sched(0),
)
logging.info('Restoring checkpoint from %s', checkpoint_dir)
state = checkpoints.restore_checkpoint(checkpoint_dir, state)
state = jax_utils.replicate(state, devices=devices)

del params

# @title Define pmapped render function.


def _model_fn(key_0, key_1, params, rays_dict, warp_extra):
    out = model.apply(
        {'params': params},
        rays_dict,
        warp_extra=warp_extra,
        rngs={
            'coarse': key_0,
            'fine': key_1
        },
        mutable=False,
    )
    return jax.lax.all_gather(out, axis_name='batch')


pmodel_fn = jax.pmap(
    # Note rng_keys are useless in eval mode since there's no randomness.
    _model_fn,
    in_axes=(0, 0, 0, 0, 0),  # Only distribute the data input.
    devices=devices,
    donate_argnums=(3,),  # Donate the 'rays' argument.
    axis_name='batch',
)

# Make random seed separate across hosts.
rng + jax.host_id()


def render_fn(rays_dict):
    _render_fn = functools.partial(
        evaluation.render_image,
        model_fn=pmodel_fn,
        device_count=len(devices),
        chunk=eval_config.chunk,
    )

    return _render_fn(state, rays_dict, rng=rng)

# @title Load cameras.


print(f'Loading cameras from {camera_dir}')
test_camera_paths = datasource.glob_cameras(camera_dir)
test_cameras = utils.parallel_map(
    datasource.load_camera, test_camera_paths, show_pbar=True)

# @title Render video frames.

rgb_imgs_dir = render_dir / 'rgb'
depth_imgs_dir = render_dir / 'depth'

render_dir.mkdir(exist_ok=True)
rgb_imgs_dir.mkdir(exist_ok=True)
depth_imgs_dir.mkdir(exist_ok=True)


results = []
for i in range(len(test_cameras)):
    rgb_img_path = rgb_imgs_dir / f"{i:05d}.png"
    depth_img_path = depth_imgs_dir / f"{i:05d}.png"
    if rgb_img_path.exists() and depth_img_path.exists():
        print(f'Frame {i+1}/{len(test_cameras)} alread exists.  Skipping...')
        continue

    print(f'Rendering frame {i + 1}/{len(test_cameras)}')
    camera = test_cameras[i]
    batch = datasets.camera_to_rays(camera)
    batch['metadata'] = {
        'appearance': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
        'warp': jnp.zeros_like(batch['origins'][..., 0, jnp.newaxis], jnp.uint32),
    }

    output = render_fn(batch)
    rgb = np.array(output['rgb'])
    depth_med = np.array(output['med_depth'])
    results.append((rgb, depth_med))
    depth_viz = viz.colorize(
        depth_med.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)

    mediapy.write_image(rgb_img_path, rgb)
    mediapy.write_image(depth_img_path, depth_viz)

    # @title Show rendered video.

fps = 30  # @param {type:'number'}

frames = []
for rgb, depth in results:
    depth_viz = viz.colorize(
        depth.squeeze(), cmin=datasource.near, cmax=datasource.far, invert=True)
    frame = np.concatenate([rgb, depth_viz], axis=1)
    frames.append(image_utils.image_to_uint8(frame))

mediapy.write_video(render_dir / 'out.mp4', frames, fps=fps)
