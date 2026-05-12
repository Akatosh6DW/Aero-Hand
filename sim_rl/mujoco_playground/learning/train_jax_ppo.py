# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Train a PPO agent using JAX on the specified environment."""

import csv
import datetime
import functools
import inspect
import json
import os
import time
import warnings
from typing import Sequence


xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["MUJOCO_GL"] = "egl"
os.environ["PYOPENGL_PLATFORM"] = "egl"

from absl import app
from absl import flags
from absl import logging
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo import networks_vision as ppo_networks_vision
from brax.training.agents.ppo import train as ppo
from etils import epath
import jax
import jax.numpy as jp
import mediapy as media
from ml_collections import config_dict
import mujoco_playground
from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import dm_control_suite_params
from mujoco_playground.config import locomotion_params
from mujoco_playground.config import manipulation_params
import mujoco
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import tensorboardX
import wandb

# Ignore the info logs from brax
logging.set_verbosity(logging.WARNING)

# Suppress warnings

# Suppress RuntimeWarnings from JAX
warnings.filterwarnings("ignore", category=RuntimeWarning, module="jax")
# Suppress DeprecationWarnings from JAX
warnings.filterwarnings("ignore", category=DeprecationWarning, module="jax")
# Suppress UserWarnings from absl (used by JAX and TensorFlow)
warnings.filterwarnings("ignore", category=UserWarning, module="absl")


def _merge_config_dict(
    base: config_dict.ConfigDict, overlay: dict | config_dict.ConfigDict
) -> config_dict.ConfigDict:
  """Recursively merges `overlay` into `base` while preserving new defaults."""
  if isinstance(overlay, config_dict.ConfigDict):
    overlay = overlay.to_dict()
  for key, value in overlay.items():
    if isinstance(value, dict):
      if key not in base or not isinstance(base[key], config_dict.ConfigDict):
        base[key] = config_dict.ConfigDict()
      _merge_config_dict(base[key], value)
    else:
      if value is None and key in base and base[key] is not None:
        continue
      base[key] = value
  return base


def _merge_network_factory_from_checkpoint(
    ppo_params: config_dict.ConfigDict,
    checkpoint_path: epath.Path,
) -> None:
  """Merges saved PPO network config into current defaults for restore/play."""
  net_cfg_path = checkpoint_path / "ppo_network_config.json"
  if not net_cfg_path.exists():
    return

  with open(net_cfg_path, "r", encoding="utf-8") as fp:
    net_cfg = json.load(fp)

  allowed_keys = {
      "policy_hidden_layer_sizes",
      "value_hidden_layer_sizes",
      "policy_obs_key",
      "value_obs_key",
      "distribution_type",
      "noise_std_type",
      "init_noise_std",
      "state_dependent_std",
      "mean_clip_scale",
      "use_distributional_critic",
      "num_quantiles",
  }
  for key, value in net_cfg.get("network_factory_kwargs", {}).items():
    if key in allowed_keys:
      ppo_params.network_factory[key] = value

  if "normalize_observations" in net_cfg:
    ppo_params.normalize_observations = bool(net_cfg["normalize_observations"])


_ENV_NAME = flags.DEFINE_string(
    "env_name",
    "LeapCubeReorient",
    f"Name of the environment. One of {', '.join(registry.ALL_ENVS)}",
)
_IMPL = flags.DEFINE_enum("impl", "jax", ["jax", "warp"], "MJX implementation")
_VISION = flags.DEFINE_boolean("vision", False, "Use vision input")
_LOAD_CHECKPOINT_PATH = flags.DEFINE_string(
    "load_checkpoint_path", None, "Path to load checkpoint from"
)
_IGNORE_CHECKPOINT_ENV_CONFIG = flags.DEFINE_boolean(
    "ignore_checkpoint_env_config",
    False,
    "Restore checkpoint weights without merging the checkpoint env config.",
)
_SUFFIX = flags.DEFINE_string("suffix", None, "Suffix for the experiment name")
_PLAY_ONLY = flags.DEFINE_boolean(
    "play_only", False, "If true, only play with the model and do not train"
)
_USE_WANDB = flags.DEFINE_boolean(
    "use_wandb",
    False,
    "Use Weights & Biases for logging (ignored in play-only mode)",
)
_USE_TB = flags.DEFINE_boolean(
    "use_tb", False, "Use TensorBoard for logging (ignored in play-only mode)"
)
_DOMAIN_RANDOMIZATION = flags.DEFINE_boolean(
    "domain_randomization", False, "Use domain randomization"
)
_SEED = flags.DEFINE_integer("seed", 1, "Random seed")
_NUM_TIMESTEPS = flags.DEFINE_integer(
    "num_timesteps", 1_000_000, "Number of timesteps"
)
_NUM_VIDEOS = flags.DEFINE_integer(
    "num_videos", 1, "Number of videos to record after training."
)
_CAMERA = flags.DEFINE_string(
    "camera", None, "Camera name for video rendering (default: free camera)."
)
_RENDER_MODE = flags.DEFINE_enum(
    "render_mode",
    "visual",
    ["visual", "collision", "overlay"],
    "Video render mode: STL only, collision only, or both.",
)
_RENDER_COLLISION_DEBUG = flags.DEFINE_boolean(
    "render_collision_debug",
    False,
    "Also render rollout*_collision_debug.mp4 with collision geoms visible.",
)
_RENDER_FORCE_HUD = flags.DEFINE_boolean(
    "render_force_hud",
    False,
    "Overlay explicit xfrc_applied arrow and numeric HUD in rendered videos.",
)
_RENDER_FORCE_HUD_SCALE_N = flags.DEFINE_float(
    "render_force_hud_scale_n",
    0.0,
    "Force scale in Newtons for the HUD arrow. <= 0 uses an inferred scale.",
)
_NUM_EVALS = flags.DEFINE_integer("num_evals", 5, "Number of evaluations")
_REWARD_SCALING = flags.DEFINE_float("reward_scaling", 0.1, "Reward scaling")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 1000, "Episode length")
_NORMALIZE_OBSERVATIONS = flags.DEFINE_boolean(
    "normalize_observations", True, "Normalize observations"
)
_ACTION_REPEAT = flags.DEFINE_integer("action_repeat", 1, "Action repeat")
_UNROLL_LENGTH = flags.DEFINE_integer("unroll_length", 10, "Unroll length")
_NUM_MINIBATCHES = flags.DEFINE_integer(
    "num_minibatches", 8, "Number of minibatches"
)
_NUM_UPDATES_PER_BATCH = flags.DEFINE_integer(
    "num_updates_per_batch", 8, "Number of updates per batch"
)
_DISCOUNTING = flags.DEFINE_float("discounting", 0.97, "Discounting")
_LEARNING_RATE = flags.DEFINE_float("learning_rate", 5e-4, "Learning rate")
_ENTROPY_COST = flags.DEFINE_float("entropy_cost", 5e-3, "Entropy cost")
_ENTROPY_WARMUP_COST = flags.DEFINE_float(
  "entropy_warmup_cost",
  -1.0,
  "Entropy cost for warmup stage. Set > 0 to enable auto entropy switch.",
)
_ENTROPY_WARMUP_RATIO = flags.DEFINE_float(
  "entropy_warmup_ratio",
  0.0,
  "Warmup stage ratio in [0,1). Example: 0.2 means first 20% steps use entropy_warmup_cost.",
)
_NUM_ENVS = flags.DEFINE_integer("num_envs", 1024, "Number of environments")
_NUM_EVAL_ENVS = flags.DEFINE_integer(
    "num_eval_envs", 128, "Number of evaluation environments"
)
_BATCH_SIZE = flags.DEFINE_integer("batch_size", 256, "Batch size")
_MAX_GRAD_NORM = flags.DEFINE_float("max_grad_norm", 1.0, "Max grad norm")
_CLIPPING_EPSILON = flags.DEFINE_float(
    "clipping_epsilon", 0.2, "Clipping epsilon for PPO"
)
_POLICY_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "policy_hidden_layer_sizes",
    [64, 64, 64],
    "Policy hidden layer sizes",
)
_VALUE_HIDDEN_LAYER_SIZES = flags.DEFINE_list(
    "value_hidden_layer_sizes",
    [64, 64, 64],
    "Value hidden layer sizes",
)
_POLICY_OBS_KEY = flags.DEFINE_string(
    "policy_obs_key", "state", "Policy obs key"
)
_VALUE_OBS_KEY = flags.DEFINE_string("value_obs_key", "state", "Value obs key")
_RSCOPE_ENVS = flags.DEFINE_integer(
    "rscope_envs",
    None,
    "Number of parallel environment rollouts to save for the rscope viewer",
)
_DETERMINISTIC_RSCOPE = flags.DEFINE_boolean(
    "deterministic_rscope",
    True,
    "Run deterministic rollouts for the rscope viewer",
)
_RUN_EVALS = flags.DEFINE_boolean(
    "run_evals",
    True,
    "Run evaluation rollouts between policy updates.",
)
_LOG_TRAINING_METRICS = flags.DEFINE_boolean(
    "log_training_metrics",
    False,
    "Whether to log training metrics and callback to progress_fn. Significantly"
    " slows down training if too frequent.",
)
_TRAINING_METRICS_STEPS = flags.DEFINE_integer(
    "training_metrics_steps",
    1_000_000,
    "Number of steps between logging training metrics. Increase if training"
    " experiences slowdown.",
)


def get_rl_config(env_name: str) -> config_dict.ConfigDict:
  if env_name in mujoco_playground.manipulation._envs:
    if _VISION.value:
      return manipulation_params.brax_vision_ppo_config(env_name, _IMPL.value)
    return manipulation_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.locomotion._envs:
    return locomotion_params.brax_ppo_config(env_name, _IMPL.value)
  elif env_name in mujoco_playground.dm_control_suite._envs:
    if _VISION.value:
      return dm_control_suite_params.brax_vision_ppo_config(
          env_name, _IMPL.value
      )
    return dm_control_suite_params.brax_ppo_config(env_name, _IMPL.value)

  raise ValueError(f"Env {env_name} not found in {registry.ALL_ENVS}.")


def rscope_fn(full_states, obs, rew, done):
  """
  All arrays are of shape (unroll_length, rscope_envs, ...)
  full_states: dict with keys 'qpos', 'qvel', 'time', 'metrics'
  obs: nd.array or dict obs based on env configuration
  rew: nd.array rewards
  done: nd.array done flags
  """
  # Calculate cumulative rewards per episode, stopping at first done flag
  done_mask = jp.cumsum(done, axis=0)
  valid_rewards = rew * (done_mask == 0)
  episode_rewards = jp.sum(valid_rewards, axis=0)
  print(
      "Collected rscope rollouts with reward"
      f" {episode_rewards.mean():.3f} +- {episode_rewards.std():.3f}"
  )


def main(argv):
  """Run training and evaluation for the specified environment."""

  del argv

  def _make_render_scene_option(render_mode: str) -> mujoco.MjvOption:
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[0] = 1  # world
    scene_option.geomgroup[1] = 1  # object
    scene_option.geomgroup[2] = int(render_mode in ("visual", "overlay"))
    scene_option.geomgroup[3] = int(render_mode in ("collision", "overlay"))
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = int(
        render_mode == "overlay"
    )
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = int(
        render_mode in ("collision", "overlay")
    )
    return scene_option

  def _infer_force_hud_scale(env) -> float:
    pcfg = getattr(env, "_config", None).perturbation_config
    total_force_clip = float(getattr(pcfg, "total_force_clip_n", 0.0))
    if total_force_clip > 0.0:
      return total_force_clip
    cube_mass = float(getattr(env, "_cube_mass", 0.0))
    gravity_force = cube_mass * 9.81
    return max(
        0.1,
        float(getattr(pcfg, "external_force_magnitude", 0.0)),
        gravity_force * float(getattr(pcfg, "orientation_flip_force_scale", 0.0)),
        gravity_force * np.sin(float(getattr(pcfg, "gravity_tilt_max_rad", 0.0))),
    )

  def _draw_force_hud(
      frames: Sequence[np.ndarray],
      traj: Sequence,
      body_id: int,
      scale_n: float,
      camera_name: str,
  ) -> list[np.ndarray]:
    hud_frames = []
    scale_n = max(scale_n, 1e-6)
    font = ImageFont.load_default()

    def _draw_arrow(draw, start, end, color, width):
      draw.line([start, end], fill=color, width=width)
      dx = end[0] - start[0]
      dy = end[1] - start[1]
      length = float(np.hypot(dx, dy))
      if length < 1e-6:
        return
      ux, uy = dx / length, dy / length
      head_len = min(14.0, max(8.0, length * 0.28))
      px, py = -uy, ux
      tip = np.array(end, dtype=np.float32)
      base = tip - head_len * np.array([ux, uy], dtype=np.float32)
      left = base + 5.0 * np.array([px, py], dtype=np.float32)
      right = base - 5.0 * np.array([px, py], dtype=np.float32)
      draw.polygon(
          [tuple(tip), tuple(left), tuple(right)],
          fill=color,
      )

    for frame, state in zip(frames, traj):
      img = Image.fromarray(np.ascontiguousarray(frame.copy()))
      draw = ImageDraw.Draw(img, "RGBA")
      force = np.asarray(state.data.xfrc_applied[body_id, :3], dtype=np.float32)
      torque = np.asarray(state.data.xfrc_applied[body_id, 3:6], dtype=np.float32)
      force_norm = float(np.linalg.norm(force))
      torque_norm = float(np.linalg.norm(torque))

      w, h = img.size
      panel_x0, panel_y0 = 18, 18
      panel_w = min(360, w - 36)
      panel_h = 184
      draw.rounded_rectangle(
          (panel_x0, panel_y0, panel_x0 + panel_w, panel_y0 + panel_h),
          radius=10,
          fill=(18, 18, 18, 196),
          outline=(180, 180, 180, 255),
          width=1,
      )
      draw.text(
          (panel_x0 + 12, panel_y0 + 10),
          f"xfrc_applied HUD ({camera_name})",
          fill=(255, 255, 255, 255),
          font=font,
      )

      center = np.array([panel_x0 + 82, panel_y0 + 104], dtype=np.int32)
      arrow_radius = 54
      draw.ellipse(
          (
              center[0] - arrow_radius,
              center[1] - arrow_radius,
              center[0] + arrow_radius,
              center[1] + arrow_radius,
          ),
          outline=(110, 110, 110, 255),
          width=1,
      )
      draw.line(
          [(center[0] - arrow_radius, center[1]), (center[0] + arrow_radius, center[1])],
          fill=(90, 90, 90, 255),
          width=1,
      )
      draw.line(
          [(center[0], center[1] - arrow_radius), (center[0], center[1] + arrow_radius)],
          fill=(90, 90, 90, 255),
          width=1,
      )
      draw.text((center[0] + arrow_radius + 8, center[1] - 6), "+X", fill=(255, 120, 120, 255), font=font)
      draw.text((center[0] - 8, center[1] - arrow_radius - 16), "+Y", fill=(120, 255, 120, 255), font=font)

      xy_force = force[:2]
      xy_scale = float(np.linalg.norm(xy_force) / scale_n)
      xy_scale = min(xy_scale, 1.0)
      arrow_vec = np.array([
          xy_force[0],
          -xy_force[1],
      ], dtype=np.float32)
      arrow_len = float(np.linalg.norm(arrow_vec))
      if arrow_len > 1e-6:
        arrow_dir = arrow_vec / arrow_len
        arrow_px = center + np.round(arrow_dir * arrow_radius * xy_scale).astype(np.int32)
        _draw_arrow(draw, tuple(center), tuple(arrow_px), (255, 90, 90, 255), 3)
      else:
        draw.ellipse(
            (center[0] - 4, center[1] - 4, center[0] + 4, center[1] + 4),
            fill=(160, 160, 160, 255),
        )

      z_base_x = panel_x0 + 176
      z_base_y0 = panel_y0 + 148
      z_base_y1 = panel_y0 + 60
      draw.line(
          [(z_base_x, z_base_y0), (z_base_x, z_base_y1)],
          fill=(90, 90, 90, 255),
          width=2,
      )
      draw.text((z_base_x + 10, z_base_y1 - 6), "+Z", fill=(120, 180, 255, 255), font=font)
      z_scale = min(abs(float(force[2])) / scale_n, 1.0)
      z_target_y = int(round(z_base_y0 - (z_base_y0 - z_base_y1) * z_scale))
      z_color = (120, 180, 255) if force[2] >= 0.0 else (255, 200, 120)
      if abs(float(force[2])) > 1e-6:
        _draw_arrow(
            draw,
            (z_base_x, z_base_y0),
            (z_base_x, z_target_y),
            (*z_color, 255),
            3,
        )
      else:
        draw.ellipse(
            (z_base_x - 4, z_base_y0 - 4, z_base_x + 4, z_base_y0 + 4),
            fill=(160, 160, 160, 255),
        )

      lines = [
          f"|F|   = {force_norm:6.3f} N",
          f"Fx    = {force[0]:+6.3f} N",
          f"Fy    = {force[1]:+6.3f} N",
          f"Fz    = {force[2]:+6.3f} N",
          f"|tau| = {torque_norm:6.3f} Nm",
          f"t     = {float(state.data.time):6.2f} s",
          f"scale = {scale_n:5.3f} N",
      ]
      text_x = panel_x0 + 220
      for idx, line in enumerate(lines):
        draw.text(
            (text_x, panel_y0 + 28 + idx * 20),
            line,
            fill=(255, 255, 255, 255),
            font=font,
        )

      status = "ACTIVE" if force_norm > 1e-6 else "idle"
      status_color = (255, 100, 100) if force_norm > 1e-6 else (180, 180, 180)
      draw.text(
          (panel_x0 + 12, panel_y0 + panel_h - 22),
          f"force status: {status}",
          fill=(*status_color, 255),
          font=font,
      )
      hud_frames.append(np.asarray(img))
    return hud_frames

  def _resolve_latest_checkpoint_dir(path: epath.Path) -> epath.Path:
    """Return latest concrete checkpoint directory under `path` if available."""
    if not path.exists() or not path.is_dir():
      raise FileNotFoundError(f"Checkpoint directory does not exist: {path}")
    children = [p for p in path.iterdir() if p.is_dir() and p.name.isdigit()]
    if not children:
      raise FileNotFoundError(
          "No numbered checkpoint directories found under "
          f"{path}."
      )
    children.sort(key=lambda p: int(p.name))
    return children[-1]

  # Load environment configuration
  env_cfg = registry.get_default_config(_ENV_NAME.value)
  env_cfg["impl"] = _IMPL.value

  if _LOAD_CHECKPOINT_PATH.value is not None and not _IGNORE_CHECKPOINT_ENV_CONFIG.value:
    ckpt_path_for_cfg = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path_for_cfg.is_dir():
      if not (
          (ckpt_path_for_cfg / "_CHECKPOINT_METADATA").exists()
          or (ckpt_path_for_cfg / "manifest.ocdbt").exists()
      ):
        try:
          ckpt_path_for_cfg = _resolve_latest_checkpoint_dir(ckpt_path_for_cfg)
        except FileNotFoundError:
          ckpt_path_for_cfg = ckpt_path_for_cfg
      config_path = ckpt_path_for_cfg / "config.json"
      if not config_path.exists():
        parent_config = ckpt_path_for_cfg.parent / "config.json"
        if parent_config.exists():
          config_path = parent_config
      if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as fp:
          saved_env_cfg = json.load(fp)
        env_cfg = _merge_config_dict(env_cfg, saved_env_cfg)
        env_cfg["impl"] = _IMPL.value
        print(f"Merged checkpoint env config from: {config_path}")

  ppo_params = get_rl_config(_ENV_NAME.value)
  if _LOAD_CHECKPOINT_PATH.value is not None:
    ckpt_path_for_net = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path_for_net.is_dir() and not (
        (ckpt_path_for_net / "_CHECKPOINT_METADATA").exists()
        or (ckpt_path_for_net / "manifest.ocdbt").exists()
    ):
      try:
        ckpt_path_for_net = _resolve_latest_checkpoint_dir(ckpt_path_for_net)
      except FileNotFoundError:
        pass
    if ckpt_path_for_net.is_dir():
      _merge_network_factory_from_checkpoint(ppo_params, ckpt_path_for_net)

  if _NUM_TIMESTEPS.present:
    ppo_params.num_timesteps = _NUM_TIMESTEPS.value
  if _PLAY_ONLY.present:
    ppo_params.num_timesteps = 0
  if _NUM_EVALS.present:
    ppo_params.num_evals = _NUM_EVALS.value
  if _REWARD_SCALING.present:
    ppo_params.reward_scaling = _REWARD_SCALING.value
  if _EPISODE_LENGTH.present:
    ppo_params.episode_length = _EPISODE_LENGTH.value
  if _NORMALIZE_OBSERVATIONS.present:
    ppo_params.normalize_observations = _NORMALIZE_OBSERVATIONS.value
  if _ACTION_REPEAT.present:
    ppo_params.action_repeat = _ACTION_REPEAT.value
  if _UNROLL_LENGTH.present:
    ppo_params.unroll_length = _UNROLL_LENGTH.value
  if _NUM_MINIBATCHES.present:
    ppo_params.num_minibatches = _NUM_MINIBATCHES.value
  if _NUM_UPDATES_PER_BATCH.present:
    ppo_params.num_updates_per_batch = _NUM_UPDATES_PER_BATCH.value
  if _DISCOUNTING.present:
    ppo_params.discounting = _DISCOUNTING.value
  if _LEARNING_RATE.present:
    ppo_params.learning_rate = _LEARNING_RATE.value
  if _ENTROPY_COST.present:
    ppo_params.entropy_cost = _ENTROPY_COST.value
  if _NUM_ENVS.present:
    ppo_params.num_envs = _NUM_ENVS.value
  if _NUM_EVAL_ENVS.present:
    ppo_params.num_eval_envs = _NUM_EVAL_ENVS.value
  if _BATCH_SIZE.present:
    ppo_params.batch_size = _BATCH_SIZE.value
  if _MAX_GRAD_NORM.present:
    ppo_params.max_grad_norm = _MAX_GRAD_NORM.value
  if _CLIPPING_EPSILON.present:
    ppo_params.clipping_epsilon = _CLIPPING_EPSILON.value
  if _POLICY_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.policy_hidden_layer_sizes = list(
        map(int, _POLICY_HIDDEN_LAYER_SIZES.value)
    )
  if _VALUE_HIDDEN_LAYER_SIZES.present:
    ppo_params.network_factory.value_hidden_layer_sizes = list(
        map(int, _VALUE_HIDDEN_LAYER_SIZES.value)
    )
  if _POLICY_OBS_KEY.present:
    ppo_params.network_factory.policy_obs_key = _POLICY_OBS_KEY.value
  if _VALUE_OBS_KEY.present:
    ppo_params.network_factory.value_obs_key = _VALUE_OBS_KEY.value
  if _VISION.value:
    env_cfg.vision = True
    env_cfg.vision_config.render_batch_size = ppo_params.num_envs
  env = registry.load(_ENV_NAME.value, config=env_cfg)
  if _RUN_EVALS.present:
    ppo_params.run_evals = _RUN_EVALS.value
  if _LOG_TRAINING_METRICS.present:
    ppo_params.log_training_metrics = _LOG_TRAINING_METRICS.value
  if _TRAINING_METRICS_STEPS.present:
    ppo_params.training_metrics_steps = _TRAINING_METRICS_STEPS.value

  print(f"Environment Config:\n{env_cfg}")
  print(f"PPO Training Parameters:\n{ppo_params}")

  # Generate unique experiment name
  now = datetime.datetime.now()
  timestamp = now.strftime("%Y%m%d-%H%M%S")
  exp_name = f"{_ENV_NAME.value}-{timestamp}"
  if _SUFFIX.value is not None:
    exp_name += f"-{_SUFFIX.value}"
  print(f"Experiment name: {exp_name}")

  # Set up logging directory
  logdir = epath.Path("logs").resolve() / exp_name
  logdir.mkdir(parents=True, exist_ok=True)
  print(f"Logs are being stored in: {logdir}")

  # Initialize Weights & Biases if required
  if _USE_WANDB.value and not _PLAY_ONLY.value:
    wandb.init(project="mjxrl", name=exp_name)
    wandb.config.update(env_cfg.to_dict())
    wandb.config.update({"env_name": _ENV_NAME.value})

  # Initialize TensorBoard if required
  if _USE_TB.value and not _PLAY_ONLY.value:
    writer = tensorboardX.SummaryWriter(logdir)

  # Handle checkpoint loading
  if _LOAD_CHECKPOINT_PATH.value is not None:
    # Convert to absolute path
    ckpt_path = epath.Path(_LOAD_CHECKPOINT_PATH.value).resolve()
    if ckpt_path.is_dir():
      is_concrete_ckpt = (
          (ckpt_path / "_CHECKPOINT_METADATA").exists()
          or (ckpt_path / "manifest.ocdbt").exists()
      )
      if is_concrete_ckpt:
        restore_checkpoint_path = ckpt_path
        print(f"Restoring from checkpoint: {restore_checkpoint_path}")
      else:
        latest_ckpt = _resolve_latest_checkpoint_dir(ckpt_path)
        restore_checkpoint_path = latest_ckpt
        print(f"Restoring from: {restore_checkpoint_path}")
    else:
      restore_checkpoint_path = ckpt_path
      print(f"Restoring from checkpoint: {restore_checkpoint_path}")
  else:
    print("No checkpoint path provided, not restoring from checkpoint")
    restore_checkpoint_path = None

  # Set up checkpoint directory
  ckpt_path = logdir / "checkpoints"
  ckpt_path.mkdir(parents=True, exist_ok=True)
  print(f"Checkpoint path: {ckpt_path}")

  # Set up CSV metrics logging (always enabled, no flag needed)
  _csv_file = open(logdir / "metrics.csv", "w", newline="", buffering=1)
  _csv_state = {"writer": None}

  # Save environment configuration
  with open(ckpt_path / "config.json", "w", encoding="utf-8") as fp:
    json.dump(env_cfg.to_dict(), fp, indent=4)

  training_params = dict(ppo_params)
  if "network_factory" in training_params:
    del training_params["network_factory"]

  network_fn = (
      ppo_networks_vision.make_ppo_networks_vision
      if _VISION.value
      else ppo_networks.make_ppo_networks
  )
  if hasattr(ppo_params, "network_factory"):
    network_factory_kwargs = dict(ppo_params.network_factory)
    supported_network_kwargs = set(inspect.signature(network_fn).parameters)
    unsupported_network_kwargs = sorted(
        set(network_factory_kwargs) - supported_network_kwargs
    )
    if unsupported_network_kwargs:
      print(
          "Dropping unsupported network factory kwargs for current brax: "
          f"{unsupported_network_kwargs}"
      )
      for key in unsupported_network_kwargs:
        network_factory_kwargs.pop(key, None)
    network_factory = functools.partial(
        network_fn, **network_factory_kwargs
    )
  else:
    network_factory = network_fn

  if _DOMAIN_RANDOMIZATION.value:
    training_params["randomization_fn"] = registry.get_domain_randomizer(
        _ENV_NAME.value
    )

  if _VISION.value:
    env = wrapper.wrap_for_brax_training(
        env,
        vision=True,
        num_vision_envs=env_cfg.vision_config.render_batch_size,
        episode_length=ppo_params.episode_length,
        action_repeat=ppo_params.action_repeat,
        randomization_fn=training_params.get("randomization_fn"),
    )

  num_eval_envs = (
      ppo_params.num_envs
      if _VISION.value
      else ppo_params.get("num_eval_envs", 128)
  )

  def _make_train_fn(stage_params, restore_path, save_path):
    stage_params = dict(stage_params)
    if "num_eval_envs" in stage_params:
      del stage_params["num_eval_envs"]
    return functools.partial(
        ppo.train,
        **stage_params,
        network_factory=network_factory,
        seed=_SEED.value,
        restore_checkpoint_path=restore_path,
        save_checkpoint_path=save_path,
        wrap_env_fn=None if _VISION.value else wrapper.wrap_for_brax_training,
        num_eval_envs=num_eval_envs,
    )

  times = [time.monotonic()]

  # Progress function for logging
  def progress(num_steps, metrics):
    times.append(time.monotonic())

    # Log to Weights & Biases
    if _USE_WANDB.value and not _PLAY_ONLY.value:
      wandb.log(metrics, step=num_steps)

    # Log to TensorBoard
    if _USE_TB.value and not _PLAY_ONLY.value:
      for key, value in metrics.items():
        writer.add_scalar(key, value, num_steps)
      writer.flush()

    # Log to CSV (always)
    if not _PLAY_ONLY.value:
      row = {"num_steps": num_steps}
      row.update({k: float(v) for k, v in metrics.items()})
      if _csv_state["writer"] is None:
        _csv_state["writer"] = csv.DictWriter(
            _csv_file,
            fieldnames=["num_steps"] + sorted(metrics.keys()),
            extrasaction="ignore",
        )
        _csv_state["writer"].writeheader()
      _csv_state["writer"].writerow(row)
      _csv_file.flush()

    if _RUN_EVALS.value:
      eval_reward = None
      # Different environments/backends can expose different reward key names.
      for k in (
          "eval/episode_reward",
          "eval/episode/sum_reward",
          "eval/episode_return",
          "episode/sum_reward",
      ):
        if k in metrics:
          eval_reward = metrics[k]
          break

      if eval_reward is None:
        reward_like = [k for k in metrics.keys() if "reward" in k]
        if reward_like:
          k = sorted(reward_like)[0]
          eval_reward = metrics[k]
          print(f"{num_steps}: reward={eval_reward:.3f} ({k})")
        else:
          print(f"{num_steps}: metrics logged (no reward key)")
      else:
        print(f"{num_steps}: reward={eval_reward:.3f}")
    if _LOG_TRAINING_METRICS.value:
      if "episode/sum_reward" in metrics:
        print(
            f"{num_steps}: mean episode"
            f" reward={metrics['episode/sum_reward']:.3f}"
        )

  # Load evaluation environment.
  eval_env = None
  if not _VISION.value:
    eval_env = registry.load(_ENV_NAME.value, config=env_cfg)
  num_envs = 1
  if _VISION.value:
    num_envs = env_cfg.vision_config.render_batch_size

  policy_params_fn = lambda *args: None
  if _RSCOPE_ENVS.value:
    # Interactive visualisation of policy checkpoints
    from rscope import brax as rscope_utils

    if not _VISION.value:
      rscope_env = registry.load(_ENV_NAME.value, config=env_cfg)
      rscope_env = wrapper.wrap_for_brax_training(
          rscope_env,
          episode_length=ppo_params.episode_length,
          action_repeat=ppo_params.action_repeat,
          randomization_fn=training_params.get("randomization_fn"),
      )
    else:
      rscope_env = env

    rscope_handle = rscope_utils.BraxRolloutSaver(
        rscope_env,
        ppo_params,
        _VISION.value,
        _RSCOPE_ENVS.value,
        _DETERMINISTIC_RSCOPE.value,
        jax.random.PRNGKey(_SEED.value),
        rscope_fn,
    )

    def policy_params_fn(current_step, make_policy, params):  # pylint: disable=unused-argument
      rscope_handle.set_make_policy(make_policy)
      rscope_handle.dump_rollout(params)

  use_entropy_warmup = (
      _ENTROPY_WARMUP_COST.value > 0.0
      and _ENTROPY_WARMUP_RATIO.value > 0.0
      and ppo_params.num_timesteps > 0
      and not _PLAY_ONLY.value
  )
  if _ENTROPY_WARMUP_RATIO.value < 0.0 or _ENTROPY_WARMUP_RATIO.value >= 1.0:
    raise ValueError("entropy_warmup_ratio must be in [0, 1).")

  if use_entropy_warmup:
    warmup_steps = int(round(ppo_params.num_timesteps * _ENTROPY_WARMUP_RATIO.value))
    warmup_steps = max(1, min(warmup_steps, ppo_params.num_timesteps - 1))
    main_steps = ppo_params.num_timesteps - warmup_steps

    stage1_ckpt_path = ckpt_path / "entropy_warmup"
    stage2_ckpt_path = ckpt_path / "entropy_main"
    stage1_ckpt_path.mkdir(parents=True, exist_ok=True)
    stage2_ckpt_path.mkdir(parents=True, exist_ok=True)

    print(
        "Entropy auto-switch enabled: "
        f"warmup {warmup_steps} steps @ {_ENTROPY_WARMUP_COST.value}, "
        f"then {main_steps} steps @ {ppo_params.entropy_cost}."
    )

    stage1_params = dict(training_params)
    stage1_params["num_timesteps"] = warmup_steps
    stage1_params["entropy_cost"] = float(_ENTROPY_WARMUP_COST.value)
    train_fn_stage1 = _make_train_fn(
        stage1_params,
        restore_checkpoint_path,
        stage1_ckpt_path,
    )
    _, _, _ = train_fn_stage1(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )

    stage2_params = dict(training_params)
    stage2_params["num_timesteps"] = main_steps
    stage2_params["entropy_cost"] = float(ppo_params.entropy_cost)
    stage1_restore_path = _resolve_latest_checkpoint_dir(stage1_ckpt_path)
    print(f"Restoring stage-2 from: {stage1_restore_path}")
    train_fn_stage2 = _make_train_fn(
        stage2_params,
      stage1_restore_path,
        stage2_ckpt_path,
    )
    make_inference_fn, params, _ = train_fn_stage2(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )
  else:
    train_fn = _make_train_fn(training_params, restore_checkpoint_path, ckpt_path)
    # Train or load the model
    make_inference_fn, params, _ = train_fn(  # pylint: disable=no-value-for-parameter
        environment=env,
        progress_fn=progress,
        policy_params_fn=policy_params_fn,
        eval_env=eval_env,
    )

  print("Done training.")
  if len(times) > 1:
    print(f"Time to JIT compile: {times[1] - times[0]}")
    print(f"Time to train: {times[-1] - times[1]}")

  if _NUM_VIDEOS.value <= 0:
    print("Skipping inference/rendering because --num_videos <= 0.")
    return

  print("Starting inference...")

  # Create inference function.
  inference_fn = make_inference_fn(params, deterministic=True)
  jit_inference_fn = jax.jit(inference_fn)

  # Run evaluation rollouts.
  def do_rollout(rng, state):
    empty_data = state.data.__class__(
        **{k: None for k in state.data.__annotations__}
    )  # pytype: disable=attribute-error
    empty_traj = state.__class__(**{k: None for k in state.__annotations__})  # pytype: disable=attribute-error
    empty_traj = empty_traj.replace(data=empty_data)

    def step(carry, _):
      state, rng = carry
      rng, act_key = jax.random.split(rng)
      act = jit_inference_fn(state.obs, act_key)[0]
      state = eval_env.step(state, act)
      traj_data = empty_traj.tree_replace({
          "data.qpos": state.data.qpos,
          "data.qvel": state.data.qvel,
          "data.time": state.data.time,
          "data.ctrl": state.data.ctrl,
          "data.mocap_pos": state.data.mocap_pos,
          "data.mocap_quat": state.data.mocap_quat,
          "data.xfrc_applied": state.data.xfrc_applied,
      })
      if _VISION.value:
        traj_data = jax.tree_util.tree_map(lambda x: x[0], traj_data)
      return (state, rng), traj_data

    _, traj = jax.lax.scan(
        step, (state, rng), None, length=_EPISODE_LENGTH.value
    )
    return traj

  rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
  reset_states = jax.jit(jax.vmap(eval_env.reset))(rng)
  if _VISION.value:
    reset_states = jax.tree_util.tree_map(lambda x: x[0], reset_states)
  traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)
  trajectories = [None] * _NUM_VIDEOS.value
  for i in range(_NUM_VIDEOS.value):
    t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
    trajectories[i] = [
        jax.tree.map(lambda x, j=j: x[j], t)
        for j in range(_EPISODE_LENGTH.value)
    ]

  # Render and save the rollout.
  render_every = 2
  fps = 1.0 / eval_env.dt / render_every
  print(f"FPS for rendering: {fps}")
  scene_option = _make_render_scene_option(_RENDER_MODE.value)
  force_hud_body_id = None
  force_hud_scale_n = None
  if _RENDER_FORCE_HUD.value:
    force_hud_body_id = int(eval_env.mj_model.body("cube").id)
    force_hud_scale_n = (
        float(_RENDER_FORCE_HUD_SCALE_N.value)
        if _RENDER_FORCE_HUD_SCALE_N.value > 0.0
        else _infer_force_hud_scale(eval_env)
    )
    print(
        "Force HUD enabled: "
        f"body='cube', scale={force_hud_scale_n:.3f}N"
    )
  for i, rollout in enumerate(trajectories):
    traj = rollout[::render_every]
    frames = eval_env.render(
        traj, height=480, width=640, camera=_CAMERA.value,
        scene_option=scene_option,
    )
    if _RENDER_FORCE_HUD.value:
      frames = _draw_force_hud(
          frames,
          traj,
          force_hud_body_id,
          force_hud_scale_n,
          _CAMERA.value or "free",
      )
    video_path = str(logdir / f"rollout{i}.mp4")
    media.write_video(video_path, frames, fps=fps)
    print(f"Rollout video saved as '{video_path}'.")
    if _RENDER_COLLISION_DEBUG.value:
      debug_scene_option = _make_render_scene_option("overlay")
      debug_frames = eval_env.render(
          traj, height=480, width=640, camera=_CAMERA.value,
          scene_option=debug_scene_option,
      )
      if _RENDER_FORCE_HUD.value:
        debug_frames = _draw_force_hud(
            debug_frames,
            traj,
            force_hud_body_id,
            force_hud_scale_n,
            _CAMERA.value or "free",
        )
      debug_video_path = str(logdir / f"rollout{i}_collision_debug.mp4")
      media.write_video(debug_video_path, debug_frames, fps=fps)
      print(f"Collision debug video saved as '{debug_video_path}'.")


if __name__ == "__main__":
  app.run(main)
