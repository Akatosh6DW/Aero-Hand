#!/usr/bin/env python3
"""Offline evaluator for AeroBottleGrasp checkpoints."""

import argparse
import functools
import json
from pathlib import Path

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train
from ml_collections import config_dict

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


def _to_config_dict(d):
  if isinstance(d, config_dict.ConfigDict):
    return d
  if isinstance(d, dict):
    cfg = config_dict.ConfigDict()
    for k, v in d.items():
      cfg[k] = _to_config_dict(v) if isinstance(v, dict) else v
    return cfg
  return d


def _merge_config_dict(
    base: config_dict.ConfigDict, overlay: dict | config_dict.ConfigDict
) -> config_dict.ConfigDict:
  if isinstance(overlay, config_dict.ConfigDict):
    overlay = overlay.to_dict()
  for k, v in overlay.items():
    if isinstance(v, dict):
      if k not in base or not isinstance(base[k], config_dict.ConfigDict):
        base[k] = config_dict.ConfigDict()
      _merge_config_dict(base[k], v)
    else:
      base[k] = v
  return base


def _load_policy(env_name: str, checkpoint_path: Path):
  env_cfg = registry.get_default_config(env_name)
  env_cfg_path = checkpoint_path / "config.json"
  if not env_cfg_path.exists():
    parent_cfg = checkpoint_path.parent / "config.json"
    if parent_cfg.exists():
      env_cfg_path = parent_cfg
  if env_cfg_path.exists():
    with open(env_cfg_path, "r", encoding="utf-8") as f:
      env_cfg = _merge_config_dict(env_cfg, json.load(f))
  env = registry.load(env_name, config=env_cfg)
  ppo_params = manipulation_params.brax_ppo_config(env_name)

  network_cfg_path = checkpoint_path / "ppo_network_config.json"
  if network_cfg_path.exists():
    with open(network_cfg_path, "r", encoding="utf-8") as f:
      network_cfg = json.load(f)
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
    merged_network_cfg = dict(ppo_params.network_factory)
    for k, v in network_cfg["network_factory_kwargs"].items():
      if k in allowed_keys:
        merged_network_cfg[k] = v
    ppo_params.network_factory = _to_config_dict(merged_network_cfg)
    ppo_params.normalize_observations = bool(
        network_cfg.get("normalize_observations", True)
    )

  network_factory_cfg = dict(ppo_params.network_factory)
  del ppo_params["network_factory"]
  if "num_timesteps" in ppo_params:
    del ppo_params["num_timesteps"]
  if "num_eval_envs" in ppo_params:
    del ppo_params["num_eval_envs"]
  if "num_envs" in ppo_params:
    ppo_params.num_envs = min(int(ppo_params.num_envs), 256)
  if "batch_size" in ppo_params:
    ppo_params.batch_size = min(int(ppo_params.batch_size), 128)
  if "num_minibatches" in ppo_params:
    ppo_params.num_minibatches = min(int(ppo_params.num_minibatches), 8)

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_factory_cfg
  )

  make_inference_fn, params, _ = ppo_train(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      num_timesteps=0,
      seed=1,
      restore_checkpoint_path=epath.Path(checkpoint_path).resolve(),
      **ppo_params,
  )

  if isinstance(params, dict):
    normalizer = params.get("normalizer") or params.get("normalizer_params")
    policy = params.get("policy") or params.get("policy_params") or params.get("params")
    if normalizer is not None and policy is not None:
      params = (normalizer, policy)

  inference_fn = make_inference_fn(params, deterministic=True)
  return env, jax.jit(inference_fn)


def _evaluate_batch(env, inference_fn, batch_size: int, seed: int):
  active_th = float(env._config.reward_config.finger_active_threshold)
  dt = float(env.dt)
  episode_length = int(env._config.episode_length)
  spawn_z = float(env._spawn_z)

  v_reset = jax.jit(jax.vmap(env.reset))
  v_step = jax.jit(jax.vmap(env.step))
  v_tip_force = jax.jit(jax.vmap(lambda d: env._get_contact_forces_efc(d)))
  v_tip_flags = jax.jit(jax.vmap(lambda d: env._get_tip_contact_flags(d)))
  v_cube_z = jax.jit(jax.vmap(lambda d: env.get_cube_position(d)[2]))

  rng = jax.random.PRNGKey(seed)
  rng, reset_rng = jax.random.split(rng)
  states = v_reset(jax.random.split(reset_rng, batch_size))

  contact_steps = jp.zeros((batch_size,), dtype=jp.int32)
  post_release_contact_steps = jp.zeros((batch_size,), dtype=jp.int32)
  max_lift = jp.full((batch_size,), -1e6, dtype=jp.float32)
  max_primary_force = jp.zeros((batch_size,), dtype=jp.float32)
  release_step = jp.full((batch_size,), -1, dtype=jp.int32)
  drop_step = jp.full((batch_size,), -1, dtype=jp.int32)
  max_hold_run = jp.zeros((batch_size,), dtype=jp.int32)
  hold_run = jp.zeros((batch_size,), dtype=jp.int32)
  nonfinite_cube = jp.zeros((batch_size,), dtype=bool)

  for step_idx in range(episode_length):
    rng, act_key = jax.random.split(rng)
    act_keys = jax.random.split(act_key, batch_size)
    action = jax.vmap(lambda obs, key: inference_fn(obs, key)[0])(states.obs, act_keys)
    states = v_step(states, action)

    tip_force = v_tip_force(states.data)
    tip_flags = v_tip_flags(states.data)
    abs_f = jp.abs(tip_force)
    wrap_forces = jp.stack([abs_f[:, 0], abs_f[:, 1], abs_f[:, 2], abs_f[:, 4]], axis=1)
    wrap_geom = jp.stack([tip_flags[:, 0], tip_flags[:, 1], tip_flags[:, 2], tip_flags[:, 4]], axis=1)
    wrap_active = jp.maximum((wrap_forces > active_th).astype(jp.float32), wrap_geom)
    wrap_count = jp.sum(wrap_active, axis=1)
    thumb_active = wrap_active[:, 3] > 0.5
    three_contact = (wrap_count >= 3.0) & thumb_active

    cube_z = v_cube_z(states.data)
    cube_finite = jp.isfinite(cube_z)
    lift = cube_z - spawn_z
    released = states.info["support_released"].astype(bool)
    done = states.done.astype(bool)

    post_release_hold = three_contact & released & (~done)

    contact_steps = contact_steps + three_contact.astype(jp.int32)
    post_release_contact_steps = post_release_contact_steps + post_release_hold.astype(jp.int32)
    safe_lift = jp.where(cube_finite, lift, -1e6)
    max_lift = jp.maximum(max_lift, safe_lift)
    max_primary_force = jp.maximum(max_primary_force, jp.mean(wrap_forces, axis=1))
    release_step = jp.where((release_step < 0) & released, step_idx + 1, release_step)
    drop_step = jp.where((drop_step < 0) & released & done, step_idx + 1, drop_step)
    hold_run = jp.where(post_release_hold, hold_run + 1, 0)
    max_hold_run = jp.maximum(max_hold_run, hold_run)
    nonfinite_cube = nonfinite_cube | (~cube_finite)

  return {
      "contact_sec": np.asarray(contact_steps, dtype=np.float32) * dt,
      "post_release_contact_sec": np.asarray(post_release_contact_steps, dtype=np.float32) * dt,
      "max_lift_m": np.asarray(max_lift, dtype=np.float32),
      "max_primary_force": np.asarray(max_primary_force, dtype=np.float32),
      "release_step": np.asarray(release_step, dtype=np.int32),
      "drop_step": np.asarray(drop_step, dtype=np.int32),
      "max_post_release_hold_sec": np.asarray(max_hold_run, dtype=np.float32) * dt,
      "nonfinite_cube": np.asarray(nonfinite_cube),
  }


def _print_summary(metrics):
  contact_sec = metrics["contact_sec"]
  post_contact_sec = metrics["post_release_contact_sec"]
  max_lift = metrics["max_lift_m"]
  max_force = metrics["max_primary_force"]
  release_step = metrics["release_step"]
  drop_step = metrics["drop_step"]
  max_hold_sec = metrics["max_post_release_hold_sec"]
  nonfinite_cube = metrics["nonfinite_cube"]

  released = release_step >= 0
  dropped_after_release = drop_step >= 0

  print("Offline Eval Summary")
  print(f"episodes: {len(contact_sec)}")
  print(
      "contact_sec mean/median/max: "
      f"{contact_sec.mean():.3f} / {np.median(contact_sec):.3f} / {contact_sec.max():.3f}"
  )
  print(
      "post_release_contact_sec mean/median/max: "
      f"{post_contact_sec.mean():.3f} / {np.median(post_contact_sec):.3f} / {post_contact_sec.max():.3f}"
  )
  print(
      "max_lift_m mean/median/max: "
      f"{max_lift.mean():.4f} / {np.median(max_lift):.4f} / {max_lift.max():.4f}"
  )
  print(
      "max_primary_force mean/median/max: "
      f"{max_force.mean():.3f} / {np.median(max_force):.3f} / {max_force.max():.3f}"
  )
  print(f"released episodes: {released.mean() * 100:.1f}%")
  print(f"dropped after release: {dropped_after_release.mean() * 100:.1f}%")
  print(f"nonfinite cube state: {nonfinite_cube.mean() * 100:.1f}%")
  print(f"lift > 1cm: {(max_lift > 0.010).mean() * 100:.1f}%")
  print(f"lift > 2cm: {(max_lift > 0.020).mean() * 100:.1f}%")
  print(f"post-release hold >= 0.5s: {(max_hold_sec >= 0.5).mean() * 100:.1f}%")
  print(f"post-release hold >= 1.0s: {(max_hold_sec >= 1.0).mean() * 100:.1f}%")
  print(f"post-release hold >= 2.0s: {(max_hold_sec >= 2.0).mean() * 100:.1f}%")

  topk = np.argsort(-max_lift)[:5]
  print("top_lift_episodes:")
  for idx in topk:
    print(
        f"  ep={idx:02d} lift={max_lift[idx]:.4f}m "
        f"contact={contact_sec[idx]:.3f}s post={post_contact_sec[idx]:.3f}s "
        f"hold={max_hold_sec[idx]:.3f}s release_step={release_step[idx]} drop_step={drop_step[idx]}"
    )


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", default="AeroBottleGraspV2Force")
  parser.add_argument("--checkpoint_path", required=True)
  parser.add_argument("--episodes", type=int, default=64)
  parser.add_argument("--seed", type=int, default=0)
  args = parser.parse_args()

  ckpt_path = Path(args.checkpoint_path).resolve()
  env, inference_fn = _load_policy(args.env_name, ckpt_path)
  metrics = _evaluate_batch(env, inference_fn, args.episodes, args.seed)
  _print_summary(metrics)


if __name__ == "__main__":
  main()
