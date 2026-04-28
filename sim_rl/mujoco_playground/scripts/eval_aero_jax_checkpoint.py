#!/usr/bin/env python3
"""Evaluate an Aero-Hand JAX PPO checkpoint and summarize diagnostics/reward parts."""

from __future__ import annotations

import argparse
import functools
import inspect
import json
import os
from pathlib import Path

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

from etils import epath
import jax
import jax.numpy as jp
import numpy as np
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.agents.ppo.train import train as ppo_train

from mujoco_playground import registry
from mujoco_playground import wrapper
from mujoco_playground.config import manipulation_params


def _merge_network_factory_from_checkpoint(ppo_params, checkpoint_path: epath.Path) -> None:
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


def _load_policy(env_name: str, checkpoint_path: str):
  env = registry.load(env_name)
  ppo_params = manipulation_params.brax_ppo_config(env_name)
  _merge_network_factory_from_checkpoint(ppo_params, epath.Path(checkpoint_path).resolve())
  ppo_params.num_envs = 1
  ppo_params.num_eval_envs = 1
  ppo_params.batch_size = 1
  ppo_params.num_minibatches = 1
  ppo_params.unroll_length = 1
  ppo_params.num_updates_per_batch = 1
  network_factory_config = ppo_params.get("network_factory", {})
  del ppo_params["network_factory"]

  supported_network_kwargs = set(inspect.signature(ppo_networks.make_ppo_networks).parameters)
  network_factory_config = {
      key: value for key, value in network_factory_config.items()
      if key in supported_network_kwargs
  }

  network_factory = functools.partial(
      ppo_networks.make_ppo_networks, **network_factory_config
  )
  if "num_timesteps" in ppo_params:
    del ppo_params["num_timesteps"]

  make_inference_fn, params, _ = ppo_train(
      environment=env,
      wrap_env_fn=wrapper.wrap_for_brax_training,
      network_factory=network_factory,
      num_timesteps=0,
      seed=1,
      restore_checkpoint_path=epath.Path(checkpoint_path).resolve(),
      **ppo_params,
  )
  inference_fn = make_inference_fn(params, deterministic=True)
  return env, jax.jit(inference_fn)


def _episode_rollout(env, inference_fn, rng):
  state = env.reset(rng)
  episode_length = int(getattr(env._config, "episode_length", 1000))

  reward_keys = tuple(k for k in state.metrics if k.startswith("reward/"))
  diag_keys = tuple(k for k in state.metrics if k.startswith("diagnostic/"))

  def _scan_step(carry, _):
    state, rng = carry
    rng, act_rng = jax.random.split(rng)
    action, _ = inference_fn(state.obs, act_rng)
    next_state = env.step(state, action)
    reward_vals = jp.stack([next_state.metrics[k] for k in reward_keys])
    diag_vals = jp.stack([next_state.metrics[k] for k in diag_keys])
    return (next_state, rng), (reward_vals, diag_vals, next_state.done)

  (state, rng), (reward_hist, diag_hist, done_hist) = jax.lax.scan(
      _scan_step, (state, rng), xs=None, length=episode_length
  )
  reward_hist = np.asarray(reward_hist)
  diag_hist = np.asarray(diag_hist)
  done_hist = np.asarray(done_hist).reshape(-1)
  done_idx = np.where(done_hist > 0.5)[0]
  steps = int(done_idx[0] + 1) if done_idx.size else episode_length
  reward_hist = reward_hist[:steps]
  diag_hist = diag_hist[:steps]

  reward_sums = {
      key: float(reward_hist[:, i].sum())
      for i, key in enumerate(reward_keys)
  }
  diag_last = {
      key: float(diag_hist[-1, i])
      for i, key in enumerate(diag_keys)
  }
  diag_step_mean = {
      key: float(diag_hist[:, i].mean())
      for i, key in enumerate(diag_keys)
  }
  diag_step_max = {
      key: float(diag_hist[:, i].max())
      for i, key in enumerate(diag_keys)
  }
  diag_sum = {
      key: float(diag_hist[:, i].sum())
      for i, key in enumerate(diag_keys)
  }
  return rng, {
      "steps": steps,
      "episode_reward": float(state.reward),
      "diagnostics_last": diag_last,
      "diagnostics_step_mean": diag_step_mean,
      "diagnostics_step_max": diag_step_max,
      "diagnostics_sum": diag_sum,
      "reward_sums": reward_sums,
  }


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--env_name", required=True)
  parser.add_argument("--checkpoint_path", required=True)
  parser.add_argument("--episodes", type=int, default=16)
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--json_out", type=Path)
  args = parser.parse_args()

  env, inference_fn = _load_policy(args.env_name, args.checkpoint_path)
  rng = jax.random.PRNGKey(args.seed)

  episodes = []
  for _ in range(args.episodes):
    rng, episode = _episode_rollout(env, inference_fn, rng)
    episodes.append(episode)

  diag_keys = sorted({k for ep in episodes for k in ep["diagnostics_sum"]})
  reward_keys = sorted({k for ep in episodes for k in ep["reward_sums"]})

  summary = {
      "env_name": args.env_name,
      "checkpoint_path": args.checkpoint_path,
      "episodes": args.episodes,
      "diagnostic_last_mean": {
          k: float(np.mean([ep["diagnostics_last"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "diagnostic_last_max": {
          k: float(np.max([ep["diagnostics_last"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "diagnostic_step_mean": {
          k: float(np.mean([ep["diagnostics_step_mean"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "diagnostic_step_max": {
          k: float(np.max([ep["diagnostics_step_max"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "diagnostic_sum_mean": {
          k: float(np.mean([ep["diagnostics_sum"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "diagnostic_sum_max": {
          k: float(np.max([ep["diagnostics_sum"].get(k, 0.0) for ep in episodes]))
          for k in diag_keys
      },
      "reward_component_mean": {
          k: float(np.mean([ep["reward_sums"].get(k, 0.0) for ep in episodes]))
          for k in reward_keys
      },
      "reward_component_max": {
          k: float(np.max([ep["reward_sums"].get(k, 0.0) for ep in episodes]))
          for k in reward_keys
      },
  }

  print(json.dumps(summary, indent=2, sort_keys=True))
  if args.json_out:
    args.json_out.write_text(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
