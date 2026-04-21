"""Render rollout videos from a checkpoint.

Handles brax API version differences by remapping checkpoint params to current
network structure if needed.

Usage:
  python learning/render_checkpoint.py \
    --checkpoint_path=logs/.../checkpoints/000011796480 \
    --output_dir=/path/to/output \
    --cameras=side,palm \
    --render_collision_debug=True \
    --num_videos=1 \
    --episode_length=800 \
    --env_name=AeroCubeGraspV2ForceCoacd
"""

import functools
import json
import pathlib

import jax
import jax.numpy as jp
import mediapy as media
import mujoco
import numpy as np

from absl import app, flags
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics as acme_rs

_CHECKPOINT_PATH = flags.DEFINE_string("checkpoint_path", None, "Path to checkpoint dir.", required=True)
_OUTPUT_DIR = flags.DEFINE_string("output_dir", None, "Output directory for videos.", required=True)
_ENV_NAME = flags.DEFINE_string("env_name", "AeroCubeGraspV2ForceCoacd", "Environment name.")
_CAMERAS = flags.DEFINE_list("cameras", ["side"], "Cameras to render.")
_RENDER_COLLISION_DEBUG = flags.DEFINE_boolean("render_collision_debug", False, "Also render collision debug video.")
_NUM_VIDEOS = flags.DEFINE_integer("num_videos", 1, "Number of rollouts to render.")
_EPISODE_LENGTH = flags.DEFINE_integer("episode_length", 800, "Episode length in steps.")
_VIDEO_PREFIX = flags.DEFINE_string("video_prefix", "rollout", "Prefix for output video filenames.")
_SEED = flags.DEFINE_integer("seed", 0, "Random seed.")
_HEIGHT = flags.DEFINE_integer("height", 480, "Video height.")
_WIDTH = flags.DEFINE_integer("width", 640, "Video width.")


def _detect_config(ckpt_path: pathlib.Path, action_size: int):
    """Detect network config from checkpoint param shapes."""
    with open(ckpt_path / "array_metadatas" / "process_0") as f:
        meta = json.load(f)

    policy_kernels = [
        m["array_metadata"] for m in meta["array_metadatas"]
        if m["array_metadata"]["param_name"].startswith("1.params.")
        and m["array_metadata"]["param_name"].endswith(".kernel")
    ]
    names = [m["param_name"] for m in policy_kernels]
    is_old_format = not any("MLP_0" in n for n in names)

    output_shape = policy_kernels[-1]["write_shape"]
    output_size = output_shape[-1]
    if output_size == action_size * 2:
        distribution_type = "normal"
        noise_std_type = "log"
    else:
        distribution_type = "tanh_normal"
        noise_std_type = "scalar"

    hidden_sizes = tuple(k["write_shape"][1] for k in policy_kernels[:-1])

    return {
        "is_old_format": is_old_format,
        "distribution_type": distribution_type,
        "noise_std_type": noise_std_type,
        "hidden_sizes": hidden_sizes,
        "output_size": output_size,
    }


def _remap_policy_old_to_new(old_policy_params: dict, action_size: int) -> dict:
    """Remap old flat policy params to new nested brax structure.

    Old: {'params': {'hidden_0': {kernel, bias}, ..., 'hidden_N': {kernel:(H,2A), bias:(2A,)}}}
    New: {'params': {'MLP_0': {'hidden_0': ..., ...}, 'Dense_0': {kernel:(H,A)}, 'std_logparam': {log_value:(A,)}}}
    """
    p = old_policy_params["params"]
    layer_keys = sorted(p.keys())  # e.g. ['hidden_0', 'hidden_1', 'hidden_2']
    num_hidden = len(layer_keys) - 1  # layers before output

    new_policy = {"MLP_0": {}, "Dense_0": {}, "std_logparam": {}}
    for i in range(num_hidden):
        new_policy["MLP_0"][f"hidden_{i}"] = {
            "kernel": np.array(p[f"hidden_{i}"]["kernel"]),
            "bias": np.array(p[f"hidden_{i}"]["bias"]),
        }

    out_kernel = np.array(p[f"hidden_{num_hidden}"]["kernel"])  # (H, 2*A)
    out_bias = np.array(p[f"hidden_{num_hidden}"]["bias"])      # (2*A,)
    new_policy["Dense_0"] = {
        "kernel": out_kernel[:, :action_size],
        "bias": out_bias[:action_size],
    }
    new_policy["std_logparam"] = {
        "log_value": out_bias[action_size:],
    }

    return {"params": new_policy}


def _remap_value_old_to_new(old_value_params: dict) -> dict:
    """Remap old flat value params to new nested brax structure.

    Old: {'params': {'hidden_0': ..., ..., 'hidden_N': {kernel:(H,1), bias:(1,)}}}
    New: {'params': {'MLP_0': {'hidden_0': ..., ...}, 'Dense_0': {kernel:(H,1)}}}
    """
    p = old_value_params["params"]
    layer_keys = sorted(p.keys())
    num_hidden = len(layer_keys) - 1

    new_value = {"MLP_0": {}, "Dense_0": {}}
    for i in range(num_hidden):
        new_value["MLP_0"][f"hidden_{i}"] = {
            "kernel": np.array(p[f"hidden_{i}"]["kernel"]),
            "bias": np.array(p[f"hidden_{i}"]["bias"]),
        }
    new_value["Dense_0"] = {
        "kernel": np.array(p[f"hidden_{num_hidden}"]["kernel"]),
        "bias": np.array(p[f"hidden_{num_hidden}"]["bias"]),
    }

    return {"params": new_value}


def main(argv):
    del argv

    ckpt_path = pathlib.Path(_CHECKPOINT_PATH.value).resolve()
    output_dir = pathlib.Path(_OUTPUT_DIR.value)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load network config from checkpoint
    with open(ckpt_path / "ppo_network_config.json") as f:
        net_cfg = json.load(f)

    action_size = net_cfg["action_size"]
    obs_size = net_cfg["observation_size"]["state"]["shape"][0]
    val_obs_size = net_cfg["observation_size"]["privileged_state"]["shape"][0]
    normalize_obs = net_cfg.get("normalize_observations", True)

    detected = _detect_config(ckpt_path, action_size)
    print(f"Checkpoint detection: {detected}")

    # Build env
    from mujoco_playground import registry
    env = registry.load(_ENV_NAME.value)

    hidden_sizes = detected["hidden_sizes"]
    distribution_type = detected["distribution_type"]
    noise_std_type = detected["noise_std_type"]

    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=hidden_sizes,
        value_hidden_layer_sizes=hidden_sizes,
        distribution_type=distribution_type,
        noise_std_type=noise_std_type,
        policy_obs_key="state",
        value_obs_key="privileged_state",
    )
    ppo_network = network_factory(
        observation_size={"state": (obs_size,), "privileged_state": (val_obs_size,)},
        action_size=action_size,
    )

    # Load checkpoint using brax's loader (handles orbax internally, requires absolute path)
    from brax.training import checkpoint as brax_ckpt
    print(f"Loading checkpoint from: {ckpt_path}")
    brax_params = brax_ckpt.load(str(ckpt_path))
    # brax_params is a list/tuple: [normalizer_state, policy_params, value_params]
    norm_state = brax_params[0]
    raw_policy = brax_params[1]
    raw_value = brax_params[2]

    if detected["is_old_format"]:
        print("Old-format checkpoint detected. Remapping params to current brax structure...")
        policy_params = _remap_policy_old_to_new(raw_policy, action_size)
        value_params = _remap_value_old_to_new(raw_value)
    else:
        policy_params = raw_policy
        value_params = raw_value

    # Build a proper RunningStatisticsState for the network's normalizer.
    # orbax restores without target so brax_params[0] is a plain dict.
    norm_raw = brax_params[0]
    if isinstance(norm_raw, acme_rs.RunningStatisticsState):
        running_norm = norm_raw
    else:
        # Plain dict restored by orbax
        running_norm = acme_rs.RunningStatisticsState(
            count=norm_raw["count"],
            mean=jax.tree_util.tree_map(jp.array, norm_raw["mean"]),
            summed_variance=jax.tree_util.tree_map(jp.array, norm_raw["summed_variance"]),
            std=jax.tree_util.tree_map(jp.array, norm_raw["std"]),
            std_eps=float(norm_raw.get("std_eps", 0.0)),
        )

    if not normalize_obs:
        running_norm = acme_rs.init_state(
            {"state": jp.zeros(obs_size), "privileged_state": jp.zeros(val_obs_size)}
        )

    # Build inference function — network.apply(processor_params, policy_params, obs)
    def make_policy_fn(policy_p, norm_rs, deterministic=True):
        def policy_fn(obs, key):
            dist_params, _ = ppo_network.policy_network.apply(norm_rs, policy_p, obs)
            if deterministic:
                action = dist_params.loc if hasattr(dist_params, "loc") else dist_params[..., :action_size]
            else:
                action = dist_params.sample(seed=key)
            return action, {}
        return policy_fn

    inference_fn = make_policy_fn(policy_params, running_norm, deterministic=True)
    jit_inference_fn = jax.jit(inference_fn)

    # Run rollouts
    def do_rollout(rng, state):
        def step(carry, _):
            state, rng = carry
            rng, act_key = jax.random.split(rng)
            act = jit_inference_fn(state.obs, act_key)[0]
            state = env.step(state, act)
            return (state, rng), state

        _, traj = jax.lax.scan(step, (state, rng), None, length=_EPISODE_LENGTH.value)
        return traj

    rng = jax.random.split(jax.random.PRNGKey(_SEED.value), _NUM_VIDEOS.value)
    reset_states = jax.jit(jax.vmap(env.reset))(rng)
    traj_stacked = jax.jit(jax.vmap(do_rollout))(rng, reset_states)

    trajectories = []
    for i in range(_NUM_VIDEOS.value):
        t = jax.tree.map(lambda x, i=i: x[i], traj_stacked)
        traj_steps = [
            jax.tree.map(lambda x, j=j: x[j], t)
            for j in range(_EPISODE_LENGTH.value)
        ]

        # ── NaN/Inf 诊断: 检测 qpos/qvel 第一次出现异常的步骤 ──
        first_nan_step = None
        for j, s in enumerate(traj_steps):
            qpos_np = np.array(s.data.qpos)
            qvel_np = np.array(s.data.qvel)
            if not np.all(np.isfinite(qpos_np)) or not np.all(np.isfinite(qvel_np)):
                first_nan_step = j
                t_sec = j * env.dt
                print(f"[NaN/Inf 诊断] rollout {i}: step {j} ({t_sec:.2f}s) 首次出现非有限值")
                print(f"  qpos nan_mask: {~np.isfinite(qpos_np)}")
                print(f"  qvel nan_mask: {~np.isfinite(qvel_np)}")
                print(f"  qpos[:7] (hand joints): {qpos_np[:7]}")
                print(f"  qpos[13:] (cube freejoint): {qpos_np[13:]}")
                break
        if first_nan_step is None:
            print(f"[NaN/Inf 诊断] rollout {i}: 全程 {_EPISODE_LENGTH.value} 步无异常 qpos/qvel")
        else:
            # 打印异常前一步的 done 状态
            if first_nan_step > 0:
                prev = traj_steps[first_nan_step - 1]
                print(f"  前一步 (step {first_nan_step - 1}) done={float(prev.done):.0f}, "
                      f"reward={float(prev.reward):.3f}")

        trajectories.append(traj_steps)

    # Render
    render_every = 2
    fps = 1.0 / env.dt / render_every
    print(f"FPS: {fps}")

    scene_option = mujoco.MjvOption()
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTFORCE] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_PERTOBJ] = True
    scene_option.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

    debug_scene_option = mujoco.MjvOption()
    debug_scene_option.geomgroup[4] = True

    for i, rollout in enumerate(trajectories):
        traj = rollout[::render_every]
        for cam in _CAMERAS.value:
            frames = env.render(
                traj, height=_HEIGHT.value, width=_WIDTH.value,
                camera=cam, scene_option=scene_option,
            )
            video_path = str(output_dir / f"{_VIDEO_PREFIX.value}{i}_{cam}.mp4")
            media.write_video(video_path, frames, fps=fps)
            print(f"Saved: {video_path}")

        if _RENDER_COLLISION_DEBUG.value:
            frames = env.render(
                traj, height=_HEIGHT.value, width=_WIDTH.value,
                camera=_CAMERAS.value[0], scene_option=debug_scene_option,
            )
            debug_path = str(output_dir / f"{_VIDEO_PREFIX.value}{i}_collision_debug.mp4")
            media.write_video(debug_path, frames, fps=fps)
            print(f"Saved: {debug_path}")

    print(f"\nAll videos saved to: {output_dir}")


if __name__ == "__main__":
    app.run(main)
