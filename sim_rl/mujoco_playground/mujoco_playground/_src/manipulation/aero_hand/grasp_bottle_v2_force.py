# Copyright 2025 TetherIA Inc.
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""V2 灵犀手 550ml 空瓶抓握任务."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import grasp_cube_v2_force


def default_config() -> config_dict.ConfigDict:
  cfg = grasp_cube_v2_force.default_config()

  cfg.action_scale = [0.08, 0.18, 0.22, 0.22, 0.14, 0.14]

  cfg.support_config.release_after_sec = 4.2
  cfg.support_config.release_ramp_sec = 0.5
  cfg.support_config.force_release_after_sec = 6.0
  cfg.support_config.support_pos = [0.028, -0.075, 0.1325]
  cfg.support_config.random_release = True
  cfg.support_config.random_release_min_sec = 4.0
  cfg.support_config.random_release_max_sec = 5.0
  cfg.support_config.min_release_active_fingers = 3
  cfg.support_config.min_release_force = 0.20

  cfg.spawn_config.cube_pos = [0.028, -0.075, 0.169]
  cfg.spawn_config.cube_jitter = [0.0015, 0.0015, 0.002]
  cfg.spawn_config.support_enabled = False

  cfg.reset_config.pre_grasp_fraction = 0.0
  cfg.reset_config.pre_grasp_noise_scale = 0.05
  cfg.reset_config.lifted_grasp_fraction = 1.0
  cfg.reset_config.lifted_grasp_noise_scale = 0.02
  cfg.reset_config.lifted_cube_z_offset = 0.020

  cfg.perturbation_config.external_force_magnitude = 0.05
  cfg.perturbation_config.gravity_tilt_max_rad = 0.25
  cfg.perturbation_config.orientation_flip_force_scale = 0.85
  cfg.perturbation_config.orientation_flip_min_hold_steps = 320

  scales = cfg.reward_config.scales
  scales.hold_position = 50.0
  scales.stable_hold = 210.0
  scales.progressive_hold = 75.0
  scales.sustained_hold_bonus = 95.0
  scales.force_contact = 20.0
  scales.grip_force = 12.0
  scales.force_balance = 22.0
  scales.finger_participation = 24.0
  scales.thumb_opposition = 34.0
  scales.soft_contact = 8.0
  scales.primary_finger_force = 62.0
  scales.pre_release_grasp = 56.0
  scales.post_release_grasp = 135.0
  scales.post_release_survival = 180.0
  scales.post_release_cheat_contact = 0.0
  scales.post_release_slip = -65.0
  scales.post_release_pose_hold = 85.0
  scales.height = 14.0
  scales.termination = -420.0
  scales.drop_risk = -55.0
  scales.palm_contact = 1.5
  scales.nonprimary_contact = 1.0
  scales.three_finger_proximity = 18.0
  scales.force_overload = -4.0

  cfg.reward_config.force_contact_threshold = 0.04
  cfg.reward_config.force_contact_saturation = 2.4
  cfg.reward_config.force_overload_threshold = 2.4
  cfg.reward_config.force_overload_soft_width = 1.0
  cfg.reward_config.soft_contact_fmin = 0.08
  cfg.reward_config.soft_contact_fmax = 2.20
  cfg.reward_config.finger_active_threshold = 0.06
  cfg.reward_config.target_lift_m = 0.015
  cfg.reward_config.lift_success_threshold_m = 0.010
  cfg.stability_config.max_abs_action = 0.65
  cfg.stability_config.motor_delta_clip = [0.03, 0.05, 0.06, 0.06, 0.05, 0.05]
  cfg.stability_config.terminate_on_nonfinite = True
  cfg.stability_config.nonfinite_penalty_mult = 3.0

  return cfg


class BottleGraspV2Force(grasp_cube_v2_force.CubeGraspV2Force):
  """Bottle grasp task using bottle-specific scene and softer reward shaping."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=xml_path or consts.GRASP_V2_BOTTLE_XML.as_posix(),
    )
    # Bottle grasp needs a fuller cylindrical wrap than the cube pinch prior.
    self._pre_grasp_pose = np.array([
        0.75,
        0.925 * 0.75,
        0.75,
        0.925 * 0.75,
        0.75,
        0.925 * 0.75,
        0.65,
        0.925 * 0.65,
        1.25,
        0.16 * 1.25,
        0.20,
    ], dtype=np.float32)
    self._lifted_grasp_pose = self._pre_grasp_pose.copy()
    self._lifted_grasp_ctrl = np.array([
        1.25, 0.20, 0.75, 0.75, 0.75, 0.65,
    ], dtype=np.float32)

  def _reward_contact(self, tip_world, cube_pos):
    del cube_pos
    target = self._spawn_pos
    body_half_extents = np.array([0.040, 0.045, 0.105], dtype=np.float32)
    diff = jp.abs(tip_world - target[None, :]) - body_half_extents
    surface_dists = jp.linalg.norm(jp.maximum(diff, 0.0), axis=1)
    return jp.mean(jp.exp(-4.0 * surface_dists))

  def _reward_hold_position(
      self, cube_pos: jax.Array, cube_linvel: jax.Array,
  ) -> jax.Array:
    xy_err = jp.linalg.norm((cube_pos - self._spawn_pos)[:2])
    xy_reward = jp.exp(-18.0 * xy_err)
    z_floor = self._spawn_z - 0.020
    z_safe = jp.clip((cube_pos[2] - z_floor) / 0.020, 0.0, 1.0)
    vel_penalty = jp.clip(jp.linalg.norm(cube_linvel) / 0.24, 0.0, 1.0)
    return xy_reward * z_safe * (1.0 - 0.35 * vel_penalty)

  def _reward_post_release_survival(
      self, cube_pos: jax.Array, cube_linvel: jax.Array,
  ) -> jax.Array:
    z_floor = self._spawn_z - 0.020
    z_safe = jp.clip((cube_pos[2] - z_floor) / 0.020, 0.0, 1.0)
    xy_err = jp.linalg.norm((cube_pos - self._spawn_pos)[:2])
    xy_safe = jp.exp(-12.0 * xy_err)
    slow = jp.exp(-2.8 * jp.linalg.norm(cube_linvel))
    return z_safe * xy_safe * slow

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    cube_z = self.get_cube_position(data)[2]
    drop_z = self._spawn_z - 0.06
    return cube_z < drop_z

  def _reward_thumb_opposition(
      self, tip_world: jax.Array, cube_pos: jax.Array,
      tip_force: jax.Array,
  ) -> jax.Array:
    """Bottle wrap rewards thumb opposition to a broader finger set."""
    tip_to_cube = cube_pos[None, :] - tip_world
    tip_to_cube_norm = tip_to_cube / (jp.linalg.norm(tip_to_cube, axis=1, keepdims=True) + 1e-6)
    thumb_dir = tip_to_cube_norm[4]
    finger_dirs = tip_to_cube_norm[:4]
    dots = jp.sum(finger_dirs * thumb_dir[None, :], axis=1)
    opposition = jax.nn.sigmoid(-5.0 * dots)

    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    finger_soft_active = jax.nn.sigmoid(30.0 * (abs_f[:4] - th * 0.5))
    thumb_soft_active = jax.nn.sigmoid(30.0 * (abs_f[4] - th * 0.5))
    gated_opposition = opposition * (0.45 * finger_soft_active * thumb_soft_active + 0.55)

    # Empty bottle prefers thumb opposing index/middle first, but ring support
    # should still earn substantial signal.
    finger_weights = jp.array([1.0, 1.0, 0.75, 0.25], dtype=jp.float32)
    weighted = gated_opposition * finger_weights
    return jp.max(weighted)

  def _reward_primary_finger_force(self, tip_force: jax.Array) -> jax.Array:
    """Bottle grasp encourages thumb+index+middle+ring wrap, not only pinch."""
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    sat = self._config.reward_config.force_contact_saturation
    wrap_forces = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]])
    normalized = jp.clip((wrap_forces - th) / (sat - th + 1e-6), 0.0, 1.0)
    weighted = normalized * jp.array([1.0, 1.0, 0.8, 1.0], dtype=jp.float32)
    mean_val = jp.sum(weighted) / 3.8
    min_val = jp.min(normalized[:3])
    thumb_val = normalized[3]
    support_bonus = 0.25 + 0.40 * min_val + 0.35 * thumb_val
    return mean_val * support_bonus

  def _reward_release_ready(self, tip_force: jax.Array) -> jax.Array:
    """Bottle-specific release readiness prefers full wrap before support drops."""
    abs_f = jp.abs(tip_force)
    wrap = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]])
    th = self._config.reward_config.finger_active_threshold
    sat = self._config.reward_config.force_contact_saturation
    soft_active = jax.nn.sigmoid(35.0 * (wrap - th * 0.5))
    active_mean = jp.mean(soft_active)
    thumb_ready = soft_active[3]
    triad_ready = jp.min(soft_active[:3])
    normalized = jp.clip((wrap - th * 0.25) / (sat - th * 0.25 + 1e-6), 0.0, 1.0)
    weighted_force = jp.dot(
        normalized, jp.array([1.0, 1.0, 0.8, 1.0], dtype=jp.float32)
    ) / 3.8
    rel_std = jp.std(wrap) / (jp.mean(wrap) + 1e-6)
    balance = jp.clip(1.0 - rel_std, 0.0, 1.0)
    return active_mean * (0.25 + 0.45 * triad_ready + 0.30 * thumb_ready) * (
        0.45 + 0.55 * balance
    ) * (0.35 + 0.65 * weighted_force)

  def _get_wrap_contact_state(
      self,
      tip_force: jax.Array,
      tip_contact_flags: jax.Array,
  ) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Bottle wrap state over index, middle, ring, thumb."""
    abs_f = jp.abs(tip_force)
    wrap_forces = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]])
    wrap_geom = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[2],
        tip_contact_flags[4],
    ])
    active_th = self._config.reward_config.finger_active_threshold
    wrap_active = jp.maximum((wrap_forces > active_th).astype(jp.float32), wrap_geom)
    wrap_count = jp.sum(wrap_active)
    return wrap_forces, wrap_active, wrap_count

  def _is_grasp_ready_for_release(
      self,
      tip_finger_forces: jax.Array,
      tip_contact_flags: jax.Array | None = None,
  ) -> jax.Array:
    """Require a bottle-style wrap before releasing support."""
    abs_f = jp.abs(tip_finger_forces)
    wrap = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]])
    if tip_contact_flags is None:
      wrap_geom = jp.zeros(4, dtype=jp.float32)
    else:
      wrap_geom = jp.array([
          tip_contact_flags[0],
          tip_contact_flags[1],
          tip_contact_flags[2],
          tip_contact_flags[4],
      ])
    active_th = self._config.reward_config.finger_active_threshold
    active = jp.maximum((wrap > active_th).astype(jp.float32), wrap_geom)
    active_count = jp.sum(active)
    thumb_active = active[3] > 0.5
    force_ok = jp.sum(wrap * jp.array([1.0, 1.0, 0.8, 1.0], dtype=jp.float32)) >= (
        self._config.support_config.min_release_force * 3.2
    )
    return (active_count >= 3.0) & thumb_active & force_ok

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    lifted_reset = state.info.get("lifted_reset", jp.array(False))

    def _hide_support(_: None) -> mjx_env.State:
      data = state.data.replace(
          mocap_pos=state.data.mocap_pos.at[self._support_mocap_id].set(
              self._support_hidden_pos
          )
      )
      info = state.info.copy()
      info["use_support"] = jp.array(False)
      info["support_released"] = jp.array(True)
      info["support_release_steps"] = jp.array(0, dtype=jp.int32)
      obs = self._get_obs(data, info, state.obs["state"])
      return state.replace(data=data, info=info, obs=obs)

    return jax.lax.cond(lifted_reset, _hide_support, lambda _: state, operand=None)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action = self._stabilize_action(action)
    action_scale = jp.array(self._config.action_scale, dtype=jp.float32)
    motor_targets = jp.clip(
        self._default_ctrl + action * action_scale,
        self._ctrl_lowers,
        self._ctrl_uppers,
    )
    motor_targets = self._stabilize_motor_targets(
        motor_targets, state.info["motor_targets"],
    )

    data = state.data
    pcfg = self._config.perturbation_config
    hold_steps = state.info["stable_hold_steps"]
    support_released = state.info["support_released"]
    step_rng = state.info["rng"]
    step_rng, force_rng, tilt_rng, flip_rng = jax.random.split(step_rng, 4)

    ext_force = state.info["perturbation_force"]
    if pcfg.external_force_enabled:
      should_apply_ext = (
          support_released
          & (hold_steps >= pcfg.external_force_min_hold_steps)
          & (hold_steps % pcfg.external_force_interval == 0)
      )
      rand_dir = jax.random.normal(force_rng, (3,))
      rand_dir = rand_dir / (jp.linalg.norm(rand_dir) + 1e-8)
      new_force = rand_dir * pcfg.external_force_magnitude
      ext_force = jp.where(should_apply_ext, new_force, jp.zeros(3))

    tilt_angle = state.info["gravity_tilt_angle"]
    if pcfg.gravity_perturbation_enabled:
      should_update_tilt = (
          support_released
          & (hold_steps >= pcfg.gravity_tilt_min_hold_steps)
          & (hold_steps % pcfg.gravity_tilt_change_interval == 0)
      )
      new_tilt = jax.random.uniform(
          tilt_rng, (2,),
          minval=-pcfg.gravity_tilt_max_rad,
          maxval=pcfg.gravity_tilt_max_rad,
      )
      tilt_angle = jp.where(should_update_tilt, new_tilt, tilt_angle)

    gravity_force = jp.array([
        self._cube_mass * 9.81 * jp.sin(tilt_angle[0]),
        self._cube_mass * 9.81 * jp.sin(tilt_angle[1]),
        0.0,
    ])

    flip_force = state.info["orientation_flip_force"]
    if pcfg.orientation_flip_enabled:
      flip_active = support_released & (hold_steps >= pcfg.orientation_flip_min_hold_steps)
      should_update_flip = flip_active & (
          hold_steps % pcfg.orientation_flip_change_interval == 0
      )
      flip_dir = jax.random.normal(flip_rng, (3,))
      flip_dir = flip_dir / (jp.linalg.norm(flip_dir) + 1e-8)
      new_flip_force = (
          flip_dir * self._cube_mass * 9.81 * pcfg.orientation_flip_force_scale
      )
      flip_force = jp.where(should_update_flip, new_flip_force, flip_force)
      flip_force = jp.where(flip_active, flip_force, jp.zeros(3))

    total_force = self._clip_force_vector(ext_force + gravity_force + flip_force)
    xfrc = data.xfrc_applied.at[self._cube_body_id, :3].set(total_force)
    data = data.replace(xfrc_applied=xfrc)

    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    nonfinite_state = self._has_nonfinite_state(data)

    tip_finger_forces = self._get_contact_forces_efc(data)
    tip_contact_flags = self._get_tip_contact_flags(data)

    support_timer = state.info["support_timer"] + 1
    ema_tip_finger_forces = (
        0.8 * state.info["ema_tip_finger_forces"] + 0.2 * tip_finger_forces
    )

    obs_alpha = float(self._config.tactile_config.obs_force_ema_alpha)
    obs_tactile_ema = (
        obs_alpha * state.info["obs_tactile_ema"]
        + (1.0 - obs_alpha) * tip_finger_forces
    )

    support_released = self._should_release_support(
        state.info["support_released"], support_timer,
        state.info["support_release_steps"], ema_tip_finger_forces,
        tip_contact_flags,
    )
    data = jax.lax.cond(
        state.info["use_support"],
        lambda d: self._set_support_state(d, support_released, support_timer),
        lambda d: self._set_support_state(d, jp.array(True), support_timer),
        data,
    )

    info = state.info.copy()
    info["rng"] = step_rng
    info["motor_targets"] = motor_targets
    info["ema_tip_finger_forces"] = ema_tip_finger_forces
    info["tip_contact_flags"] = tip_contact_flags
    info["obs_tactile_ema"] = obs_tactile_ema
    info["support_released"] = support_released
    info["support_timer"] = support_timer
    info["gravity_tilt_angle"] = tilt_angle
    info["perturbation_force"] = ext_force
    info["orientation_flip_force"] = flip_force
    info["last_last_act"] = state.info["last_act"]
    info["last_act"] = action

    def _handle_nonfinite(_):
      return self._make_nonfinite_failure_state(
          state, action, motor_targets, info,
      )

    def _handle_normal(_):
      cube_pos_hold = self.get_cube_position(data)
      cube_above = cube_pos_hold[2] > (self._spawn_z - 0.012)
      wrap_forces, wrap_active, wrap_count = self._get_wrap_contact_state(
          ema_tip_finger_forces, tip_contact_flags,
      )
      thumb_active = wrap_active[3] > 0.0
      palm_contact_hold = self._has_cube_contact(data, self._palm_contact_gids)
      mean_wrap_force = jp.mean(wrap_forces[:3]) + 0.5 * wrap_forces[3]
      hold_ready = (
          (wrap_count >= 3.0)
          | ((wrap_count >= 2.0) & thumb_active & (palm_contact_hold > 0.0))
      )
      is_holding = (
          cube_above
          & support_released
          & hold_ready
          & (mean_wrap_force > 0.08)
      )
      info["stable_hold_steps"] = jp.where(
          is_holding,
          state.info["stable_hold_steps"] + 1,
          jp.maximum(
              state.info["stable_hold_steps"] - 2,
              jp.array(0, dtype=jp.int32),
          ),
      )

      obs = self._get_obs(data, info, state.obs["state"])
      done = self._get_termination(data) & (
          support_timer >= self._termination_grace_steps
      )

      rewards = self._get_reward(data, action, info, done)
      rewards = {
          k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
      }
      reward = sum(rewards.values()) * self.dt
      diagnostics = self._get_diagnostics(
          data, tip_finger_forces, tip_contact_flags, support_released,
          info["stable_hold_steps"], done,
      )
      diagnostics["diagnostic/lifted_reset"] = info.get(
          "lifted_reset", jp.array(False),
      ).astype(jp.float32)
      diagnostics["diagnostic/nonfinite_state"] = jp.array(0.0, dtype=reward.dtype)

      new_metrics = {}
      for k, v in rewards.items():
        new_metrics[f"reward/{k}"] = v
        new_metrics[f"reward_sq/{k}"] = jp.square(v)
      new_metrics.update(diagnostics)
      new_metrics["reward"] = reward

      done = done.astype(reward.dtype)
      return state.replace(
          data=data, obs=obs, reward=reward, done=done,
          metrics=new_metrics, info=info,
      )

    return jax.lax.cond(nonfinite_state, _handle_nonfinite, _handle_normal, operand=None)

  def _get_diagnostics(
      self,
      data: mjx.Data,
      tip_force: jax.Array,
      tip_contact_flags: jax.Array,
      support_released: jax.Array,
      stable_hold_steps: jax.Array,
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    abs_f = jp.abs(tip_force)
    wrap_forces = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]])
    any_abs_f = jp.abs(self._get_contact_forces_efc(
        data, self._finger_contact_gids,
    ))
    wrap_any_forces = jp.array([any_abs_f[0], any_abs_f[1], any_abs_f[2], any_abs_f[4]])
    active_th = self._config.reward_config.finger_active_threshold
    wrap_geom = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[2],
        tip_contact_flags[4],
    ])
    wrap_active = jp.maximum((wrap_forces > active_th).astype(jp.float32), wrap_geom)
    wrap_count = jp.sum(wrap_active)
    wrap_any_active = (wrap_any_forces > active_th).astype(jp.float32)
    wrap_any_count = jp.sum(wrap_any_active)
    non_tip_primary_contact = (
        (wrap_any_count > wrap_count) & (wrap_any_count >= 2.0)
    ).astype(jp.float32)
    palm_contact = self._has_cube_contact(data, self._palm_contact_gids)
    thumb_ready = wrap_active[3] > 0.0
    hold_contact = (
        (wrap_count >= 3.0)
        | ((wrap_count >= 2.0) & thumb_ready & (palm_contact > 0.0))
    ).astype(jp.float32)
    two_plus_contact = ((wrap_count >= 2.0) & thumb_ready).astype(jp.float32)

    cube_pos = self.get_cube_position(data)
    cube_linvel = self.get_cube_linvel(data)
    lift_height = cube_pos[2] - self._spawn_z
    lift_threshold = getattr(
        self._config.reward_config, 'lift_success_threshold_m', 0.010,
    )
    lift_success = (lift_height > lift_threshold).astype(jp.float32)
    hold_success_steps = int(np.round(
        getattr(self._config.reward_config, 'hold_success_sec', 30.0) / self.dt,
    ))
    cube_above = (cube_pos[2] > (self._spawn_z - 0.012)).astype(jp.float32)
    hold_success = (
        (stable_hold_steps >= hold_success_steps)
        & (cube_above > 0.0)
        & (hold_contact > 0.0)
    ).astype(jp.float32)

    lin_speed = jp.linalg.norm(cube_linvel)
    slip_event = (
        (hold_contact > 0.0)
        & support_released
        & ((lin_speed > 0.12) | (cube_linvel[2] < -0.03))
    ).astype(jp.float32)
    pinky_contact = jp.maximum(
        (abs_f[3] > active_th).astype(jp.float32),
        self._has_cube_contact(data, self._finger_contact_gids["pinky"]),
    )
    drop = done.astype(jp.float32)

    return {
        "diagnostic/success": hold_success,
        "diagnostic/three_finger_contact": hold_contact,
        "diagnostic/two_plus_primary_contact": two_plus_contact,
        "diagnostic/contact_duration_sec": hold_contact * self.dt,
        "diagnostic/lift_success": lift_success,
        "diagnostic/hold_success": hold_success,
        "diagnostic/drop": drop,
        "diagnostic/normal_force_mean": jp.mean(wrap_forces),
        "diagnostic/tangent_force_approx": jp.linalg.norm(cube_linvel[:2]) * hold_contact,
        "diagnostic/slip_event": slip_event,
        "diagnostic/primary_active_count": wrap_count,
        "diagnostic/primary_any_active_count": wrap_any_count,
        "diagnostic/non_tip_primary_contact": non_tip_primary_contact,
        "diagnostic/nonprimary_contact": pinky_contact,
        "diagnostic/palm_contact": palm_contact,
        "diagnostic/support_released": support_released.astype(jp.float32),
        "diagnostic/lift_height": lift_height,
        "termination/drop": drop,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    tip_world = self.get_fingertip_positions(data).reshape(5, 3) + palm_pos
    tip_dists = jp.linalg.norm(tip_world - cube_pos[None, :], axis=1)
    min_tip_dist = jp.min(tip_dists)
    wrap_tip_dists = jp.array([tip_dists[0], tip_dists[1], tip_dists[2], tip_dists[4]])
    mean_wrap_dist = jp.mean(wrap_tip_dists)
    max_wrap_dist = jp.max(wrap_tip_dists)
    hand_q = data.qpos[self._hand_qids]
    tip_finger_forces = info["ema_tip_finger_forces"]
    tip_contact_flags = info.get("tip_contact_flags", jp.zeros(5, dtype=jp.float32))
    cube_linvel = self.get_cube_linvel(data)
    cube_angvel = self.get_cube_angvel(data)

    near_gate = jp.clip(1.0 - min_tip_dist / 0.10, 0.0, 1.0)
    mcp_angles = jp.take(hand_q, jp.array(grasp_cube_v2_force._V2_FINGER_MCP_IDS))
    mcp_gate = jp.clip(jp.mean(mcp_angles) / 0.8, 0.0, 1.0)

    support_released = info.get("support_released", jp.array(True))
    if self._support_ramp_steps > 0:
      support_timer = info.get("support_timer", jp.array(0))
      ramp_elapsed = jp.clip(
          (support_timer - self._support_release_steps).astype(jp.float32),
          0.0, float(self._support_ramp_steps),
      )
      released_gate = jp.where(
          support_released, ramp_elapsed / float(self._support_ramp_steps), 0.0,
      )
    else:
      released_gate = support_released.astype(jp.float32)

    wrap_forces, wrap_active, wrap_count = self._get_wrap_contact_state(
        tip_finger_forces, tip_contact_flags,
    )
    thumb_active = wrap_active[3]
    wrap_gate = jp.clip((wrap_count - 1.0) / 3.0, 0.0, 1.0) * (0.4 + 0.6 * thumb_active)
    palm_contact = self._has_cube_contact(data, self._palm_contact_gids)
    palm_wrap_gate = (
        palm_contact
        * thumb_active
        * jp.clip((wrap_count - 1.0) / 2.0, 0.0, 1.0)
    )
    strong_wrap_gate = jp.maximum(
        jp.clip((wrap_count - 2.0) / 2.0, 0.0, 1.0) * thumb_active,
        0.8 * palm_wrap_gate,
    )
    pinky_contact = jp.maximum(
        (jp.abs(tip_finger_forces[3]) > self._config.reward_config.finger_active_threshold).astype(jp.float32),
        self._has_cube_contact(data, self._finger_contact_gids["pinky"]),
    )

    return {
        "approach": self._reward_approach(mean_wrap_dist),
        "three_finger_proximity": self._reward_three_finger_proximity(max_wrap_dist),
        "contact": self._reward_contact(tip_world, cube_pos),
        "thumb_engage": self._reward_thumb_engage(tip_dists),
        "closure": self._reward_closure(hand_q) * near_gate,
        "pip_closure": self._reward_pip_closure(hand_q) * near_gate * mcp_gate,
        "human_pose": self._reward_human_pose(hand_q) * near_gate,
        "grip_force": self._reward_grip_force(tip_finger_forces),
        "hold_position": self._reward_hold_position(cube_pos, cube_linvel) * released_gate * strong_wrap_gate,
        "stable_hold": self._reward_stable_hold(
            cube_pos, cube_linvel, cube_angvel, tip_finger_forces,
        ) * released_gate * strong_wrap_gate,
        "force_contact": self._reward_force_contact(tip_finger_forces) * released_gate,
        "primary_geom_contact": self._reward_primary_geom_contact(tip_contact_flags),
        "progressive_hold": self._reward_progressive_hold(info) * strong_wrap_gate,
        "sustained_hold_bonus": self._reward_sustained_hold_bonus(info) * strong_wrap_gate,
        "force_balance": self._reward_force_balance(tip_finger_forces),
        "finger_participation": self._reward_finger_participation(tip_finger_forces) * (0.4 + 0.6 * wrap_gate),
        "thumb_opposition": self._reward_thumb_opposition(
            tip_world, cube_pos, tip_finger_forces,
        ) * (0.4 + 0.6 * wrap_gate),
        "primary_finger_force": self._reward_primary_finger_force(
            tip_finger_forces,
        ) * (0.4 + 0.6 * wrap_gate),
        "pre_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * (1.0 - released_gate) * (0.4 + 0.6 * wrap_gate),
        "post_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * released_gate * (0.4 + 0.6 * wrap_gate),
        "post_release_survival": self._reward_post_release_survival(
            cube_pos, cube_linvel,
        ) * released_gate * strong_wrap_gate,
        "post_release_cheat_contact": jp.array(0.0, dtype=jp.float32),
        "post_release_slip": self._cost_post_release_slip(
            cube_linvel,
        ) * released_gate * strong_wrap_gate,
        "post_release_pose_hold": self._reward_post_release_pose_hold(
            hand_q,
        ) * released_gate * strong_wrap_gate,
        "soft_contact": self._reward_soft_contact(tip_finger_forces),
        "idle_follow": self._reward_idle_follow(hand_q, tip_finger_forces),
        "height": self._reward_height(cube_pos, palm_pos) * wrap_gate,
        "survival": 1.0 - done,
        "termination": done,
        "drop_risk": self._cost_drop_risk(cube_pos, cube_linvel),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "action_accel": self._cost_action_accel(action, info["last_act"], info["last_last_act"]),
        "torques": self._cost_torques(data.actuator_force),
        "force_overload": self._cost_force_overload(tip_finger_forces),
        "palm_contact": palm_contact * (0.3 + 0.7 * wrap_gate),
        "nonprimary_contact": pinky_contact * (0.2 + 0.8 * wrap_gate),
    }


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Bottle-specific domain randomization."""
  env = BottleGraspV2Force()
  mj_model = env.mj_model
  cube_body_id = mj_model.body("cube").id
  object_geom_ids = np.array(
      [
          geom_id for geom_id in range(mj_model.ngeom)
          if mj_model.geom_bodyid[geom_id] == cube_body_id
      ],
      dtype=np.int32,
  )
  hand_qids = mjx_env.get_qpos_ids(mj_model, consts.V2_JOINT_NAMES)
  hand_body_names = [
      "palm",
      "right_index_proximal",
      "right_index_distal",
      "right_middle_proximal",
      "right_middle_distal",
      "right_ring_proximal",
      "right_ring_distal",
      "right_pinky_proximal",
      "right_pinky_distal",
      "right_thumb_base",
      "right_thumb_mid",
      "right_thumb_tip",
  ]
  hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
  fingertip_geom_ids = np.array(
      sorted({gid for gids in env._finger_tip_contact_gids.values() for gid in gids}),
      dtype=np.int32,
  )
  nq_hand = consts.V2_NQ

  @jax.vmap
  def rand(local_rng):
    local_rng, key = jax.random.split(local_rng)
    bottle_friction = jax.random.uniform(key, (1,), minval=0.75, maxval=1.35)
    geom_friction = model.geom_friction.at[object_geom_ids, 0].set(bottle_friction)

    local_rng, key = jax.random.split(local_rng)
    fingertip_friction = jax.random.uniform(key, (1,), minval=0.9, maxval=1.5)
    geom_friction = geom_friction.at[fingertip_geom_ids, 0].set(fingertip_friction)

    local_rng, key1, key2 = jax.random.split(local_rng, 3)
    dmass = jax.random.uniform(key1, minval=0.75, maxval=1.30)
    cube_mass = model.body_mass[cube_body_id]
    body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass,
    )
    dpos = jax.random.uniform(key2, (3,), minval=-4e-3, maxval=4e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos,
    )

    local_rng, key = jax.random.split(local_rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(key, shape=(nq_hand,), minval=-0.035, maxval=0.035),
    )

    local_rng, key = jax.random.split(local_rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=0.8, maxval=1.2,
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    local_rng, key = jax.random.split(local_rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=1.0, maxval=1.05,
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    local_rng, key = jax.random.split(local_rng)
    dmass_hand = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1,
    )
    body_mass = body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass_hand,
    )

    local_rng, key = jax.random.split(local_rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.9, maxval=1.1,
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    local_rng, key = jax.random.split(local_rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
        key, (nq_hand,), minval=0.9, maxval=1.1,
    )
    dof_damping = model.dof_damping.at[hand_qids].set(kd)

    return (
        geom_friction,
        body_mass,
        body_inertia,
        body_ipos,
        qpos0,
        dof_frictionloss,
        dof_armature,
        dof_damping,
        actuator_gainprm,
        actuator_biasprm,
    )

  (
      geom_friction,
      body_mass,
      body_inertia,
      body_ipos,
      qpos0,
      dof_frictionloss,
      dof_armature,
      dof_damping,
      actuator_gainprm,
      actuator_biasprm,
  ) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({
      "geom_friction": 0,
      "body_mass": 0,
      "body_inertia": 0,
      "body_ipos": 0,
      "qpos0": 0,
      "dof_frictionloss": 0,
      "dof_armature": 0,
      "dof_damping": 0,
      "actuator_gainprm": 0,
      "actuator_biasprm": 0,
  })

  model = model.tree_replace({
      "geom_friction": geom_friction,
      "body_mass": body_mass,
      "body_inertia": body_inertia,
      "body_ipos": body_ipos,
      "qpos0": qpos0,
      "dof_frictionloss": dof_frictionloss,
      "dof_armature": dof_armature,
      "dof_damping": dof_damping,
      "actuator_gainprm": actuator_gainprm,
      "actuator_biasprm": actuator_biasprm,
  })
  return model, in_axes
