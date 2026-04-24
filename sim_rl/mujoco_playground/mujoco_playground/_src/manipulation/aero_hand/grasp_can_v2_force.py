# Copyright 2025 TetherIA Inc.
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""V2 灵犀手小圆柱/易拉罐代理抓握任务."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import grasp_bottle_v2_force


_CAN_DIAGNOSTIC_KEYS = (
    "diagnostic/index_wrap_contact",
    "diagnostic/middle_wrap_contact",
    "diagnostic/ring_wrap_contact",
    "diagnostic/thumb_wrap_contact",
    "diagnostic/triad_wrap_count",
    "diagnostic/joint_palm_contact",
)


def default_config() -> config_dict.ConfigDict:
  cfg = grasp_bottle_v2_force.default_config()

  # 当前目标改为更小的圆柱代理，控制和接触阈值再收紧一档。
  # CAN06: 第一阶段先让策略在“当前 pose + 支撑存在 + 无DR”下学会稳定包裹，
  # 后续再逐层加回晃动/翻转扰动。
  cfg.action_scale = [0.07, 0.22, 0.22, 0.22, 0.24, 0.09]

  cfg.support_config.release_after_sec = 9.8
  cfg.support_config.release_ramp_sec = 1.6
  cfg.support_config.force_release_after_sec = 13.2
  # Keep the user-provided can/hand pose fixed.  The support top surface is
  # aligned with the can bottom instead of intersecting the can body, so the
  # hand must learn the load transfer rather than relying on penetration.
  cfg.support_config.support_pos = [0.008041, -0.040830, 0.108128]
  cfg.support_config.random_release = False
  cfg.support_config.random_release_min_sec = 4.2
  cfg.support_config.random_release_max_sec = 4.2
  cfg.support_config.min_release_active_fingers = 4
  cfg.support_config.min_release_force = 0.12
  cfg.support_config.require_grasp_for_release = True

  cfg.spawn_config.cube_pos = [0.008041, -0.040830, 0.132128]
  cfg.spawn_config.cube_jitter = [0.0, 0.0, 0.0]
  cfg.spawn_config.support_enabled = True

  cfg.reset_config.pre_grasp_fraction = 1.0
  cfg.reset_config.pre_grasp_noise_scale = 0.01
  cfg.reset_config.lifted_grasp_fraction = 0.0
  cfg.reset_config.lifted_grasp_noise_scale = 0.01
  cfg.reset_config.lifted_cube_z_offset = 0.012

  cfg.perturbation_config.external_force_enabled = False
  cfg.perturbation_config.gravity_perturbation_enabled = False
  cfg.perturbation_config.orientation_flip_enabled = False

  scales = cfg.reward_config.scales
  scales.hold_position = 900.0
  scales.stable_hold = 600.0
  scales.progressive_hold = 200.0
  scales.sustained_hold_bonus = 260.0
  scales.supported_hold_position = 5.0
  scales.short_hold_seed = 4.0
  scales.late_support_dependence = -110.0
  scales.release_height_retention = 980.0
  scales.post_release_force_support = 520.0
  scales.post_release_joint_palm_hold = 1040.0
  scales.cradle_lock = 820.0
  scales.core_force_tripod = 100.0
  scales.joint_palm_clamp = 820.0
  scales.whole_hand_wrap = 120.0
  scales.simultaneous_wrap = 240.0
  scales.core_cup_wrap = 300.0
  scales.core_contact = 700.0
  scales.ring_engage = 190.0
  scales.ring_proximity = 55.0
  scales.contact = 6.0
  scales.closure = 8.0
  scales.pip_closure = 6.0
  scales.force_contact = 28.0
  scales.grip_force = 60.0
  scales.force_balance = 180.0
  scales.finger_participation = 60.0
  scales.thumb_engage = 24.0
  scales.thumb_opposition = 46.0
  scales.soft_contact = 12.0
  scales.primary_finger_force = 190.0
  scales.primary_geom_contact = 90.0
  scales.human_pose = 6.0
  scales.pre_release_grasp = 150.0
  scales.post_release_grasp = 360.0
  scales.post_release_survival = 1650.0
  scales.post_release_cheat_contact = -40.0
  scales.post_release_slip = -140.0
  scales.post_release_pose_hold = 140.0
  scales.height = 12.0
  scales.termination = -700.0
  scales.drop_risk = -180.0
  scales.palm_contact = -2.0
  scales.nonprimary_contact = -4.0
  scales.three_finger_proximity = 30.0
  scales.force_overload = -30.0

  cfg.reward_config.force_contact_threshold = 0.015
  cfg.reward_config.force_contact_saturation = 0.90
  cfg.reward_config.force_overload_threshold = 0.85
  cfg.reward_config.force_overload_soft_width = 0.18
  cfg.reward_config.soft_contact_fmin = 0.015
  cfg.reward_config.soft_contact_fmax = 0.65
  cfg.reward_config.finger_active_threshold = 0.020
  cfg.reward_config.target_lift_m = 0.012
  cfg.reward_config.lift_success_threshold_m = 0.008
  cfg.stability_config.max_abs_action = 0.55
  cfg.stability_config.motor_delta_clip = [0.016, 0.035, 0.040, 0.040, 0.028, 0.026]
  cfg.stability_config.terminate_on_nonfinite = True
  cfg.stability_config.nonfinite_penalty_mult = 3.0

  return cfg


class CanGraspV2Force(grasp_bottle_v2_force.BottleGraspV2Force):
  """Small-cylinder grasp task using the bottle-wrap scaffold with gentler force limits."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=xml_path or consts.GRASP_V2_CAN_XML.as_posix(),
    )
    # 小圆柱半径更小，四指与拇指都收窄一点，保持“像人手包裹罐身”的初始姿态。
    self._pre_grasp_pose = np.array([
        0.3400,
        0.3145,
        0.3520,
        0.3256,
        0.3380,
        0.3127,
        0.3000,
        0.2775,
        1.18,
        0.1200,
        0.18,
    ], dtype=np.float32)
    self._lifted_grasp_pose = self._pre_grasp_pose.copy()
    self._spawn_quat = jp.array([
        0.707715, -0.007412, 0.706458, 0.000577,
    ], dtype=jp.float32)
    self._default_ctrl = jp.array([
        1.18, 0.36, 0.74, 0.78, 0.72, 0.52,
    ], dtype=jp.float32)
    self._lifted_grasp_ctrl = np.array([
        1.18, 0.36, 0.74, 0.78, 0.72, 0.52,
    ], dtype=np.float32)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    state = super().reset(rng)
    qpos = state.data.qpos.at[-4:].set(self._spawn_quat)
    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=state.data.qvel,
        ctrl=state.data.ctrl,
        mocap_pos=state.data.mocap_pos,
    )
    obs = self._get_obs(data, state.info, state.obs["state"])
    metrics = state.metrics.copy()
    for key in _CAN_DIAGNOSTIC_KEYS:
      metrics[key] = jp.zeros(())
    return state.replace(data=data, obs=obs, metrics=metrics)

  def _reward_closure(self, hand_q: jax.Array) -> jax.Array:
    """Reward gentle closure from the user-provided starting pose."""
    q = jp.array([hand_q[0], hand_q[2], hand_q[4], hand_q[6], hand_q[8], hand_q[9], hand_q[10]])
    target = jp.array([
        self._pre_grasp_pose[0],
        self._pre_grasp_pose[2],
        self._pre_grasp_pose[4],
        self._pre_grasp_pose[6],
        self._pre_grasp_pose[8],
        self._pre_grasp_pose[9],
        self._pre_grasp_pose[10],
    ], dtype=jp.float32)
    weights = jp.array([1.1, 1.2, 1.15, 0.85, 1.3, 0.9, 1.0], dtype=jp.float32)
    close_window = jp.array([0.18, 0.18, 0.16, 0.14, 0.14, 0.08, 0.08], dtype=jp.float32)
    over_window = jp.array([0.34, 0.34, 0.30, 0.25, 0.24, 0.16, 0.15], dtype=jp.float32)
    delta = q - target
    gentle_close = jp.clip(0.35 + delta / (close_window + 1e-6), 0.0, 1.0)
    over_close = jp.clip((delta - over_window) / (over_window + 1e-6), 0.0, 1.0)
    per_joint = gentle_close * (1.0 - 0.65 * over_close)
    return jp.sum(weights * per_joint) / jp.sum(weights)

  def _reward_pip_closure(self, hand_q: jax.Array) -> jax.Array:
    """Reward passive PIP follow-through after the can start pose."""
    q = jp.array([hand_q[1], hand_q[3], hand_q[5], hand_q[7]], dtype=jp.float32)
    target = jp.array([
        self._pre_grasp_pose[1],
        self._pre_grasp_pose[3],
        self._pre_grasp_pose[5],
        self._pre_grasp_pose[7],
    ], dtype=jp.float32)
    weights = jp.array([1.1, 1.2, 1.0, 0.8], dtype=jp.float32)
    close_window = jp.array([0.16, 0.16, 0.14, 0.12], dtype=jp.float32)
    over_window = jp.array([0.30, 0.30, 0.26, 0.22], dtype=jp.float32)
    delta = q - target
    gentle_close = jp.clip(0.35 + delta / (close_window + 1e-6), 0.0, 1.0)
    over_close = jp.clip((delta - over_window) / (over_window + 1e-6), 0.0, 1.0)
    per_joint = gentle_close * (1.0 - 0.65 * over_close)
    return jp.sum(weights * per_joint) / jp.sum(weights)

  def _reward_human_pose(self, hand_q: jax.Array) -> jax.Array:
    """Cup-style whole-hand pose prior centered on the user-provided initial grasp."""
    target = jp.array(self._pre_grasp_pose, dtype=jp.float32)
    tol = jp.array([0.22, 0.20, 0.22, 0.20, 0.20, 0.18, 0.18, 0.16, 0.18, 0.10, 0.10], dtype=jp.float32)
    weights = jp.array([1.1, 0.9, 1.2, 1.0, 1.15, 0.95, 0.9, 0.7, 1.35, 0.95, 1.0], dtype=jp.float32)
    err = (hand_q - target) / (tol + 1e-6)
    return jp.sum(weights * jp.exp(-2.2 * jp.square(err))) / jp.sum(weights)

  def _reward_grip_force(self, tip_force: jax.Array) -> jax.Array:
    """Can grasp should reward thumb+index+middle+ring wrap, not cube pinch."""
    abs_f = jp.abs(tip_force)
    wrap = jp.array([abs_f[0], abs_f[1], abs_f[2], abs_f[4]], dtype=jp.float32)
    per_finger = jp.clip(wrap / 0.08, 0.0, 1.0)
    weights = jp.array([1.0, 1.0, 0.9, 1.1], dtype=jp.float32)
    return jp.sum(weights * per_finger) / jp.sum(weights)

  def _reward_force_balance(self, tip_force: jax.Array) -> jax.Array:
    """Encourage balanced load-bearing force over index/middle/thumb."""
    abs_f = jp.abs(tip_force)
    core = jp.array([abs_f[0], abs_f[1], abs_f[4]], dtype=jp.float32)
    mean_f = jp.mean(core)
    force_level = jp.clip((mean_f - 0.015) / 0.10, 0.0, 1.0)
    rel_std = jp.std(core) / (mean_f + 1e-6)
    ring_assist = jp.clip(abs_f[2] / 0.05, 0.0, 1.0)
    return force_level * jp.clip(1.0 - rel_std, 0.0, 1.0) * (0.9 + 0.1 * ring_assist)

  def _reward_primary_geom_contact(self, tip_contact_flags: jax.Array) -> jax.Array:
    """Reward simultaneous wrap contact over index/middle/ring/thumb."""
    wrap = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[2],
        tip_contact_flags[4],
    ], dtype=jp.float32)
    triad = jp.mean(wrap[:3])
    return jp.mean(wrap) * (0.25 + 0.45 * triad + 0.30 * wrap[3])

  def _reward_finger_participation(self, tip_force: jax.Array) -> jax.Array:
    """Prioritize the four wrap digits; pinky can help but should not dominate."""
    th = self._config.reward_config.finger_active_threshold
    active = (jp.abs(tip_force) > th).astype(jp.float32)
    wrap = jp.array([active[0], active[1], active[2], active[4]], dtype=jp.float32)
    pinky = active[3]
    return 0.85 * jp.mean(wrap) + 0.15 * pinky

  def _reward_contact(self, tip_world, cube_pos):
    del cube_pos
    target = self._spawn_pos
    body_half_extents = jp.array([0.034, 0.024, 0.024], dtype=jp.float32)
    diff = jp.abs(tip_world - target[None, :]) - body_half_extents
    surface_dists = jp.linalg.norm(jp.maximum(diff, 0.0), axis=1)
    return jp.mean(jp.exp(-4.8 * surface_dists))

  def _reward_thumb_opposition(
      self, tip_world: jax.Array, cube_pos: jax.Array,
      tip_force: jax.Array,
  ) -> jax.Array:
    """Thumb should support the can from below/front, not only oppose from the side."""
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    thumb_soft = jax.nn.sigmoid(35.0 * (abs_f[4] - th * 0.45))
    index_soft = jax.nn.sigmoid(28.0 * (abs_f[0] - th * 0.35))
    middle_soft = jax.nn.sigmoid(28.0 * (abs_f[1] - th * 0.35))
    ring_soft = jax.nn.sigmoid(28.0 * (abs_f[2] - th * 0.30))

    thumb_tip = tip_world[4]
    thumb_delta = cube_pos - thumb_tip
    thumb_near = jp.exp(-18.0 * jp.linalg.norm(thumb_delta))
    # Bias the policy toward a thumb pose that sits slightly below the can
    # centerline and contributes load support while the fingers press inward.
    thumb_under = jp.clip((cube_pos[2] - thumb_tip[2] + 0.018) / 0.040, 0.0, 1.0)
    finger_clamp = jp.clip(0.45 * middle_soft + 0.30 * index_soft + 0.25 * ring_soft, 0.0, 1.0)
    return thumb_soft * thumb_near * (0.25 + 0.75 * finger_clamp) * (0.35 + 0.65 * thumb_under)

  def _get_wrap_contact_flags(self, data) -> jax.Array:
    return jp.array([
        self._has_cube_contact(data, self._finger_contact_gids["index"]),
        self._has_cube_contact(data, self._finger_contact_gids["middle"]),
        self._has_cube_contact(data, self._finger_contact_gids["ring"]),
        self._has_cube_contact(data, self._finger_contact_gids["pinky"]),
        self._has_cube_contact(data, self._finger_contact_gids["thumb"]),
    ], dtype=jp.float32)

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

    whole_finger_forces = self._get_contact_forces_efc(data, self._finger_contact_gids)
    whole_contact_flags = self._get_wrap_contact_flags(data)

    support_timer = state.info["support_timer"] + 1
    ema_finger_forces = (
        0.8 * state.info["ema_tip_finger_forces"] + 0.2 * whole_finger_forces
    )

    obs_alpha = float(self._config.tactile_config.obs_force_ema_alpha)
    obs_tactile_ema = (
        obs_alpha * state.info["obs_tactile_ema"]
        + (1.0 - obs_alpha) * whole_finger_forces
    )

    support_released = self._should_release_support(
        state.info["support_released"], support_timer,
        state.info["support_release_steps"], ema_finger_forces,
        whole_contact_flags,
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
    info["ema_tip_finger_forces"] = ema_finger_forces
    info["tip_contact_flags"] = whole_contact_flags
    info["obs_tactile_ema"] = obs_tactile_ema
    info["support_released"] = support_released
    info["support_timer"] = support_timer
    info["gravity_tilt_angle"] = tilt_angle
    info["perturbation_force"] = ext_force
    info["orientation_flip_force"] = flip_force
    info["last_last_act"] = state.info["last_act"]
    info["last_act"] = action

    def _handle_nonfinite(_):
      failed_state = self._make_nonfinite_failure_state(
          state, action, motor_targets, info,
      )
      metrics = failed_state.metrics.copy()
      for key in _CAN_DIAGNOSTIC_KEYS:
        metrics[key] = jp.zeros(())
      return failed_state.replace(metrics=metrics)

    def _handle_normal(_):
      cube_pos_hold = self.get_cube_position(data)
      cube_above = cube_pos_hold[2] > (self._spawn_z - 0.012)
      wrap_forces, wrap_active, wrap_count = self._get_wrap_contact_state(
          ema_finger_forces, whole_contact_flags,
      )
      triad_count = jp.sum(wrap_active[:3])
      thumb_active = wrap_active[3] > 0.0
      palm_contact_hold = self._has_cube_contact(data, self._palm_contact_gids)
      mean_wrap_force = jp.mean(wrap_forces[:3]) + 0.5 * wrap_forces[3]
      hold_ready = (
          ((wrap_count >= 3.0) & thumb_active)
          | ((triad_count >= 2.0) & thumb_active & (palm_contact_hold > 0.0))
          | (
              (palm_contact_hold > 0.0)
              & (wrap_forces[1] > 0.016)
              & (wrap_forces[2] > 0.014)
              & (wrap_forces[3] > 0.012)
          )
          | (
              (wrap_forces[0] > 0.012)
              & (wrap_forces[1] > 0.012)
              & (wrap_forces[3] > 0.012)
          )
          | (
              (palm_contact_hold > 0.0)
              & (wrap_forces[3] > 0.012)
              & ((wrap_forces[0] + wrap_forces[1]) > 0.030)
          )
      )
      is_holding = (
          cube_above
          & hold_ready
          & (mean_wrap_force > 0.018)
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
      drop_now = self._get_termination(data)
      curriculum_armed = support_released | (
          support_timer >= self._force_release_steps
      )
      low_before_release = (
          (cube_pos_hold[2] < (self._spawn_z - 0.035))
          & (~curriculum_armed)
      )
      done = (drop_now & curriculum_armed) | low_before_release
      done = done & (support_timer >= self._termination_grace_steps)

      rewards = self._get_reward(data, action, info, done)
      rewards = {
          k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
      }
      reward = sum(rewards.values()) * self.dt
      diagnostics = self._get_diagnostics(
          data, ema_finger_forces, whole_contact_flags, support_released,
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

  def _is_grasp_ready_for_release(
      self,
      tip_finger_forces: jax.Array,
      tip_contact_flags: jax.Array | None = None,
  ) -> jax.Array:
    """Cylinder release must come from a human-like wrap, not pinch transfer."""
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
    min_active = float(self._config.support_config.min_release_active_fingers)
    thumb_active = active[3] > 0.5
    middle_ready = active[1] > 0.5
    ring_ready = active[2] > 0.5
    triad_ready = jp.sum(active[:3]) >= 2.0
    force_ok = jp.sum(
        wrap * jp.array([1.0, 1.0, 0.9, 1.0], dtype=jp.float32)
    ) >= (self._config.support_config.min_release_force * 3.5)
    return (
        (active_count >= min_active)
        & thumb_active
        & middle_ready
        & ring_ready
        & triad_ready
        & force_ok
    )

  def _get_diagnostics(
      self,
      data,
      tip_force,
      tip_contact_flags,
      support_released,
      stable_hold_steps,
      done,
  ):
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
    triad_count = jp.sum(wrap_active[:3])
    joint_palm_contact = (
        (palm_contact > 0.0)
        & (thumb_ready)
        & (triad_count >= 2.0)
    ).astype(jp.float32)
    hold_contact = (
        ((wrap_count >= 3.0) & thumb_ready)
        | ((triad_count >= 2.0) & thumb_ready & (palm_contact > 0.0))
    ).astype(jp.float32)
    two_plus_contact = ((triad_count >= 2.0) & thumb_ready).astype(jp.float32)

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
        "diagnostic/index_wrap_contact": wrap_active[0],
        "diagnostic/middle_wrap_contact": wrap_active[1],
        "diagnostic/ring_wrap_contact": wrap_active[2],
        "diagnostic/thumb_wrap_contact": wrap_active[3],
        "diagnostic/triad_wrap_count": triad_count,
        "diagnostic/primary_any_active_count": wrap_any_count,
        "diagnostic/non_tip_primary_contact": non_tip_primary_contact,
        "diagnostic/nonprimary_contact": pinky_contact,
        "diagnostic/palm_contact": palm_contact,
        "diagnostic/joint_palm_contact": joint_palm_contact,
        "diagnostic/support_released": support_released.astype(jp.float32),
        "diagnostic/lift_height": lift_height,
        "termination/drop": drop,
    }

  def _get_reward(
      self,
      data,
      action,
      info,
      done,
  ):
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
    mcp_angles = jp.take(hand_q, jp.array(grasp_bottle_v2_force.grasp_cube_v2_force._V2_FINGER_MCP_IDS))
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
    support_timer = info.get("support_timer", jp.array(0))
    late_support_steps = jp.maximum(
        support_timer - (self._support_release_steps + max(5, self._support_ramp_steps // 2)),
        jp.array(0, dtype=jp.int32),
    ).astype(jp.float32)
    late_support_gate = jp.clip(late_support_steps / 30.0, 0.0, 1.0)

    wrap_forces, wrap_active, wrap_count = self._get_wrap_contact_state(
        tip_finger_forces, tip_contact_flags,
    )
    thumb_active = wrap_active[3]
    triad_count = jp.sum(wrap_active[:3])
    wrap_gate = jp.clip((triad_count - 1.0) / 2.0, 0.0, 1.0) * (0.35 + 0.65 * thumb_active)
    palm_contact = self._has_cube_contact(data, self._palm_contact_gids)
    palm_wrap_gate = (
        palm_contact
        * thumb_active
        * jp.clip((triad_count - 1.0) / 2.0, 0.0, 1.0)
    )
    strong_wrap_gate = jp.maximum(
        jp.clip((triad_count - 2.0) / 1.0, 0.0, 1.0) * thumb_active,
        0.75 * palm_wrap_gate,
    )
    pinky_contact = jp.maximum(
        (jp.abs(tip_finger_forces[3]) > self._config.reward_config.finger_active_threshold).astype(jp.float32),
        self._has_cube_contact(data, self._finger_contact_gids["pinky"]),
    )
    palm_only_contact = palm_contact * (1.0 - jp.clip(triad_count / 2.0, 0.0, 1.0))
    cheat_contact = jp.clip(jp.maximum(palm_only_contact, 0.6 * pinky_contact), 0.0, 1.0)
    clean_wrap_gate = jp.maximum(wrap_gate, 0.8 * palm_wrap_gate) * (1.0 - cheat_contact)
    clean_strong_wrap_gate = strong_wrap_gate * (1.0 - cheat_contact)
    pose_gate = jp.maximum(0.35, jp.maximum(near_gate, 0.65 * palm_contact + 0.35 * clean_wrap_gate))
    soft_margin = self._config.reward_config.finger_active_threshold * 0.75
    wrap_soft = jp.maximum(
        jp.clip((wrap_forces - soft_margin) / 0.05, 0.0, 1.0),
        wrap_active,
    )
    joint_band_soft = jp.clip(
        0.18 * wrap_soft[0] + 0.60 * wrap_soft[1] + 0.55 * wrap_soft[2],
        0.0,
        1.0,
    )
    joint_band_active = jp.clip(
        0.18 * wrap_active[0] + 0.60 * wrap_active[1] + 0.55 * wrap_active[2],
        0.0,
        1.0,
    )
    joint_palm_clamp = (
        palm_contact
        * (0.2 + 0.8 * joint_band_soft)
        * (0.25 + 0.75 * thumb_active)
        * (1.0 - palm_only_contact)
    )
    ulnar_wrap = (
        jp.minimum(wrap_soft[1], wrap_soft[2])
        * (0.25 + 0.75 * palm_contact)
        * (0.25 + 0.75 * thumb_active)
        * (1.0 - palm_only_contact)
    )
    core_cup_wrap = (
        jp.maximum(jp.minimum(wrap_soft[1], wrap_soft[2]), 0.85 * joint_band_soft)
        * (0.35 + 0.65 * wrap_soft[3])
        * (0.45 + 0.55 * palm_contact)
        * (1.0 - palm_only_contact)
    )
    hold_gate = jp.maximum(
        clean_strong_wrap_gate,
        jp.maximum(
            0.92 * joint_palm_clamp,
            jp.maximum(
                0.75 * core_cup_wrap * (1.0 - cheat_contact),
                0.70 * ulnar_wrap,
            ),
        ),
    )
    core_contact = (
        jp.maximum(jp.minimum(wrap_active[1], wrap_active[2]), 0.85 * joint_band_active)
        * (0.35 + 0.65 * wrap_active[3])
        * (0.45 + 0.55 * palm_contact)
        * (1.0 - palm_only_contact)
    )
    simultaneous_wrap = (
        jp.maximum(
            0.7 * core_cup_wrap,
            jp.maximum(jp.min(wrap_soft[:3]) * wrap_soft[3], 0.85 * joint_palm_clamp),
        )
        * (1.0 - cheat_contact)
        * (0.55 + 0.45 * palm_contact)
    )
    ring_pose = jp.clip(
        0.30 + (hand_q[4] - self._pre_grasp_pose[4]) / 0.16,
        0.0,
        1.0,
    )
    ring_proximity = jp.exp(-10.0 * tip_dists[2])
    ring_engage = (
        (0.65 * wrap_soft[2] + 0.25 * ring_proximity + 0.10 * ring_pose)
        * (0.35 + 0.65 * thumb_active)
        * (1.0 - palm_only_contact)
    )
    thumb_force_soft = jp.clip((wrap_forces[3] - 0.010) / 0.035, 0.0, 1.0)
    thumb_pose_under = jp.clip(
        (cube_pos[2] - tip_world[4, 2] + 0.028) / 0.070,
        0.0,
        1.0,
    )
    thumb_under_soft = jp.clip(
        0.55 * thumb_force_soft + 0.45 * thumb_pose_under,
        0.0,
        1.0,
    )
    z_keep = jp.clip((cube_pos[2] - (self._spawn_z - 0.012)) / 0.020, 0.0, 1.0)
    xy_keep = jp.exp(-10.0 * jp.linalg.norm((cube_pos - self._spawn_pos)[:2]))
    slow_keep = jp.exp(-3.5 * jp.linalg.norm(cube_linvel))
    release_height_retention = z_keep * xy_keep * slow_keep * released_gate * hold_gate
    core_forces = jp.array([wrap_forces[0], wrap_forces[1], wrap_forces[3]], dtype=jp.float32)
    core_force_mean = jp.mean(core_forces)
    core_force_level = jp.clip((core_force_mean - 0.018) / 0.11, 0.0, 1.0)
    core_force_balance = jp.clip(
        1.0 - jp.std(core_forces) / (core_force_mean + 1e-6),
        0.0,
        1.0,
    )
    post_release_force_support = (
        released_gate
        * z_keep
        * (0.35 + 0.65 * joint_palm_clamp)
        * core_force_level
        * (0.45 + 0.55 * core_force_balance)
    )
    post_release_joint_palm_hold = (
        released_gate
        * z_keep
        * xy_keep
        * slow_keep
        * joint_palm_clamp
        * (0.25 + 0.75 * ulnar_wrap)
        * (0.30 + 0.70 * ring_engage)
    )
    cradle_lock = (
        released_gate
        * z_keep
        * (0.35 + 0.65 * xy_keep)
        * (0.35 + 0.65 * slow_keep)
        * (0.25 + 0.75 * palm_contact)
        * (0.30 + 0.70 * thumb_under_soft)
        * (0.30 + 0.70 * joint_palm_clamp)
        * (0.20 + 0.80 * ulnar_wrap)
        * (0.25 + 0.75 * ring_engage)
        * (1.0 - 0.60 * cheat_contact)
    )
    core_force_soft = jp.clip((core_forces - 0.020) / 0.05, 0.0, 1.0)
    core_force_tripod = (
        jp.min(core_force_soft)
        * (0.45 + 0.55 * palm_contact)
        * (1.0 - cheat_contact)
    )

    return {
        "approach": self._reward_approach(mean_wrap_dist),
        "three_finger_proximity": self._reward_three_finger_proximity(max_wrap_dist),
        "contact": self._reward_contact(tip_world, cube_pos),
        "whole_hand_wrap": clean_wrap_gate * (0.4 + 0.6 * palm_wrap_gate),
        "simultaneous_wrap": simultaneous_wrap,
        "core_cup_wrap": core_cup_wrap,
        "core_contact": core_contact,
        "joint_palm_clamp": joint_palm_clamp,
        "core_force_tripod": core_force_tripod,
        "ring_engage": ring_engage,
        "ring_proximity": ring_proximity,
        "thumb_engage": self._reward_thumb_engage(tip_dists),
        "closure": self._reward_closure(hand_q) * pose_gate,
        "pip_closure": self._reward_pip_closure(hand_q) * pose_gate * (0.45 + 0.55 * mcp_gate),
        "human_pose": self._reward_human_pose(hand_q) * (0.5 + 0.5 * pose_gate),
        "grip_force": self._reward_grip_force(tip_finger_forces) * (1.0 - cheat_contact),
        "supported_hold_position": self._reward_hold_position(
            cube_pos, cube_linvel,
        ) * (1.0 - released_gate) * hold_gate,
        "short_hold_seed": jp.clip(
            info.get("stable_hold_steps", jp.array(0)).astype(jp.float32) / 80.0,
            0.0,
            1.0,
        ) * hold_gate,
        "release_height_retention": release_height_retention,
        "late_support_dependence": late_support_gate * (1.0 - released_gate) * hold_gate,
        "post_release_force_support": post_release_force_support,
        "post_release_joint_palm_hold": post_release_joint_palm_hold,
        "cradle_lock": cradle_lock,
        "hold_position": self._reward_hold_position(cube_pos, cube_linvel) * released_gate * hold_gate,
        "stable_hold": self._reward_stable_hold(
            cube_pos, cube_linvel, cube_angvel, tip_finger_forces,
        ) * released_gate * hold_gate,
        "force_contact": self._reward_force_contact(tip_finger_forces) * released_gate,
        "primary_geom_contact": self._reward_primary_geom_contact(tip_contact_flags),
        "progressive_hold": self._reward_progressive_hold(info) * hold_gate,
        "sustained_hold_bonus": self._reward_sustained_hold_bonus(info) * hold_gate,
        "force_balance": self._reward_force_balance(tip_finger_forces) * (1.0 - cheat_contact),
        "finger_participation": self._reward_finger_participation(tip_finger_forces) * (0.35 + 0.65 * clean_wrap_gate),
        "thumb_opposition": self._reward_thumb_opposition(
            tip_world, cube_pos, tip_finger_forces,
        ) * (0.35 + 0.65 * clean_wrap_gate),
        "primary_finger_force": self._reward_primary_finger_force(
            tip_finger_forces,
        ) * (0.35 + 0.65 * clean_wrap_gate),
        "pre_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * (1.0 - released_gate) * (0.35 + 0.65 * clean_wrap_gate),
        "post_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * released_gate * (0.35 + 0.65 * clean_wrap_gate),
        "post_release_survival": self._reward_post_release_survival(
            cube_pos, cube_linvel,
        ) * released_gate * hold_gate,
        "post_release_cheat_contact": released_gate * cheat_contact,
        "post_release_slip": self._cost_post_release_slip(
            cube_linvel,
        ) * released_gate * hold_gate,
        "post_release_pose_hold": self._reward_post_release_pose_hold(
            hand_q,
        ) * released_gate * hold_gate,
        "soft_contact": self._reward_soft_contact(tip_finger_forces),
        "idle_follow": self._reward_idle_follow(hand_q, tip_finger_forces),
        "height": self._reward_height(cube_pos, palm_pos) * clean_wrap_gate,
        "survival": 1.0 - done,
        "termination": done,
        "drop_risk": self._cost_drop_risk(cube_pos, cube_linvel),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "action_accel": self._cost_action_accel(action, info["last_act"], info["last_last_act"]),
        "torques": self._cost_torques(data.actuator_force),
        "force_overload": self._cost_force_overload(tip_finger_forces),
        "palm_contact": palm_only_contact * (0.5 + 0.5 * (1.0 - released_gate)),
        "nonprimary_contact": pinky_contact * (0.25 + 0.75 * clean_wrap_gate),
    }


domain_randomize = grasp_bottle_v2_force.domain_randomize
