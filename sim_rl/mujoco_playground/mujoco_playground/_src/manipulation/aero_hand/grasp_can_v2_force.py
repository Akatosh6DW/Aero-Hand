# Copyright 2025 TetherIA Inc.
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""V2 灵犀手小圆柱/易拉罐代理抓握任务."""

from typing import Any, Dict, Optional, Union

import jax.numpy as jp
import numpy as np
from ml_collections import config_dict

from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import grasp_bottle_v2_force


def default_config() -> config_dict.ConfigDict:
  cfg = grasp_bottle_v2_force.default_config()

  # 当前目标改为更小的圆柱代理，控制和接触阈值再收紧一档。
  cfg.action_scale = [0.07, 0.16, 0.18, 0.18, 0.12, 0.12]

  cfg.support_config.release_after_sec = 3.8
  cfg.support_config.release_ramp_sec = 0.5
  cfg.support_config.force_release_after_sec = 5.2
  cfg.support_config.support_pos = [0.013113, -0.040712, 0.108458]
  cfg.support_config.random_release = True
  cfg.support_config.random_release_min_sec = 3.5
  cfg.support_config.random_release_max_sec = 4.6
  cfg.support_config.min_release_active_fingers = 3
  cfg.support_config.min_release_force = 0.15

  cfg.spawn_config.cube_pos = [0.013113, -0.040712, 0.132458]
  cfg.spawn_config.cube_jitter = [0.0010, 0.0010, 0.0015]
  cfg.spawn_config.support_enabled = False

  cfg.reset_config.pre_grasp_fraction = 0.0
  cfg.reset_config.pre_grasp_noise_scale = 0.04
  cfg.reset_config.lifted_grasp_fraction = 1.0
  cfg.reset_config.lifted_grasp_noise_scale = 0.02
  cfg.reset_config.lifted_cube_z_offset = 0.018

  cfg.perturbation_config.external_force_magnitude = 0.03
  cfg.perturbation_config.gravity_tilt_max_rad = 0.20
  cfg.perturbation_config.orientation_flip_force_scale = 0.70
  cfg.perturbation_config.orientation_flip_min_hold_steps = 360

  scales = cfg.reward_config.scales
  scales.hold_position = 58.0
  scales.stable_hold = 230.0
  scales.progressive_hold = 85.0
  scales.sustained_hold_bonus = 110.0
  scales.force_contact = 16.0
  scales.grip_force = 10.0
  scales.force_balance = 24.0
  scales.finger_participation = 28.0
  scales.thumb_opposition = 38.0
  scales.soft_contact = 12.0
  scales.primary_finger_force = 54.0
  scales.pre_release_grasp = 62.0
  scales.post_release_grasp = 150.0
  scales.post_release_survival = 195.0
  scales.post_release_cheat_contact = 0.0
  scales.post_release_slip = -72.0
  scales.post_release_pose_hold = 95.0
  scales.height = 12.0
  scales.termination = -460.0
  scales.drop_risk = -62.0
  scales.palm_contact = 1.0
  scales.nonprimary_contact = 0.8
  scales.three_finger_proximity = 22.0
  scales.force_overload = -10.0

  cfg.reward_config.force_contact_threshold = 0.03
  cfg.reward_config.force_contact_saturation = 1.2
  cfg.reward_config.force_overload_threshold = 1.25
  cfg.reward_config.force_overload_soft_width = 0.35
  cfg.reward_config.soft_contact_fmin = 0.05
  cfg.reward_config.soft_contact_fmax = 1.10
  cfg.reward_config.finger_active_threshold = 0.04
  cfg.reward_config.target_lift_m = 0.012
  cfg.reward_config.lift_success_threshold_m = 0.008
  cfg.stability_config.max_abs_action = 0.55
  cfg.stability_config.motor_delta_clip = [0.02, 0.04, 0.045, 0.045, 0.04, 0.04]
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
    self._lifted_grasp_ctrl = np.array([
        1.18, 0.18, 0.3400, 0.3520, 0.3380, 0.3000,
    ], dtype=np.float32)

  def _reward_contact(self, tip_world, cube_pos):
    del cube_pos
    target = self._spawn_pos
    body_half_extents = jp.array([0.034, 0.024, 0.024], dtype=jp.float32)
    diff = jp.abs(tip_world - target[None, :]) - body_half_extents
    surface_dists = jp.linalg.norm(jp.maximum(diff, 0.0), axis=1)
    return jp.mean(jp.exp(-4.8 * surface_dists))


domain_randomize = grasp_bottle_v2_force.domain_randomize
