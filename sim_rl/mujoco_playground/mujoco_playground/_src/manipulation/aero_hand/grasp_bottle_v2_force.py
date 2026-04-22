# Copyright 2025 TetherIA Inc.
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""V2 灵犀手 550ml 空瓶抓握任务."""

from typing import Any, Dict, Optional, Union

import jax
import numpy as np
from ml_collections import config_dict
from mujoco import mjx

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import grasp_cube_v2_force


def default_config() -> config_dict.ConfigDict:
  cfg = grasp_cube_v2_force.default_config()

  cfg.action_scale = [0.08, 0.18, 0.22, 0.22, 0.14, 0.14]

  cfg.support_config.release_after_sec = 2.6
  cfg.support_config.release_ramp_sec = 0.4
  cfg.support_config.force_release_after_sec = 3.6
  cfg.support_config.support_pos = [0.028, -0.090, 0.1385]
  cfg.support_config.random_release = True
  cfg.support_config.random_release_min_sec = 2.2
  cfg.support_config.random_release_max_sec = 3.2

  cfg.spawn_config.cube_pos = [0.028, -0.090, 0.175]
  cfg.spawn_config.cube_jitter = [0.0015, 0.0015, 0.002]

  cfg.reset_config.pre_grasp_fraction = 0.45
  cfg.reset_config.pre_grasp_noise_scale = 0.08
  cfg.reset_config.lifted_cube_z_offset = 0.012

  cfg.perturbation_config.external_force_magnitude = 0.05
  cfg.perturbation_config.gravity_tilt_max_rad = 0.25
  cfg.perturbation_config.orientation_flip_force_scale = 0.85
  cfg.perturbation_config.orientation_flip_min_hold_steps = 320

  scales = cfg.reward_config.scales
  scales.hold_position = 40.0
  scales.stable_hold = 150.0
  scales.progressive_hold = 45.0
  scales.sustained_hold_bonus = 65.0
  scales.force_contact = 18.0
  scales.grip_force = 12.0
  scales.force_balance = 22.0
  scales.finger_participation = 18.0
  scales.thumb_opposition = 24.0
  scales.soft_contact = 8.0
  scales.primary_finger_force = 48.0
  scales.post_release_grasp = 110.0
  scales.post_release_survival = 135.0
  scales.post_release_cheat_contact = -8.0
  scales.post_release_slip = -55.0
  scales.post_release_pose_hold = 55.0
  scales.height = 14.0
  scales.termination = -420.0
  scales.drop_risk = -55.0
  scales.palm_contact = -3.0
  scales.nonprimary_contact = -2.0
  scales.three_finger_proximity = 10.0
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

  def _reward_contact(self, tip_world, cube_pos):
    del cube_pos
    target = self._spawn_pos
    body_half_extents = np.array([0.040, 0.045, 0.105], dtype=np.float32)
    diff = jax.numpy.abs(tip_world - target[None, :]) - body_half_extents
    surface_dists = jax.numpy.linalg.norm(jax.numpy.maximum(diff, 0.0), axis=1)
    return jax.numpy.mean(jax.numpy.exp(-4.0 * surface_dists))


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
