# Copyright 2025 TetherIA Inc.
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
#更改rotate_z为grasp_cube
"""Grasp a cube with TetherIA Aero Hand Open."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import base as aero_hand_base


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      action_scale=[0.02, 0.02, 0.02, 0.02, 0.7, 0.003, 0.012],
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=1,
      noise_config=config_dict.create(
          level=1.0,
          scales=config_dict.create(
              joint_pos=0.05,
              tendon_length=0.005,
          ),
      ),
      support_config=config_dict.create(
          # Automatically remove support after this many seconds.
          release_after_sec=1.2,
          # Support start pose (matches scene_mjx_grasp.xml).
          support_pos=[-0.066, 0.0, 0.067],
          # Move support far below scene once released.
          support_hidden_pos=[0.0, 0.0, -10.0],
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              height=4.0,
              approach=4.0,  # 单指接近（保留但降权）
              contact=1.5,  # 近接触阈值奖励
              multi_finger=3.0,  # 多指同时接近奖励
              thumb_engage=2.5,  # 拇指参与奖励
              closure=1.2,  # 四指+拇指屈曲协同奖励
              survival=0.05,
              termination=-150.0,
              action_rate=-0.02,
              torques=-0.001,
          ),
      ),
  )


class CubeGrasp(aero_hand_base.AeroHandEnv):
  """Grasp a cube vertically and hold it against gravity."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.GRASP_XML.as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)

    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)
    self._cube_qids = mjx_env.get_qpos_ids(self.mj_model, ["cube_freejoint"])
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._cube_geom_id = self._mj_model.geom("cube").id

    self._support_body_id = self._mj_model.body("cube_support").id
    self._support_mocap_id = int(
        self._mj_model.body_mocapid[self._support_body_id]
    )
    if self._support_mocap_id < 0:
      raise ValueError("cube_support must be a mocap body.")

    self._support_pos = jp.array(self._config.support_config.support_pos)
    self._support_hidden_pos = jp.array(
        self._config.support_config.support_hidden_pos
    )
    self._support_release_steps = max(
        1,
        int(np.round(self._config.support_config.release_after_sec / self.dt)),
    )

    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    self._default_pose = self._init_q[self._hand_qids]
    self._lowers, self._uppers = self.mj_model.jnt_range[self._hand_qids].T

    self._init_tendon = jp.array(home_key.ctrl)
    self._default_tendon = self._init_tendon

  def reset(self, rng: jax.Array) -> mjx_env.State:
    # Randomize hand qpos and qvel.
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    q_hand = jp.clip(
        self._default_pose + 0.1 * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    # Randomize cube qpos and qvel; start resting on support pedestal.
    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    start_pos = jp.array([-0.066, 0.0, 0.085]) + jax.random.uniform(
        p_rng, (3,), minval=jp.array([-0.004, -0.004, -0.001]),
        maxval=jp.array([0.004, 0.004, 0.001])
    )
    del quat_rng  # Keep cube orientation stable at reset.
    start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    mocap_pos = self._initial_mocap_pos()
    mocap_pos = mocap_pos.at[self._support_mocap_id].set(self._support_pos)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._default_tendon,  # Change: only use the control tendons
        mocap_pos=mocap_pos,
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_cube_angvel": jp.zeros(3),
        "support_released": jp.array(False),
        "support_timer": jp.array(0, dtype=jp.int32),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # Change: 14 is the sum of the number of the tendon/joint sensors (7) and the number of the control actions (7)
    obs_history = jp.zeros(self._config.history_len * 14)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)  # pylint: disable=redefined-outer-name
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:

    action_scale_custom = jp.array(self._config.action_scale, dtype=jp.float32)
    motor_targets = self._default_tendon + action * action_scale_custom
    # NOTE: no clipping.
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    support_timer = state.info["support_timer"] + 1
    support_released = self._should_release_support(
        state.info["support_released"], support_timer
    )
    data = self._set_support_state(data, support_released)

    state.info["motor_targets"] = motor_targets
    state.info["support_released"] = support_released
    state.info["support_timer"] = support_timer

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt  # pylint: disable=redefined-outer-name

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_cube_angvel"] = self.get_cube_angvel(data)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state

  def _initial_mocap_pos(self) -> jax.Array:
      if self.mj_model.nmocap == 0:
          return jp.zeros((0, 3), dtype=jp.float32)
      mocap_pos = np.zeros((self.mj_model.nmocap, 3), dtype=np.float32)
      for body_id in range(self.mj_model.nbody):
          mocap_id = int(self.mj_model.body_mocapid[body_id])
          if mocap_id >= 0:
              mocap_pos[mocap_id] = self.mj_model.body_pos[body_id]
      return jp.array(mocap_pos)

  def _should_release_support(
      self, already_released: jax.Array, support_timer: jax.Array
  ) -> jax.Array:
      timed_release = support_timer >= self._support_release_steps
      return jp.logical_or(already_released, timed_release)

  def _set_support_state(
          self, data: mjx.Data, support_released: jax.Array
  ) -> mjx.Data:
      if self.mj_model.nmocap == 0:
          return data

      support_pos = jp.where(
              support_released,
              self._support_hidden_pos,
              self._support_pos,
      )
      mocap_pos = data.mocap_pos.at[self._support_mocap_id].set(support_pos)
      return data.replace(mocap_pos=mocap_pos)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    fall_termination = self.get_cube_position(data)[2] < -0.05
    return fall_termination

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:

    info["rng"], noise_rng = jax.random.split(info["rng"])

    # ------- tendon length sensor -------
    tendon_lengths = jp.zeros(
        (len(consts.SENSOR_TENDON_NAMES),), dtype=jp.float32
    )
    for idx, name in enumerate(consts.SENSOR_TENDON_NAMES):
      v = mjx_env.get_sensor_data(self.mj_model, data, name)
      v = jp.ravel(v)[0]
      tendon_lengths = tendon_lengths.at[idx].set(v)

    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_tendon_lengths = (
        tendon_lengths
        + (2 * jax.random.uniform(noise_rng, shape=tendon_lengths.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.tendon_length
    )

    # ------- joint angle sensor -------
    joint_angles = jp.zeros((len(consts.SENSOR_JOINT_NAMES),), dtype=jp.float32)
    for idx, name in enumerate(consts.SENSOR_JOINT_NAMES):
      v = mjx_env.get_sensor_data(self.mj_model, data, name)
      v = jp.ravel(v)[0]
      joint_angles = joint_angles.at[idx].set(v)

    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    state = jp.concatenate([
        noisy_tendon_lengths,
        noisy_joint_angles,
        info["last_act"],
    ])

    joint_angles = data.qpos[self._hand_qids]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_quat = self.get_cube_orientation(data)
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    fingertip_positions = self.get_fingertip_positions(data)
    joint_torques = data.actuator_force

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        joint_torques,
        fingertip_positions,
        cube_pos_error,
        cube_quat,
        cube_angvel,
        cube_linvel,
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    tip_world = self.get_fingertip_positions(data).reshape(5, 3) + palm_pos
    tip_dists = jp.linalg.norm(tip_world - cube_pos[None, :], axis=1)
    min_tip_dist = jp.min(tip_dists)
    hand_q = data.qpos[self._hand_qids]
    return {
        "height": self._reward_height(cube_pos, palm_pos),
        "approach": self._reward_approach(min_tip_dist),
        "contact": self._reward_contact(min_tip_dist),
        "multi_finger": self._reward_multi_finger(tip_dists),
        "thumb_engage": self._reward_thumb_engage(tip_dists),
        "closure": self._reward_closure(hand_q),
        "survival": self._reward_survival(done),
        "termination": done,
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        "torques": self._cost_torques(data.actuator_force),
    }

  def _reward_height(
      self, cube_pos: jax.Array, palm_pos: jax.Array
  ) -> jax.Array:
    # Encourage lifted cube that stays close to the palm in horizontal plane.
    lift = jp.clip((cube_pos[2] - 0.05) / 0.08, 0.0, 1.0)
    xy_err = jp.linalg.norm((cube_pos - palm_pos)[:2])
    palm_alignment = jp.exp(-40.0 * jp.square(xy_err))
    return lift * palm_alignment

  def _reward_survival(self, done: jax.Array) -> jax.Array:
    return 1.0 - done

  def _reward_approach(self, min_tip_dist: jax.Array) -> jax.Array:
    # Dense shaping: encourage fingertips moving close to the cube.
    return jp.exp(-30.0 * min_tip_dist)

  def _reward_contact(self, min_tip_dist: jax.Array) -> jax.Array:
    # Sparse-ish bonus after reaching near-contact range.
    return (min_tip_dist < 0.02).astype(jp.float32)

  def _reward_multi_finger(self, tip_dists: jax.Array) -> jax.Array:
    # Encourage multiple fingertips (not only one) to approach the cube.
    close_count = jp.sum((tip_dists < 0.03).astype(jp.float32))
    return close_count / 5.0

  def _reward_thumb_engage(self, tip_dists: jax.Array) -> jax.Array:
    # Fingertip order follows consts.FINGERTIP_NAMES: if,mf,rf,pf,th.
    thumb_dist = tip_dists[4]
    return jp.exp(-35.0 * thumb_dist)

  def _reward_closure(self, hand_q: jax.Array) -> jax.Array:
    # Encourage all fingers to participate instead of two-finger-only grasps.
    # hand_q order: 12 finger joints + thumb_abd + thumb flex/mcp/ip.
    finger_mcp = jp.take(hand_q, jp.array([0, 3, 6, 9]))
    finger_close = jp.mean(jp.clip(finger_mcp / 1.2, 0.0, 1.0))
    thumb_close = jp.mean(jp.clip(hand_q[13:16] / 1.0, 0.0, 1.0))
    return 0.6 * finger_close + 0.4 * thumb_close

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(
        jp.abs(qvel) * jp.abs(qfrc_actuator)
    )  # Change: only use the control joints

  def _cost_linvel(self, cube_linvel: jax.Array) -> jax.Array:
    return jp.linalg.norm(cube_linvel, ord=1, axis=-1)

  def _reward_angvel(
      self, cube_angvel: jax.Array, cube_pos_error: jax.Array
  ) -> jax.Array:
    # Unconditionally maximize angvel in the z-direction.
    del cube_pos_error  # Unused.
    return cube_angvel @ jp.array([0.0, 0.0, 1.0])

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_pose(self, joint_angles: jax.Array) -> jax.Array:
    return jp.sum(jp.square(joint_angles - self._default_pose))


def domain_randomize(model: mjx.Model, rng: jax.Array):
    mj_model = CubeGrasp().mj_model
    cube_geom_id = mj_model.geom("cube").id
    cube_body_id = mj_model.body("cube").id
    hand_qids = mjx_env.get_qpos_ids(mj_model, consts.JOINT_NAMES)
    hand_body_names = [
        "palm",
        "right_index_f_link",
        "right_index_proximal_link",
        "right_index_middle_link",
        "right_index_distal_link",
        "right_middle_f_link",
        "right_middle_proximal_link",
        "right_middle_middle_link",
        "right_middle_distal_link",
        "right_ring_f_link",
        "right_ring_proximal_link",
        "right_ring_middle_link",
        "right_ring_distal_link",
        "right_pinky_f_link",
        "right_pinky_proximal_link",
        "right_pinky_middle_link",
        "right_pinky_distal_link",
        "right_t_link",
        "right_thumb_mcp_link",
        "right_thumb_proximal_link",
        "right_thumb_distal_link",
    ]
    hand_body_ids = np.array([mj_model.body(n).id for n in hand_body_names])
    fingertip_geoms = ["if_tip", "mf_tip", "rf_tip", "pf_tip", "th_tip"]
    fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

    @jax.vmap
    def rand(rng):
      # Cube friction: =U(0.1, 0.5).
      rng, key = jax.random.split(rng)
      cube_friction = jax.random.uniform(key, (1,), minval=0.1, maxval=0.5)
      geom_friction = model.geom_friction.at[
          cube_geom_id : cube_geom_id + 1, 0
      ].set(cube_friction)

      # Fingertip friction: =U(0.5, 1.0).
      fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
      geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(
          fingertip_friction
      )

      # Scale cube mass: *U(0.8, 1.2).
      rng, key1, key2 = jax.random.split(rng, 3)
      dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
      cube_mass = model.body_mass[cube_body_id]
      body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
      body_inertia = model.body_inertia.at[cube_body_id].set(
          model.body_inertia[cube_body_id] * dmass
      )
      dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
      body_ipos = model.body_ipos.at[cube_body_id].set(
          model.body_ipos[cube_body_id] + dpos
      )

      # Jitter qpos0: +U(-0.05, 0.05).
      rng, key = jax.random.split(rng)
      qpos0 = model.qpos0
      qpos0 = qpos0.at[hand_qids].set(
          qpos0[hand_qids]
          + jax.random.uniform(key, shape=(16,), minval=-0.05, maxval=0.05)
      )

      # Scale static friction: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
          key, shape=(16,), minval=0.5, maxval=2.0
      )
      dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

      # Scale armature: *U(1.0, 1.05).
      rng, key = jax.random.split(rng)
      armature = model.dof_armature[hand_qids] * jax.random.uniform(
          key, shape=(16,), minval=1.0, maxval=1.05
      )
      dof_armature = model.dof_armature.at[hand_qids].set(armature)

      # Scale all link masses: *U(0.9, 1.1).
      rng, key = jax.random.split(rng)
      dmass = jax.random.uniform(
          key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1
      )
      body_mass = model.body_mass.at[hand_body_ids].set(
          model.body_mass[hand_body_ids] * dmass
      )

      # Joint stiffness: *U(0.8, 1.2).
      rng, key = jax.random.split(rng)
      kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
          key, (model.nu,), minval=0.8, maxval=1.2
      )
      actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
      actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

      # Joint damping: *U(0.8, 1.2).
      rng, key = jax.random.split(rng)
      kd = model.dof_damping[hand_qids] * jax.random.uniform(
          key, (16,), minval=0.8, maxval=1.2
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
