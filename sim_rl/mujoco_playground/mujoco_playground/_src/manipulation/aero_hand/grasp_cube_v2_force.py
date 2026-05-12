# Copyright 2025 TetherIA Inc.
# Licensed under the Apache License, Version 2.0
# ==============================================================================
"""V2 灵犀手 force-aware cube grasp task.

灵犀手 V2 硬件特征：
  - 11 关节 (4指各 MCP+PIP, 拇指 CMC_ABD+CMC_FLEX+MCP)
  - 6 通道直接关节位置控制 (非腱绳)
  - PIP = 0.925 * MCP 等式约束
  - 拇指 CMC_FLEX = 0.16 * CMC_ABD 等式约束
  - 80 路触觉传感器 (5指 × 16 taxels)
"""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
import mujoco
from mujoco import mjx
import numpy as np

from mujoco_playground._src import mjx_env
from mujoco_playground._src.manipulation.aero_hand import aero_hand_constants as consts
from mujoco_playground._src.manipulation.aero_hand import base as aero_hand_base


# ── V2 关节索引常量 ──────────────────────────────────────────────────────────
# qpos 顺序（由 XML body 树深度优先遍历决定）：
#   index_mcp=0, index_pip=1,
#   middle_mcp=2, middle_pip=3,
#   ring_mcp=4, ring_pip=5,
#   pinky_mcp=6, pinky_pip=7,
#   thumb_cmc_abd=8, thumb_cmc_flex=9, thumb_mcp=10
_V2_FINGER_MCP_IDS = [0, 2, 4, 6]   # index, middle, ring, pinky
_V2_FINGER_PIP_IDS = [1, 3, 5, 7]
_V2_THUMB_ABD_ID = 8
_V2_THUMB_FLEX_ID = 9
_V2_THUMB_MCP_ID = 10

_DIAGNOSTIC_METRIC_KEYS = (
    "diagnostic/success",
    "diagnostic/three_finger_contact",
    "diagnostic/two_plus_primary_contact",
    "diagnostic/contact_duration_sec",
    "diagnostic/lift_success",
    "diagnostic/hold_success",
    "diagnostic/drop",
    "diagnostic/normal_force_mean",
    "diagnostic/tangent_force_approx",
    "diagnostic/slip_event",
    "diagnostic/primary_active_count",
    "diagnostic/primary_any_active_count",
    "diagnostic/non_tip_primary_contact",
    "diagnostic/nonprimary_contact",
    "diagnostic/palm_contact",
    "diagnostic/support_released",
    "diagnostic/nonfinite_state",
    "diagnostic/lifted_reset",
    "diagnostic/lift_height",
    "termination/drop",
)

# 执行器顺序: [thumb_rot, thumb_flex, index, middle, ring, pinky]
# 共 6 通道，与 V1 HW6 对齐。


def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.05,
      sim_dt=0.01,
      # V2 直接关节位置控制：action_scale 为关节角度增量。
      # PPO tanh_normal action ∈ (-1, +1)。
      # 每通道 act=1.0 可将 ctrl 从 0 推到接近上限。
      # [thumb_rot, thumb_flex, index, middle, ring, pinky]
      # C03: new pinch home is already near contact; restart with local residual
      # actions so random exploration does not destroy the feasible geometry.
      action_scale=[0.12, 0.35, 0.30, 0.30, 0.02, 0.02],  # C21: ring/pinky 0.08→0.02 near-lock
      action_repeat=1,
      episode_length=800,
      early_termination=True,
      history_len=1,
      force_history_len=1,
      noise_config=config_dict.create(
          level=0.3,
          force_ema_alpha=0.85,
          scales=config_dict.create(
              hw_pos=0.005,
              hw_force=0.05,
          ),
      ),
      tactile_config=config_dict.create(
          use_pooled_obs=True,
          taxel_weights=[
              0.7, 1.0, 1.0, 0.7,
              1.0, 1.4, 1.4, 1.0,
              1.0, 1.4, 1.4, 1.0,
              0.7, 1.0, 1.0, 0.7,
          ],
          use_real_tactile=True,
          force_saturation_n=3.0,
          obs_force_ema_alpha=0.7,
      ),
      stability_config=config_dict.create(
          max_abs_action=1.0,
          motor_delta_clip=[0.03, 0.03, 0.03, 0.03, 0.03, 0.03],  # C41: smooth per-step target jumps to reduce DR-induced late slip
          terminate_on_nonfinite=True,
          nonfinite_penalty_mult=2.0,
      ),
      support_config=config_dict.create(
          release_after_sec=2.0,              # C15: early release for more post-release training
          release_ramp_sec=0.5,               # C15: faster ramp (0.5s instead of 1.0s)
          support_pos=[0.021, -0.065, 0.1308],      # C156: continue x scan 2mm farther toward old cube line
          support_hidden_pos=[0.0, 0.0, -10.0],
          min_release_active_fingers=2,   # C12: easier conditional release
          min_release_force=0.10,         # C136: restore C124 gate while validating smoothed thumb collision
          require_grasp_for_release=True,   # C05: release when primary force/geom contact is ready
          force_release_after_sec=3.0,      # C15: force release at 3s (was 8s)
          # R66: 随机支撑释放 (1.5-4.0s)
          random_release=True,
          random_release_min_sec=1.7,       # C36: broaden release timing without dropping too early
          random_release_max_sec=3.0,       # C36: expose later support removal after C35 recovery
      ),
      spawn_config=config_dict.create(
          cube_pos=[0.021, -0.065, 0.1463],           # C156: keep support/cube pair aligned after x -4mm scan
          cube_jitter=[0.0, 0.0, 0.0],
          support_enabled=True,
      ),
      reset_config=config_dict.create(
          hand_qpos_noise_scale=0.01,     # R99: start close to desired hand shape
          pre_grasp_fraction=0.0,         # R99: home keyframe already is the desired pregrasp
          pre_grasp_noise_scale=0.12,     # C36: tighten geometry slightly while lifted curriculum grows
          lifted_grasp_fraction=0.12,     # C36: raise unsupported-start exposure after C35 stayed stable
          lifted_grasp_noise_scale=0.04,  # C35: keep lifted starts tidy; avoid destabilizing a strong 30s policy
          lifted_cube_z_offset=0.012,     # C35: slightly more clearance on lifted starts
          termination_grace_sec=5.0,       # C03: keep early pinch exploration from truncating immediately
      ),
      # C26: 中等扰动配置 — C25证明强扰动(0.15N)训练>14M步会退化
      perturbation_config=config_dict.create(
          # 外力脉冲: 支撑释放后每隔interval步施加一次随机力
          external_force_enabled=True,
          external_force_magnitude=0.13,   # C38: slightly harder pulse stress after C37 stabilized early flip/tilt
          external_force_interval=30,      # C31: 更频繁外力脉冲
          external_force_min_hold_steps=60,  # C38: bring pulses in a bit earlier
          # 重力方向扰动: 模拟手腕姿态变化(倾斜/翻转)
          gravity_perturbation_enabled=True,
          gravity_tilt_max_rad=0.45,        # C31: 约26度, 介于C26与强扰动之间
          gravity_tilt_change_interval=60,  # C31: 更频繁倾斜变化
          gravity_tilt_min_hold_steps=120,  # C37: bring tilt earlier once C35/C36 proved stable
          # C28: 全向翻转等效扰动。只在已形成较长稳定持握后触发,
          # 用随机3D等效重力检验手腕翻转/倒置时的力闭合稳定度。
          orientation_flip_enabled=True,
          orientation_flip_force_scale=1.25,  # C31: 1.25 * mg, 更接近翻转/倒置应力
          orientation_flip_change_interval=80,  # C31: 4s保持一次翻转方向
          orientation_flip_min_hold_steps=260,  # C37: start flip stress earlier for 30s hold hardening
          total_force_clip_n=0.35,           # C32: clip combined perturbation force to avoid MJX blow-up
          # 关节角度观测噪声
          joint_obs_noise_enabled=True,
          joint_obs_noise_std=0.015,        # C26: 0.02→0.015rad, 中等
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              # ── R78: 强制主要手指参与 ──
              # R78核心: hold类奖励乘以primary_gate, grip_force仅计三指
              # primary_finger_force软化公式, thumb_opposition软化门控
              # R81: 力封闭优先(Rajeswaran 2017) — 降低运动学约束, 提升力信号
              hold_position=45.0,          # R104: timed release needs stronger unsupported pose retention
              stable_hold=185.0,           # C30: push 27s holds over the 30s target
              progressive_hold=55.0,       # C30: keep dense gradient near 20-30s
              sustained_hold_bonus=85.0,  # C30: stronger 30s milestone pressure
              force_contact=15.0,
              primary_geom_contact=25.0,   # C04: thumb contact can be geometric before force loads
              approach=12.0,               # R91: 恢复R89值(R90减弱导致bootstrapping失败)
              contact=4.0,                # R91: 恢复R89值
              thumb_engage=20.0,           # R91: 恢复R89值
              closure=6.0,                 # R81: 12→6, 不强制关节角度
              pip_closure=3.0,             # R81: 5→3, 同上
              human_pose=2.0,              # R81: 3→2, 同上
              grip_force=15.0,
              force_balance=28.0,           # C29: improve primary force closure under DR
              finger_participation=5.0,
              thumb_opposition=30.0,
              soft_contact=5.0,
              primary_finger_force=70.0,  # C39: firmer primary triad before external pulses land
              pre_release_grasp=40.0,      # C39: strengthen release-ready shaping
              post_release_grasp=125.0,    # C29: rebuild active pinch after support release
              post_release_survival=155.0,  # C30: reduce remaining 18% drop rate
              post_release_cheat_contact=-70.0,  # R105: suppress palm/ring/pinky support after release
              post_release_slip=-55.0,     # C35: stronger punishment for late DR-induced slip
              post_release_pose_hold=70.0, # C35: reinforce clean unsupported geometry through 30s+
              idle_follow=0.0,
              height=18.0,                 # R92: 5→18, 明确鼓励离开支撑面的主动抬升
              # 控制约束 (不门控)
              survival=1.0,
              termination=-380.0,          # C30: remaining failures are mostly drops
              action_rate=-0.02,           # R91: -0.01→-0.02, 增加动作平滑防止崩溃
              action_accel=-0.005,
              torques=-0.00003,
              force_overload=0.0,
              drop_risk=-55.0,             # C35: stronger pre-drop shaping under flip/tilt perturbations
              palm_contact=-28.0,          # C29: keep cleanup, avoid over-suppressing recovery
              nonprimary_contact=-20.0,    # C29: keep cleanup, slightly relax C28 harshness
              three_finger_proximity=15.0, # R91: 恢复R89值
          ),
          force_contact_threshold=0.06,
          force_contact_saturation=3.0,
          force_overload_threshold=2.8,
          force_overload_soft_width=1.5,
          soft_contact_fmin=0.1,
          soft_contact_fmax=2.5,
          finger_active_threshold=0.08,
          target_lift_m=0.012,             # R93: 几何扫描显示+20mm不可达, 目标改为12mm
          lift_success_threshold_m=0.010,  # R93: 成功阈值改为可达的10mm离台高度
          hold_success_sec=30.0,           # R103: target is sustained 30s hold after support release
      ),
  )


def default_config_qbr() -> config_dict.ConfigDict:
  """Current-hand cube config tuned for qbr thumb coupling under DR."""
  cfg = default_config()
  scales = cfg.reward_config.scales
  # C50: C48 already reaches full-episode survival under DR. The remaining gap
  # is that the primary-contact diagnostic is slightly too strict after the
  # hand-parameter update, so keep the stable C48 shaping and recalibrate the
  # force-contact thresholds a notch lower.
  scales.stable_hold = 185.0
  scales.progressive_hold = 55.0
  scales.post_release_survival = 155.0
  scales.post_release_pose_hold = 50.0
  scales.hold_position = 45.0
  scales.drop_risk = -45.0
  scales.palm_contact = -28.0
  scales.nonprimary_contact = -20.0
  scales.pre_release_grasp = 35.0
  scales.post_release_grasp = 125.0
  scales.force_balance = 28.0
  scales.primary_finger_force = 65.0
  scales.thumb_opposition = 30.0
  scales.three_finger_proximity = 15.0

  # Keep the same DR envelope that C47/C48 proved learnable.
  pcfg = cfg.perturbation_config
  pcfg.external_force_magnitude = 0.10
  pcfg.external_force_interval = 35
  pcfg.external_force_min_hold_steps = 90
  pcfg.gravity_tilt_max_rad = 0.40
  pcfg.gravity_tilt_change_interval = 70
  pcfg.gravity_tilt_min_hold_steps = 170
  pcfg.orientation_flip_force_scale = 1.00
  pcfg.orientation_flip_change_interval = 120
  pcfg.orientation_flip_min_hold_steps = 400

  cfg.reset_config.pre_grasp_noise_scale = 0.10  # C146: validate best reset-noise point for smoothed thumb collision
  cfg.reset_config.lifted_grasp_fraction = 0.0
  cfg.reset_config.lifted_grasp_noise_scale = 0.06
  cfg.reset_config.lifted_cube_z_offset = 0.01

  cfg.support_config.random_release_min_sec = 1.7
  cfg.support_config.random_release_max_sec = 2.7

  cfg.reward_config.force_contact_threshold = 0.06
  cfg.reward_config.finger_active_threshold = 0.08
  cfg.reward_config.soft_contact_fmin = 0.1
  return cfg


class CubeGraspV2Force(aero_hand_base.AeroHandEnv):
  """V2 灵犀手方块抓握任务 (force-aware, 6 通道直接关节控制)。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    selected_xml = xml_path or consts.GRASP_V2_XML.as_posix()
    super().__init__(
        xml_path=selected_xml,
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.V2_JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.V2_JOINT_NAMES)

    # 支撑台
    self._support_body_id = self._mj_model.body("cube_support").id
    self._support_mocap_id = int(self._mj_model.body_mocapid[self._support_body_id])
    if self._support_mocap_id < 0:
      raise ValueError("cube_support must be a mocap body.")

    self._support_pos = jp.array(self._config.support_config.support_pos)
    self._support_hidden_pos = jp.array(self._config.support_config.support_hidden_pos)
    self._support_release_steps = max(
        1, int(np.round(self._config.support_config.release_after_sec / self.dt)),
    )
    force_release_sec = getattr(
        self._config.support_config, 'force_release_after_sec', 0.0,
    )
    self._force_release_steps = (
        max(1, int(np.round(force_release_sec / self.dt)))
        if force_release_sec > 0.0 else 0
    )
    ramp_sec = getattr(self._config.support_config, 'release_ramp_sec', 0.0)
    self._support_ramp_steps = max(
        0, int(np.round(ramp_sec / self.dt)),
    )
    self._termination_grace_steps = max(
        0, int(np.round(
            getattr(self._config.reset_config, 'termination_grace_sec', 0.0) / self.dt,
        )),
    )

    # R66: 随机支撑释放参数
    self._random_release = getattr(self._config.support_config, 'random_release', False)
    self._random_release_min_steps = max(
        1, int(np.round(
            getattr(self._config.support_config, 'random_release_min_sec', 1.5) / self.dt)),
    )
    self._random_release_max_steps = max(
        1, int(np.round(
            getattr(self._config.support_config, 'random_release_max_sec', 4.0) / self.dt)),
    )

    # 初始姿态
    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    home_hand_pose = self._init_q[self._hand_qids]
    lowers_np, uppers_np = self.mj_model.jnt_range[self._hand_qids].T
    self._lowers = jp.array(lowers_np, dtype=jp.float32)
    self._uppers = jp.array(uppers_np, dtype=jp.float32)
    self._default_pose = home_hand_pose  # from XML home keyframe

    # 默认控制：全部为 0（张开）
    self._default_ctrl = jp.array(home_key.ctrl, dtype=jp.float32)
    ctrl_lowers_np, ctrl_uppers_np = self.mj_model.actuator_ctrlrange.T
    self._ctrl_lowers = jp.array(ctrl_lowers_np, dtype=jp.float32)
    self._ctrl_uppers = jp.array(ctrl_uppers_np, dtype=jp.float32)

    # 生成位置
    self._spawn_z = jp.array(self._config.spawn_config.cube_pos[2], dtype=jp.float32)
    self._spawn_pos = jp.array(self._config.spawn_config.cube_pos, dtype=jp.float32)

    # Passive coupling defaults for the legacy cube hand model.
    self._finger_pip_coupling = 0.925
    self._thumb_passive_source_id = _V2_THUMB_ABD_ID
    self._thumb_passive_ratio = 0.16

    # 触觉权重
    self._taxel_weights = self._build_taxel_weights()

    # C02: 用户指定姿态的几何可行化版本。
    # 原始粗略目标 [1.38, 0.0, 0.55, 0.55, 1.4, 1.4] 会让
    # index/middle 指尖高出 cube 约38-40mm; 调到 thumb_flex=0.35,
    # index/middle=0.9 后三指 tip 距 cube 表面更接近 clean pinch。
    # actuator目标: [thumb_rot, thumb_flex, index, middle, ring, pinky]
    #             = [1.38, 0.0, 0.55, 0.55, 1.4, 1.4]
    # qpos顺序: index_mcp(0), index_pip(1), middle_mcp(2), middle_pip(3),
    #           ring_mcp(4), ring_pip(5), pinky_mcp(6), pinky_pip(7),
    #           thumb_cmc_abd(8), thumb_cmc_flex(9), thumb_mcp(10)
    self._pre_grasp_pose = jp.array([
        0.55,         # index_mcp
        0.925 * 0.55,
        0.55,         # middle_mcp
        0.925 * 0.55,
        1.3963,       # ring_mcp, ctrl target is 1.4 but joint range ends at 1.3963
        1.2915,
        1.3963,       # pinky_mcp
        1.2915,
        1.3788,       # thumb_cmc_abd, ctrl target is 1.38
        0.16 * 1.3788,
        0.0,          # thumb_mcp / thumb_flex actuator
    ], dtype=jp.float32)

    # R94/R99: lifted curriculum uses the same hand shape if re-enabled later.
    self._lifted_grasp_pose = jp.array([
        0.55,
        0.925 * 0.55,
        0.55,
        0.925 * 0.55,
        1.3963,
        1.2915,
        1.3963,
        1.2915,
        1.3788,
        0.16 * 1.3788,
        0.0,
    ], dtype=jp.float32)
    self._lifted_grasp_ctrl = jp.array([
        1.38, 0.0, 0.55, 0.55, 1.4, 1.4,
    ], dtype=jp.float32)

    self._using_coacd_collision = any(
        "_coacd_" in (mujoco.mj_id2name(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        for i in range(self.mj_model.ngeom)
    )
    self._using_fitted_collision = any(
        "_fitted_" in (mujoco.mj_id2name(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        for i in range(self.mj_model.ngeom)
    )
    self._using_capsule_collision = any(
        "_capsule_" in (mujoco.mj_id2name(
            self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, i) or "")
        for i in range(self.mj_model.ngeom)
    )

    # 每根手指所有碰撞 geom 的 ID (用于物理接触/作弊诊断)。
    # box 版使用手写语义 geom；CoACD 版退化为按 body 前缀分组。
    self._finger_contact_gids = self._build_finger_contact_gids(tip_only=False)
    # 指尖/指腹 geom 的 ID (用于 tactile obs / reward / release gating / diagnostics)。
    # CoACD 检查版没有独立指尖语义盒，暂用远端/拇指末端 body 的凸分解 geom。
    self._finger_tip_contact_gids = self._build_finger_contact_gids(tip_only=True)
    self._finger_order = ["index", "middle", "ring", "pinky", "thumb"]
    self._cube_geom_id = int(self.mj_model.geom('cube').id)
    self._cube_body_id = int(self.mj_model.body('cube').id)
    self._cube_mass = float(self.mj_model.body_mass[self._cube_body_id])
    self._nonprimary_contact_gids = (
        self._finger_contact_gids["ring"] + self._finger_contact_gids["pinky"]
    )
    self._palm_contact_gids = self._build_palm_contact_gids()

  def _geom_ids_by_names(self, names: list[str]) -> list[int]:
    ids = []
    for name in names:
      geom_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, name)
      if geom_id >= 0:
        ids.append(int(geom_id))
    return ids

  def _apply_passive_joint_couplings(self, q_hand: jax.Array) -> jax.Array:
    q_hand = q_hand.at[jp.array(_V2_FINGER_PIP_IDS)].set(
        self._finger_pip_coupling * q_hand[jp.array(_V2_FINGER_MCP_IDS)],
    )
    q_hand = q_hand.at[_V2_THUMB_FLEX_ID].set(
        self._thumb_passive_ratio * q_hand[self._thumb_passive_source_id],
    )
    return jp.clip(q_hand, self._lowers, self._uppers)

  def _geom_ids_by_prefixes(self, prefixes: list[str]) -> list[int]:
    ids = []
    for geom_id in range(self.mj_model.ngeom):
      name = mujoco.mj_id2name(
          self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
      if any(name.startswith(prefix) for prefix in prefixes):
        ids.append(int(geom_id))
    return ids

  def _build_finger_contact_gids(self, *, tip_only: bool) -> dict[str, list[int]]:
    if not self._using_coacd_collision and not self._using_fitted_collision and not self._using_capsule_collision:
      source = (
          consts.V2_FINGERTIP_CONTACT_GEOMS
          if tip_only else consts.V2_FINGER_CONTACT_GEOMS
      )
      return {
          finger: self._geom_ids_by_names(list(geom_names))
          for finger, geom_names in source.items()
      }

    # CoACD, fitted, and capsule all use the same body-level prefix structure.
    if self._using_coacd_collision:
      suffix = "_coacd_"
    elif self._using_fitted_collision:
      suffix = "_fitted_"
    else:
      suffix = "_capsule_"
    if tip_only:
      prefixes = {
          "index": [f"right_index_distal{suffix}"],
          "middle": [f"right_middle_distal{suffix}"],
          "ring": [f"right_ring_distal{suffix}"],
          "pinky": [f"right_pinky_distal{suffix}"],
          "thumb": [f"right_thumb_tip{suffix}"],
      }
    else:
      prefixes = {
          "index": [f"right_index_proximal{suffix}", f"right_index_distal{suffix}"],
          "middle": [f"right_middle_proximal{suffix}", f"right_middle_distal{suffix}"],
          "ring": [f"right_ring_proximal{suffix}", f"right_ring_distal{suffix}"],
          "pinky": [f"right_pinky_proximal{suffix}", f"right_pinky_distal{suffix}"],
          "thumb": [
              f"right_thumb_base{suffix}",
              f"right_thumb_mid{suffix}",
              f"right_thumb_tip{suffix}",
          ],
      }
    gids = {
        finger: self._geom_ids_by_prefixes(list(finger_prefixes))
        for finger, finger_prefixes in prefixes.items()
    }
    missing = [finger for finger, ids in gids.items() if not ids]
    if missing:
      raise ValueError(f"CoACD contact geom groups missing for: {missing}")
    return gids

  def _build_palm_contact_gids(self) -> list[int]:
    if self._using_coacd_collision:
      gids = self._geom_ids_by_prefixes(["palm_coacd_"])
      if not gids:
        raise ValueError("CoACD palm collision geoms are missing.")
      return gids
    if self._using_fitted_collision:
      gids = self._geom_ids_by_prefixes(["palm_fitted_"])
      if not gids:
        raise ValueError("Fitted palm collision geoms are missing.")
      return gids
    if self._using_capsule_collision:
      gids = self._geom_ids_by_prefixes(["palm_capsule_"])
      if not gids:
        raise ValueError("Capsule palm collision geoms are missing.")
      return gids
    return self._geom_ids_by_names([
        "palm_col_main",
        "palm_col_top",
        "palm_col_bot",
        "palm_col_left",
        "palm_col_right",
        "palm_rubber",
    ])

  def _build_taxel_weights(self) -> jax.Array:
    weights = jp.array(self._config.tactile_config.taxel_weights, dtype=jp.float32)
    return weights / (jp.sum(weights) + 1e-6)

  def _get_tactile_obs_dim(self) -> int:
    if bool(self._config.tactile_config.use_pooled_obs):
      return 5
    return len(consts.V2_SENSOR_FORCE_NAMES)

  # ── reset ────────────────────────────────────────────────────────────────

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, pos_rng, vel_rng, mode_rng = jax.random.split(rng, 4)

    # R89/R94: 一部分episode从预抓取或lifted可行抓取状态开始，缩短探索距离
    pre_grasp_frac = getattr(self._config.reset_config, 'pre_grasp_fraction', 0.0)
    lifted_grasp_frac = getattr(
        self._config.reset_config, 'lifted_grasp_fraction', 0.0,
    )
    mode_u = jax.random.uniform(mode_rng)
    use_lifted_grasp = mode_u < lifted_grasp_frac
    use_pre_grasp = (mode_u >= lifted_grasp_frac) & (
        mode_u < lifted_grasp_frac + pre_grasp_frac
    )
    normal_noise = self._config.reset_config.hand_qpos_noise_scale
    pre_grasp_noise = getattr(self._config.reset_config, 'pre_grasp_noise_scale', 0.15)
    lifted_noise = getattr(self._config.reset_config, 'lifted_grasp_noise_scale', 0.06)
    noise_scale = jp.where(
        use_lifted_grasp,
        lifted_noise,
        jp.where(use_pre_grasp, pre_grasp_noise, normal_noise),
    )
    base_pose = jp.where(
        use_lifted_grasp,
        self._lifted_grasp_pose,
        jp.where(use_pre_grasp, self._pre_grasp_pose, self._default_pose),
    )

    q_hand = jp.clip(
        base_pose
        + noise_scale * jax.random.normal(pos_rng, (consts.V2_NQ,)),
        self._lowers,
        self._uppers,
    )
    # Keep passive equality-coupled joints consistent after reset noise.
    q_hand = self._apply_passive_joint_couplings(q_hand)
    v_hand = jp.zeros(consts.V2_NV)

    rng, p_rng = jax.random.split(rng)
    base_pos = jp.array(self._config.spawn_config.cube_pos, dtype=jp.float32)
    jitter = jp.array(self._config.spawn_config.cube_jitter, dtype=jp.float32)
    lifted_z = getattr(self._config.reset_config, 'lifted_cube_z_offset', 0.010)
    start_pos = base_pos + jax.random.uniform(
        p_rng, (3,), minval=-jitter, maxval=jitter,
    )
    start_pos = start_pos.at[2].add(
        jp.where(use_lifted_grasp, lifted_z, 0.0),
    )
    start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])

    mocap_pos = self._initial_mocap_pos()
    configured_support = jp.array(bool(self._config.spawn_config.support_enabled))
    # R94b: lifted curriculum still keeps the normal support/release schedule.
    # Hiding support at reset caused lifted samples to drop before contact force
    # could build, so the curriculum never occupied 25% of rollout time.
    use_support = configured_support
    support_init = jp.where(use_support, self._support_pos, self._support_hidden_pos)
    mocap_pos = mocap_pos.at[self._support_mocap_id].set(support_init)

    ctrl_init = jp.where(
        use_lifted_grasp, self._lifted_grasp_ctrl, self._default_ctrl,
    )

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=ctrl_init,
        mocap_pos=mocap_pos,
    )

    # R66: 随机支撑释放时间
    rng, release_rng = jax.random.split(rng)
    if self._random_release:
      support_release_steps = jax.random.randint(
          release_rng, (), self._random_release_min_steps,
          self._random_release_max_steps + 1,
      )
    else:
      support_release_steps = jp.array(self._support_release_steps, dtype=jp.int32)

    action_scale = jp.array(self._config.action_scale, dtype=jp.float32)
    lifted_last_act = (self._lifted_grasp_ctrl - self._default_ctrl) / action_scale
    init_last_act = jp.where(
        use_lifted_grasp, lifted_last_act, jp.zeros(self.mjx_model.nu),
    )

    info = {
        "rng": rng,
        "last_act": init_last_act,
        "last_last_act": init_last_act,
        "motor_targets": data.ctrl,
        "ema_tip_finger_forces": jp.zeros(5, dtype=jp.float32),
        "tip_contact_flags": jp.zeros(5, dtype=jp.float32),
        "obs_tactile_ema": jp.zeros(5, dtype=jp.float32),
        "support_released": ~use_support,
        "support_timer": jp.array(0, dtype=jp.int32),
        "use_support": use_support,
        "stable_hold_steps": jp.array(0, dtype=jp.int32),
        "support_release_steps": support_release_steps,  # R66: per-episode (may be random)
        "lifted_reset": use_lifted_grasp,
        # C22: 扰动状态
        "gravity_tilt_angle": jp.zeros(2, dtype=jp.float32),  # (tilt_x, tilt_y) radians
        "perturbation_force": jp.zeros(3, dtype=jp.float32),  # current external force on cube
        "orientation_flip_force": jp.zeros(3, dtype=jp.float32),
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
      metrics[f"reward_sq/{k}"] = jp.zeros(())
    for k in _DIAGNOSTIC_METRIC_KEYS:
      metrics[k] = jp.zeros(())
    metrics["reward"] = jp.zeros(())  # EvalWrapper 兼容

    # state_dim: motor(6) + tactile(5) + last_act(6) + cube_pos_error(3) + cube_vel(3)
    #           + cube_quat(4) + fingertip_to_cube(15) + support_phase(2) + C20_hold_balance(2) = 46
    tactile_dim = self._get_tactile_obs_dim()  # 5
    state_dim = self.mjx_model.nu + tactile_dim + self.mjx_model.nu + 6 + 4 + 15 + 2 + 2  # C20: +2 hold_duration + force_balance
    obs_history = jp.zeros(self._config.history_len * state_dim)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  # ── step ─────────────────────────────────────────────────────────────────

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

    # ── C22: 扰动 — 外力脉冲 + 重力倾斜等效力 ─────────────────────────
    data = state.data
    pcfg = self._config.perturbation_config
    hold_steps = state.info["stable_hold_steps"]
    support_released = state.info["support_released"]
    step_rng = state.info["rng"]
    step_rng, force_rng, tilt_rng, flip_rng = jax.random.split(step_rng, 4)

    # --- 外力脉冲: 在方块上施加随机方向的力 ---
    ext_force = state.info["perturbation_force"]
    if pcfg.external_force_enabled:
      should_apply_ext = (
          support_released
          & (hold_steps >= pcfg.external_force_min_hold_steps)
          & (hold_steps % pcfg.external_force_interval == 0)
      )
      # 随机方向单位向量 * magnitude
      rand_dir = jax.random.normal(force_rng, (3,))
      rand_dir = rand_dir / (jp.linalg.norm(rand_dir) + 1e-8)
      new_force = rand_dir * pcfg.external_force_magnitude
      ext_force = jp.where(should_apply_ext, new_force, jp.zeros(3))

    # --- 重力倾斜等效力: 模拟手腕倾斜 ---
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

    # 重力倾斜 → 等效侧向力: F = m*g*sin(tilt), 方向=(sin(tx), sin(ty), 0)
    gravity_force = jp.array([
        self._cube_mass * 9.81 * jp.sin(tilt_angle[0]),
        self._cube_mass * 9.81 * jp.sin(tilt_angle[1]),
        0.0,
    ])

    # --- C28: 整手翻转等效力 ---
    # MuJoCo模型中手基座固定, 因此用持续的随机3D等效重力来近似
    # 手腕翻转后方块相对抓取坐标系受到的全向重力挑战。
    flip_force = state.info["orientation_flip_force"]
    if pcfg.orientation_flip_enabled:
      flip_active = (
          support_released
          & (hold_steps >= pcfg.orientation_flip_min_hold_steps)
      )
      should_update_flip = (
          flip_active
          & (hold_steps % pcfg.orientation_flip_change_interval == 0)
      )
      flip_dir = jax.random.normal(flip_rng, (3,))
      flip_dir = flip_dir / (jp.linalg.norm(flip_dir) + 1e-8)
      new_flip_force = (
          flip_dir * self._cube_mass * 9.81 * pcfg.orientation_flip_force_scale
      )
      flip_force = jp.where(should_update_flip, new_flip_force, flip_force)
      flip_force = jp.where(flip_active, flip_force, jp.zeros(3))

    # 合并外力: xfrc_applied shape = (nbody, 6), 前3为力, 后3为力矩
    total_force = self._clip_force_vector(ext_force + gravity_force + flip_force)
    xfrc = data.xfrc_applied.at[self._cube_body_id, :3].set(total_force)
    data = data.replace(xfrc_applied=xfrc)

    data = mjx_env.step(self.mjx_model, data, motor_targets, self.n_substeps)
    nonfinite_state = self._has_nonfinite_state(data)

    # efc_force 指尖/指腹接触力
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
    # C22: 扰动状态持久化
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
      # Progressive hold tracking
      cube_pos_hold = self.get_cube_position(data)
      cube_above = cube_pos_hold[2] > (self._spawn_z - 0.010)
      primary_hold_forces = jp.array([
          jp.abs(ema_tip_finger_forces[0]),
          jp.abs(ema_tip_finger_forces[1]),
          jp.abs(ema_tip_finger_forces[4]),
      ])
      primary_hold_flags = jp.array([
          tip_contact_flags[0],
          tip_contact_flags[1],
          tip_contact_flags[4],
      ])
      primary_hold_active = jp.maximum(
          (primary_hold_forces > 0.10).astype(jp.float32),
          primary_hold_flags,
      )
      primary_hold_count = jp.sum(primary_hold_active)
      is_holding = cube_above & (primary_hold_count >= 2.0) & support_released
      info["stable_hold_steps"] = jp.where(
          is_holding,
          state.info["stable_hold_steps"] + 1,
          jp.array(0, dtype=jp.int32),
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
      new_metrics["reward"] = reward  # EvalWrapper 兼容

      done = done.astype(reward.dtype)
      return state.replace(
          data=data, obs=obs, reward=reward, done=done,
          metrics=new_metrics, info=info,
      )

    return jax.lax.cond(nonfinite_state, _handle_nonfinite, _handle_normal, operand=None)

  # ── 辅助方法 ─────────────────────────────────────────────────────────────

  def _initial_mocap_pos(self) -> jax.Array:
    if self.mj_model.nmocap == 0:
      return jp.zeros((0, 3), dtype=jp.float32)
    mocap_pos = np.zeros((self.mj_model.nmocap, 3), dtype=np.float32)
    for body_id in range(self.mj_model.nbody):
      mocap_id = int(self.mj_model.body_mocapid[body_id])
      if mocap_id >= 0:
        mocap_pos[mocap_id] = self.mj_model.body_pos[body_id]
    return jp.array(mocap_pos)

  def _stabilize_action(self, action: jax.Array) -> jax.Array:
    max_abs_action = float(
        getattr(getattr(self._config, "stability_config", object()), "max_abs_action", 1.0)
    )
    if max_abs_action >= 1.0:
      return action
    return jp.clip(action, -max_abs_action, max_abs_action)

  def _stabilize_motor_targets(
      self, motor_targets: jax.Array, last_targets: jax.Array,
  ) -> jax.Array:
    delta_clip = getattr(
        getattr(self._config, "stability_config", object()), "motor_delta_clip", None,
    )
    if delta_clip is None:
      return motor_targets
    delta_clip = jp.asarray(delta_clip, dtype=jp.float32)
    return jp.clip(motor_targets, last_targets - delta_clip, last_targets + delta_clip)

  def _has_nonfinite_state(self, data: mjx.Data) -> jax.Array:
    if not bool(getattr(getattr(self._config, "stability_config", object()), "terminate_on_nonfinite", True)):
      return jp.array(False)
    cube_pos = self.get_cube_position(data)
    cube_linvel = self.get_cube_linvel(data)
    checks = [
        jp.all(jp.isfinite(data.qpos)),
        jp.all(jp.isfinite(data.qvel)),
        jp.all(jp.isfinite(data.ctrl)),
        jp.all(jp.isfinite(cube_pos)),
        jp.all(jp.isfinite(cube_linvel)),
    ]
    if data.act is not None:
      checks.append(jp.all(jp.isfinite(data.act)))
    return ~jp.all(jp.stack(checks))

  def _clip_force_vector(self, force: jax.Array) -> jax.Array:
    clip_n = float(
        getattr(self._config.perturbation_config, "total_force_clip_n", 0.0)
    )
    if clip_n <= 0.0:
      return force
    norm = jp.linalg.norm(force)
    scale = jp.minimum(1.0, clip_n / (norm + 1e-8))
    return force * scale

  def _make_nonfinite_failure_state(
      self,
      state: mjx_env.State,
      action: jax.Array,
      motor_targets: jax.Array,
      info: dict[str, Any],
  ) -> mjx_env.State:
    penalty_mult = float(
        getattr(
            getattr(self._config, "stability_config", object()),
            "nonfinite_penalty_mult",
            2.0,
        )
    )
    reward = (
        float(self._config.reward_config.scales.termination)
        * penalty_mult
        * self.dt
    )
    reward = jp.array(reward, dtype=jp.float32)
    info["motor_targets"] = motor_targets
    info["last_last_act"] = state.info["last_act"]
    info["last_act"] = action
    info["stable_hold_steps"] = jp.array(0, dtype=jp.int32)

    new_metrics = {}
    for k in self._config.reward_config.scales.keys():
      new_metrics[f"reward/{k}"] = jp.zeros((), dtype=jp.float32)
      new_metrics[f"reward_sq/{k}"] = jp.zeros((), dtype=jp.float32)
    new_metrics["reward/termination"] = reward / self.dt
    new_metrics["reward_sq/termination"] = jp.square(reward / self.dt)
    for k in _DIAGNOSTIC_METRIC_KEYS:
      new_metrics[k] = jp.zeros((), dtype=jp.float32)
    new_metrics["diagnostic/drop"] = jp.array(1.0, dtype=jp.float32)
    new_metrics["diagnostic/nonfinite_state"] = jp.array(1.0, dtype=jp.float32)
    new_metrics["termination/drop"] = jp.array(1.0, dtype=jp.float32)
    new_metrics["diagnostic/lifted_reset"] = info.get(
        "lifted_reset", jp.array(False),
    ).astype(jp.float32)
    new_metrics["reward"] = reward

    return state.replace(
        data=state.data,
        obs=state.obs,
        reward=reward,
        done=jp.array(1.0, dtype=jp.float32),
        metrics=new_metrics,
        info=info,
    )

  def _should_release_support(
      self, already_released: jax.Array, support_timer: jax.Array,
      release_steps: jax.Array = None,
      tip_finger_forces: jax.Array | None = None,
      tip_contact_flags: jax.Array | None = None,
  ) -> jax.Array:
    # R66: use per-episode release steps if provided.
    # C05: release on primary force/geom readiness, with a late forced release
    # fallback so post-release failures still enter the training distribution.
    threshold = release_steps if release_steps is not None else self._support_release_steps
    time_ready = support_timer >= threshold
    if tip_finger_forces is None:
      grasp_ready = jp.array(True)
    else:
      grasp_ready = self._is_grasp_ready_for_release(
          tip_finger_forces, tip_contact_flags,
      )
    require_grasp = jp.array(
        bool(getattr(self._config.support_config, 'require_grasp_for_release', True))
    )
    force_release = (
        (support_timer >= self._force_release_steps)
        if self._force_release_steps > 0 else jp.array(False)
    )
    release_now = (time_ready & (grasp_ready | ~require_grasp)) | force_release
    return jp.logical_or(already_released, release_now)

  def _is_grasp_ready_for_release(
      self,
      tip_finger_forces: jax.Array,
      tip_contact_flags: jax.Array | None = None,
  ) -> jax.Array:
    abs_f = jp.abs(tip_finger_forces)
    primary = jp.array([abs_f[0], abs_f[1], abs_f[4]])
    if tip_contact_flags is None:
      primary_geom = jp.zeros(3, dtype=jp.float32)
    else:
      primary_geom = jp.array([
          tip_contact_flags[0],
          tip_contact_flags[1],
          tip_contact_flags[4],
      ])
    active_th = self._config.reward_config.finger_active_threshold
    active = jp.maximum((primary > active_th).astype(jp.float32), primary_geom)
    active_count = jp.sum(active)
    min_active = float(self._config.support_config.min_release_active_fingers)
    force_ok = jp.sum(primary) >= self._config.support_config.min_release_force
    return (active_count >= min_active) & force_ok

  def _set_support_state(
      self, data: mjx.Data, support_released: jax.Array,
      support_timer: jax.Array = None,
  ) -> mjx.Data:
    if self.mj_model.nmocap == 0:
      return data
    if support_timer is not None and self._support_ramp_steps > 0:
      elapsed = jp.clip(
          (support_timer - self._support_release_steps).astype(jp.float32),
          0.0, float(self._support_ramp_steps),
      )
      alpha = elapsed / float(self._support_ramp_steps)
      ramp_done = alpha >= 1.0
      ramp_target = self._support_pos.at[2].set(self._support_pos[2] - 0.05)
      ramped_pos = (1.0 - alpha) * self._support_pos + alpha * ramp_target
      ramped_pos = jp.where(ramp_done, self._support_hidden_pos, ramped_pos)
      support_pos = jp.where(support_released, ramped_pos, self._support_pos)
    else:
      support_pos = jp.where(
          support_released, self._support_hidden_pos, self._support_pos,
      )
    mocap_pos = data.mocap_pos.at[self._support_mocap_id].set(support_pos)
    return data.replace(mocap_pos=mocap_pos)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    cube_z = self.get_cube_position(data)[2]
    drop_z = self._spawn_z - 0.04  # R86: 0.03→0.04, 高位方块需更多容差
    return cube_z < drop_z

  def _has_cube_contact(self, data: mjx.Data, geom_ids: list[int]) -> jax.Array:
    contact = data.contact
    active = contact.dist < 0.0
    cube_id = self._cube_geom_id
    hit = jp.array(False)
    for gid in geom_ids:
      g0 = (contact.geom[:, 0] == gid) & (contact.geom[:, 1] == cube_id)
      g1 = (contact.geom[:, 1] == gid) & (contact.geom[:, 0] == cube_id)
      hit = hit | jp.any((g0 | g1) & active)
    return hit.astype(jp.float32)

  def _get_tip_contact_flags(self, data: mjx.Data) -> jax.Array:
    return jp.array([
        self._has_cube_contact(data, self._finger_tip_contact_gids["index"]),
        self._has_cube_contact(data, self._finger_tip_contact_gids["middle"]),
        self._has_cube_contact(data, self._finger_tip_contact_gids["ring"]),
        self._has_cube_contact(data, self._finger_tip_contact_gids["pinky"]),
        self._has_cube_contact(data, self._finger_tip_contact_gids["thumb"]),
    ], dtype=jp.float32)

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
    primary_forces = jp.array([abs_f[0], abs_f[1], abs_f[4]])
    any_abs_f = jp.abs(self._get_contact_forces_efc(
        data, self._finger_contact_gids,
    ))
    primary_any_forces = jp.array([any_abs_f[0], any_abs_f[1], any_abs_f[4]])
    active_th = self._config.reward_config.finger_active_threshold
    primary_geom = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[4],
    ])
    primary_active = jp.maximum(
        (primary_forces > active_th).astype(jp.float32),
        primary_geom,
    )
    primary_count = jp.sum(primary_active)
    primary_any_active = (primary_any_forces > active_th).astype(jp.float32)
    primary_any_count = jp.sum(primary_any_active)
    non_tip_primary_contact = (
        (primary_any_count > primary_count) & (primary_any_count >= 2.0)
    ).astype(jp.float32)
    three_contact = (primary_count >= 3.0).astype(jp.float32)
    two_plus_contact = (primary_count >= 2.0).astype(jp.float32)

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
    cube_above = (cube_pos[2] > (self._spawn_z - 0.010)).astype(jp.float32)
    hold_success = (
        (stable_hold_steps >= hold_success_steps)
        & (cube_above > 0.0)
        & (three_contact > 0.0)
    ).astype(jp.float32)

    lin_speed = jp.linalg.norm(cube_linvel)
    slip_event = (
        (three_contact > 0.0)
        & support_released
        & ((lin_speed > 0.12) | (cube_linvel[2] < -0.03))
    ).astype(jp.float32)
    nonprimary_contact = jp.maximum(
        ((abs_f[2] > active_th) | (abs_f[3] > active_th)).astype(jp.float32),
        self._has_cube_contact(data, self._nonprimary_contact_gids),
    )
    palm_contact = self._has_cube_contact(data, self._palm_contact_gids)
    drop = done.astype(jp.float32)

    return {
        "diagnostic/success": hold_success,
        "diagnostic/three_finger_contact": three_contact,
        "diagnostic/two_plus_primary_contact": two_plus_contact,
        "diagnostic/contact_duration_sec": three_contact * self.dt,
        "diagnostic/lift_success": lift_success,
        "diagnostic/hold_success": hold_success,
        "diagnostic/drop": drop,
        "diagnostic/normal_force_mean": jp.mean(primary_forces),
        "diagnostic/tangent_force_approx": jp.linalg.norm(cube_linvel[:2]) * three_contact,
        "diagnostic/slip_event": slip_event,
        "diagnostic/primary_active_count": primary_count,
        "diagnostic/primary_any_active_count": primary_any_count,
        "diagnostic/non_tip_primary_contact": non_tip_primary_contact,
        "diagnostic/nonprimary_contact": nonprimary_contact,
        "diagnostic/palm_contact": palm_contact,
        "diagnostic/support_released": support_released.astype(jp.float32),
        "diagnostic/lift_height": lift_height,
        "termination/drop": drop,
    }

  # ── 接触力提取 ───────────────────────────────────────────────────────────

  def _get_contact_forces_efc(
      self,
      data: mjx.Data,
      contact_gids: dict[str, list[int]] | None = None,
  ) -> jax.Array:
    """通过 efc_force + contact.geom 提取每指接触力。

    默认只累加指尖/指腹碰撞 geom，作为 actor 低维触觉、释放门控、
    三指捏握 reward 与诊断的压力来源；近端/远端碰撞仍可通过 contact_gids
    显式传入。
    Returns: shape (5,) 顺序 [index, middle, ring, pinky, thumb]。
    """
    if contact_gids is None:
      contact_gids = self._finger_tip_contact_gids
    contact = data.contact
    active = contact.dist < 0.0
    safe_idx = jp.maximum(contact.efc_address, 0)
    forces = jp.abs(data.efc_force[safe_idx])
    forces = jp.where(active & (contact.efc_address >= 0), forces, 0.0)

    cube_id = self._cube_geom_id

    def _finger_force(gid):
      g0 = (contact.geom[:, 0] == gid) & (contact.geom[:, 1] == cube_id)
      g1 = (contact.geom[:, 1] == gid) & (contact.geom[:, 0] == cube_id)
      mask = (g0 | g1) & active
      return jp.sum(jp.where(mask, forces, 0.0))

    per_finger = []
    for finger_name in self._finger_order:
      gids = contact_gids[finger_name]
      finger_total = sum(_finger_force(gid) for gid in gids)
      per_finger.append(finger_total)
    return jp.stack(per_finger)

  # ── 观测 ─────────────────────────────────────────────────────────────────

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array,
  ) -> Dict[str, jax.Array]:
    info["rng"], noise_rng = jax.random.split(info["rng"])

    hw_pos = jp.asarray(info["motor_targets"], dtype=jp.float32)
    noisy_hw_pos = (
        hw_pos
        + (2 * jax.random.uniform(noise_rng, shape=hw_pos.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.hw_pos
    )
    # C22: 关节角度观测噪声 — 模拟传感器不确定性
    if self._config.perturbation_config.joint_obs_noise_enabled:
      info["rng"], jnoise_rng = jax.random.split(info["rng"])
      joint_noise = jax.random.normal(jnoise_rng, shape=hw_pos.shape) * self._config.perturbation_config.joint_obs_noise_std
      noisy_hw_pos = noisy_hw_pos + joint_noise

    # 触觉观测 — 直接使用 EMA 平滑后的力信号 (干净, 无额外噪声)
    sat = float(self._config.tactile_config.force_saturation_n)
    force_obs = jp.clip(info["obs_tactile_ema"] / sat, 0.0, 1.0)  # (5,)

    # V2 R06+: 将方块反馈加入策略可见观测
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = (palm_pos - cube_pos)  # 3D
    cube_linvel = self.get_cube_linvel(data)
    cube_vel_scaled = cube_linvel / 0.15   # normalize by max expected velocity

    # R66: 方块姿态 (4D quaternion)
    cube_quat = self.get_cube_orientation(data)  # 4D

    # R66: 5 指尖到方块中心的 3D 相对距离 (15D)
    fingertip_rel = self.get_fingertip_positions(data).reshape(5, 3)  # relative to grasp_site
    fingertip_world = fingertip_rel + palm_pos  # world frame
    fingertip_to_cube = (fingertip_world - cube_pos[None, :]).reshape(-1)  # (15,)
    fingertip_to_cube_scaled = fingertip_to_cube / 0.05  # normalize by ~5cm scale

    # R90: support phase signals — 让agent预判support释放时机
    support_timer_f = info.get("support_timer", jp.array(0)).astype(jp.float32)
    release_steps_f = info.get("support_release_steps",
                               jp.array(self._support_release_steps)).astype(jp.float32)
    # pre_release_progress: 0 at reset → 1 when support is about to release
    pre_release_progress = jp.clip(
        support_timer_f / jp.maximum(release_steps_f, 1.0), 0.0, 1.0)
    # post_release_progress: 0 before release, ramp 0→1 during release ramp, 1 after
    ramp_steps_f = float(max(self._support_ramp_steps, 1))
    post_release_elapsed = jp.maximum(support_timer_f - release_steps_f, 0.0)
    post_release_progress = jp.clip(post_release_elapsed / ramp_steps_f, 0.0, 1.0)

    # C20: 持握时长归一化 — 让actor感知已持握多久 (0→1 over 600 steps = 30s)
    hold_steps_f = info.get("stable_hold_steps", jp.array(0)).astype(jp.float32)
    hold_duration_normalized = jp.clip(hold_steps_f / 600.0, 0.0, 1.0)

    # C20: 三指力平衡度 — 让actor感知力是否均匀 (1=完美均衡, 0=完全不均)
    _obs_forces = jp.abs(info["ema_tip_finger_forces"])
    _obs_primary = jp.array([_obs_forces[0], _obs_forces[1], _obs_forces[4]])
    _obs_mean_f = jp.mean(_obs_primary)
    _obs_rel_std = jp.std(_obs_primary) / (_obs_mean_f + 1e-6)
    force_balance_obs = jp.clip(1.0 - _obs_rel_std, 0.0, 1.0) * jp.clip(
        _obs_mean_f / 0.1, 0.0, 1.0)  # 有力时才有意义

    state = jp.concatenate([
        noisy_hw_pos,              # 6D motor targets
        force_obs,                 # 5D tactile (clean EMA, no noise)
        info["last_act"],          # 6D
        cube_pos_error,            # 3D cube position error (palm-cube)
        cube_vel_scaled,           # 3D cube velocity (normalized)
        cube_quat,                 # 4D cube orientation (R66)
        fingertip_to_cube_scaled,  # 15D fingertip-to-cube distances (R66)
        jp.array([pre_release_progress, post_release_progress]),  # 2D support phase (R90)
        jp.array([hold_duration_normalized, force_balance_obs]),  # 2D C20: hold progress + force balance
    ])

    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    # privileged state (uses cube data already computed above)
    # cube_quat, fingertip_to_cube already in state (R66),
    # but keep raw fingertip_positions + angvel etc. for value function
    cube_angvel = self.get_cube_angvel(data)
    joint_angles = data.qpos[self._hand_qids]

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        data.actuator_force,
        fingertip_rel.reshape(-1),  # raw fingertip relative positions
        cube_pos_error,
        cube_quat,
        cube_angvel,
        cube_linvel,
    ])

    return {
        "state": obs_history,
        "privileged_state": privileged_state,
    }

  # ── 奖励 ─────────────────────────────────────────────────────────────────

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
    # R89: primary 3指距离(index=0, middle=1, thumb=4), 替代全5指均值
    primary_tip_dists = jp.array([tip_dists[0], tip_dists[1], tip_dists[4]])
    mean_primary_dist = jp.mean(primary_tip_dists)
    max_primary_dist = jp.max(primary_tip_dists)
    hand_q = data.qpos[self._hand_qids]
    tip_finger_forces = info["ema_tip_finger_forces"]
    tip_contact_flags = info.get("tip_contact_flags", jp.zeros(5, dtype=jp.float32))
    cube_linvel = self.get_cube_linvel(data)
    cube_angvel = self.get_cube_angvel(data)

    # 门控信号
    near_gate = jp.clip(1.0 - min_tip_dist / 0.10, 0.0, 1.0)
    contact_gate = jp.clip(1.0 - min_tip_dist / 0.06, 0.0, 1.0)
    mcp_angles = jp.take(hand_q, jp.array(_V2_FINGER_MCP_IDS))
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

    # R78: 主要手指(拇指+食指+中指)参与度门控
    # — hold/progressive奖励必须在主要手指有接触时才给分
    _pf_abs = jp.abs(tip_finger_forces)
    _pf_th = self._config.reward_config.finger_active_threshold
    _pf_force_active = jp.array([
        (_pf_abs[0] > _pf_th).astype(jp.float32),  # index
        (_pf_abs[1] > _pf_th).astype(jp.float32),  # middle
        (_pf_abs[4] > _pf_th).astype(jp.float32),  # thumb
    ])
    _pf_geom_active = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[4],
    ])
    _pf_active = jp.maximum(_pf_force_active, _pf_geom_active)
    _pf_count = jp.sum(_pf_active)  # 0~3
    # soft gate: 0 if <=1 active, ramp 1→2→3
    primary_gate = jp.clip((_pf_count - 1.0) / 2.0, 0.0, 1.0)
    palm_contact = self._has_cube_contact(data, self._palm_contact_gids)
    nonprimary_force_contact = (
        (_pf_abs[2] > _pf_th) | (_pf_abs[3] > _pf_th)
    ).astype(jp.float32)
    nonprimary_contact = jp.maximum(
        nonprimary_force_contact,
        self._has_cube_contact(data, self._nonprimary_contact_gids),
    )
    cheat_contact = jp.clip(jp.maximum(palm_contact, nonprimary_contact), 0.0, 1.0)
    clean_primary_gate = primary_gate * (1.0 - cheat_contact)
    clean_released_gate = released_gate * (1.0 - cheat_contact)

    # R49+R78: 分层门控
    return {
        # 形状引导 (不门控 — 维持Phase 1行为)
        # R89: approach仅用primary三指均值, 不再引导palm靠近
        "approach": self._reward_approach(mean_primary_dist),
        "three_finger_proximity": self._reward_three_finger_proximity(max_primary_dist),
        "contact": self._reward_contact(tip_world, cube_pos),
        "thumb_engage": self._reward_thumb_engage(tip_dists),
        "closure": self._reward_closure(hand_q) * near_gate,
        "pip_closure": self._reward_pip_closure(hand_q) * near_gate * mcp_gate,
        "human_pose": self._reward_human_pose(hand_q) * near_gate,
        # 抓握验证 (R78: hold类乘以primary_gate, 必须主要手指参与)
        # C11: un-gate from released — train force during support phase too
        "grip_force": self._reward_grip_force(tip_finger_forces) * (1.0 - cheat_contact),
        "hold_position": self._reward_hold_position(cube_pos, cube_linvel) * released_gate * clean_primary_gate,
        "stable_hold": self._reward_stable_hold(
            cube_pos, cube_linvel, cube_angvel, tip_finger_forces,
        ) * released_gate * clean_primary_gate,
        "force_contact": self._reward_force_contact(tip_finger_forces) * released_gate,
        "primary_geom_contact": self._reward_primary_geom_contact(tip_contact_flags),
        "progressive_hold": self._reward_progressive_hold(info) * clean_primary_gate,
        "sustained_hold_bonus": self._reward_sustained_hold_bonus(info) * clean_primary_gate,
        "force_balance": self._reward_force_balance(tip_finger_forces) * (1.0 - cheat_contact),
        "finger_participation": self._reward_finger_participation(tip_finger_forces) * released_gate,
        # R83: 移除released_gate — 支撑阶段也需要学习拇指对立位置
        "thumb_opposition": self._reward_thumb_opposition(
            tip_world, cube_pos, tip_finger_forces,
        ),
        # C11: un-gate from released — train force during support phase too
        "primary_finger_force": self._reward_primary_finger_force(
            tip_finger_forces,
        ) * (1.0 - cheat_contact),
        "pre_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * (1.0 - released_gate) * (1.0 - cheat_contact),
        "post_release_grasp": self._reward_release_ready(
            tip_finger_forces,
        ) * released_gate * (1.0 - cheat_contact),
        "post_release_survival": self._reward_post_release_survival(
            cube_pos, cube_linvel,
        ) * released_gate * primary_gate * (1.0 - cheat_contact),
        "post_release_cheat_contact": released_gate * cheat_contact,
        "post_release_slip": self._cost_post_release_slip(
            cube_linvel,
        ) * released_gate * primary_gate * (1.0 - cheat_contact),
        "post_release_pose_hold": self._reward_post_release_pose_hold(
            hand_q,
        ) * released_gate * primary_gate * (1.0 - cheat_contact),
        "soft_contact": self._reward_soft_contact(tip_finger_forces),
        "idle_follow": self._reward_idle_follow(hand_q, tip_finger_forces),
        "height": self._reward_height(cube_pos, palm_pos) * clean_primary_gate,
        "survival": 1.0 - done,
        "termination": done,
        "drop_risk": self._cost_drop_risk(cube_pos, cube_linvel),
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "action_accel": self._cost_action_accel(action, info["last_act"], info["last_last_act"]),
        "torques": self._cost_torques(data.actuator_force),
        "force_overload": self._cost_force_overload(tip_finger_forces),
        "palm_contact": palm_contact,
        "nonprimary_contact": nonprimary_contact,
    }

  # ── 奖励组件 ─────────────────────────────────────────────────────────────

  def _reward_approach(self, mean_primary_dist: jax.Array) -> jax.Array:
    # R89: 仅用primary三指(thumb+index+middle)均值, 不引导palm/ring/pinky
    return jp.exp(-15.0 * mean_primary_dist)

  def _reward_three_finger_proximity(self, max_primary_dist: jax.Array) -> jax.Array:
    """R89: 三指同时靠近方块的显式正梯度。

    使用三指中最远的那根(max)作为瓶颈指标:
    只有当三指都靠近时才有高奖励, 避免只有1-2指靠近就拿高分。
    max=100mm → 0.37, max=50mm → 0.61, max=20mm → 0.82, max=10mm → 0.90
    """
    return jp.exp(-10.0 * max_primary_dist)

  def _reward_contact(self, tip_world: jax.Array, cube_pos: jax.Array) -> jax.Array:
    diff = jp.abs(tip_world - cube_pos[None, :]) - 0.0125
    surface_dists = jp.linalg.norm(jp.maximum(diff, 0.0), axis=1)
    return jp.mean(jp.exp(-5.0 * surface_dists))

  def _reward_thumb_engage(self, tip_dists: jax.Array) -> jax.Array:
    # R81: -35→-12, 让10cm处有0.30奖励(原来≈0), 引导拇指靠近方块
    return jp.exp(-12.0 * tip_dists[4])

  def _reward_closure(self, hand_q: jax.Array) -> jax.Array:
    """V2: 四指 MCP (60%) + 拇指三关节均值 (40%)。"""
    finger_mcp = jp.take(hand_q, jp.array(_V2_FINGER_MCP_IDS))
    finger_close = jp.mean(jp.clip(finger_mcp / 1.2, 0.0, 1.0))
    # 拇指: abd(8) + flex(9) + mcp(10)
    thumb_close = jp.mean(jp.clip(
        jp.array([hand_q[8] / 1.2, hand_q[9] / 0.5, hand_q[10] / 0.7]),
        0.0, 1.0,
    ))
    return 0.6 * finger_close + 0.4 * thumb_close

  def _reward_pip_closure(self, hand_q: jax.Array) -> jax.Array:
    """V2: PIP 远端弯曲奖励 (V2 无 DIP，仅奖励 PIP)。"""
    finger_pip = jp.take(hand_q, jp.array(_V2_FINGER_PIP_IDS))
    finger_score = jp.mean(jp.clip(finger_pip / 1.1, 0.0, 1.0))
    thumb_score = jp.clip(hand_q[10] / 0.7, 0.0, 1.0)
    return 0.8 * finger_score + 0.2 * thumb_score

  def _reward_human_pose(self, hand_q: jax.Array) -> jax.Array:
    """R74 自然抓握姿态目标。

    V2 每指 2 级 (MCP + PIP)，无 DIP。
    拇指: CMC_ABD + CMC_FLEX + MCP。
    R74: 放松 ring/pinky 目标角度 (0.5 rad)，不强制闭合。
    """
    # 四指目标 [MCP, PIP] × 4
    # R74: index/middle 保持 1.1/0.9, ring/pinky 放松到 0.5/0.4
    finger_targets = jp.array([
        1.10, 0.90,  # index
        1.10, 0.90,  # middle
        0.50, 0.40,  # ring   — R74: 放松
        0.50, 0.40,  # pinky  — R74: 放松
    ])
    # 拇指目标 [abd(8), flex(9), mcp(10)]
    thumb_targets = jp.array([1.10, 0.30, 0.50])

    finger_q = jp.take(hand_q, jp.array([0, 1, 2, 3, 4, 5, 6, 7]))
    finger_err = jp.square(finger_q - finger_targets)
    # R74: index/middle 权重更高
    finger_weights = jp.array([1.5, 1.5, 1.5, 1.5, 0.5, 0.5, 0.5, 0.5])
    finger_score = jp.sum(finger_weights * jp.exp(-8.0 * finger_err)) / jp.sum(finger_weights)

    thumb_q = jp.array([hand_q[8], hand_q[9], hand_q[10]])
    thumb_err = jp.square(thumb_q - thumb_targets)
    thumb_score = jp.mean(jp.exp(-8.0 * thumb_err))

    return 0.6 * finger_score + 0.4 * thumb_score

  def _reward_grip_force(self, tip_force: jax.Array) -> jax.Array:
    """R78: 仅计算主要三指(index,middle,thumb)的力。

    不再取 top-3 任意手指, 强制策略用正确手指夹持。
    """
    abs_f = jp.abs(tip_force)
    # R78: 只看 index(0), middle(1), thumb(4)
    primary = jp.array([abs_f[0], abs_f[1], abs_f[4]])
    per_finger = jp.clip(primary / 0.15, 0.0, 1.0)
    return jp.mean(per_finger)

  def _reward_hold_position(
      self, cube_pos: jax.Array, cube_linvel: jax.Array,
  ) -> jax.Array:
    # R92: 持握位置只约束水平漂移和防掉落，不再用3D距离惩罚抬升。
    # R91 的 dist_from_spawn 会让 cube 抬高 2cm 时 hold_position 下降，
    # 与 lift 目标冲突。
    xy_err = jp.linalg.norm((cube_pos - self._spawn_pos)[:2])
    xy_reward = jp.exp(-40.0 * xy_err)
    z_floor = self._spawn_z - 0.012
    z_safe = jp.clip((cube_pos[2] - z_floor) / 0.012, 0.0, 1.0)
    vel_penalty = jp.clip(jp.linalg.norm(cube_linvel) / 0.20, 0.0, 1.0)  # R66: 0.15→0.20
    return xy_reward * z_safe * (1.0 - 0.5 * vel_penalty)  # R66: 0.7→0.5 降低速度惩罚

  def _reward_stable_hold(
      self,
      cube_pos: jax.Array,
      cube_linvel: jax.Array,
      cube_angvel: jax.Array,
      tip_force: jax.Array,
  ) -> jax.Array:
    active_th = self._config.reward_config.finger_active_threshold
    active_count = jp.sum((jp.abs(tip_force) > active_th).astype(jp.float32))
    contact_gate = jp.clip((active_count - 1.0) / 2.0, 0.0, 1.0)
    lin_stable = jp.exp(-6.0 * jp.linalg.norm(cube_linvel))
    ang_stable = jp.exp(-4.0 * jp.linalg.norm(cube_angvel))
    return contact_gate * lin_stable * ang_stable

  def _reward_force_contact(self, tip_force: jax.Array) -> jax.Array:
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.force_contact_threshold
    sat = self._config.reward_config.force_contact_saturation
    normalized = jp.clip((abs_f - th) / (sat - th + 1e-6), 0.0, 1.0)
    return jp.mean(normalized)

  def _reward_primary_geom_contact(self, tip_contact_flags: jax.Array) -> jax.Array:
    primary = jp.array([
        tip_contact_flags[0],
        tip_contact_flags[1],
        tip_contact_flags[4],
    ])
    return jp.mean(primary) * (0.35 + 0.65 * jp.min(primary))

  def _reward_progressive_hold(self, info: dict) -> jax.Array:
    """Reward that grows with consecutive stable hold duration.

    C20: 改为线性增长 + 更高cap，让14s→30s持握有持续梯度。
    旧: sqrt(steps/50) cap=3.0 → step 450 (22.5s) 就封顶，14s→30s几乎无梯度。
    新: steps/200 cap=5.0 → step 1000 (50s) 才封顶。
    At step 100(5s): 0.5, step 200(10s): 1.0, step 400(20s): 2.0, step 600(30s): 3.0, step 1000(50s): 5.0
    """
    steps = info["stable_hold_steps"].astype(jp.float32)
    return jp.minimum(steps / 200.0, 5.0)

  def _reward_force_balance(self, tip_force: jax.Array) -> jax.Array:
    # R81: 仅看主要三指(index,middle,thumb)的力平衡
    # Rajeswaran 2017: 力封闭的本质是对抗力平衡
    abs_f = jp.abs(tip_force)
    primary = jp.array([abs_f[0], abs_f[1], abs_f[4]])  # index, middle, thumb
    mean_f = jp.mean(primary)
    force_level = jp.clip((mean_f - 0.04) / 0.16, 0.0, 1.0)
    rel_std = jp.std(primary) / (mean_f + 1e-6)
    return force_level * jp.clip(1.0 - rel_std, 0.0, 1.0)

  def _reward_sustained_hold_bonus(self, info: dict) -> jax.Array:
    """C20: 长时间持握阶梯奖励。

    持握超过200步(10s)给第一档bonus=1.0，超过400步(20s)第二档=2.0，超过600步(30s)=3.0。
    每步给予该bonus，产生递增的总奖励。
    与progressive_hold互补：progressive_hold是连续梯度，sustained是阶梯式milestone。
    """
    steps = info["stable_hold_steps"].astype(jp.float32)
    bonus_10s = (steps > 200.0).astype(jp.float32)  # 持握>10s
    bonus_20s = (steps > 400.0).astype(jp.float32)  # 持握>20s
    bonus_30s = (steps > 600.0).astype(jp.float32)
    return bonus_10s + bonus_20s + bonus_30s

  def _reward_finger_participation(self, tip_force: jax.Array) -> jax.Array:
    th = self._config.reward_config.finger_active_threshold
    active = (jp.abs(tip_force) > th).astype(jp.float32)
    all_ratio = jp.mean(active)
    non_thumb_ratio = jp.mean(active[:4])
    return 0.6 * all_ratio + 0.4 * non_thumb_ratio

  def _reward_thumb_opposition(
      self, tip_world: jax.Array, cube_pos: jax.Array,
      tip_force: jax.Array,
  ) -> jax.Array:
    """R74: 拇指对立奖励 — 偏好 index/middle 对立。

    计算拇指→方块 和 每根手指→方块 的方向向量点积,
    负值表示对立。要求双方都有接触力。
    R74: index/middle 权重=1.0, ring=0.3, pinky=0.0。
    """
    # tip_world shape: (5, 3) — [index, middle, ring, pinky, thumb]
    tip_to_cube = cube_pos[None, :] - tip_world  # (5, 3)
    tip_to_cube_norm = tip_to_cube / (jp.linalg.norm(tip_to_cube, axis=1, keepdims=True) + 1e-6)
    thumb_dir = tip_to_cube_norm[4]  # thumb direction to cube

    # Dot product of thumb direction with each finger direction
    finger_dirs = tip_to_cube_norm[:4]  # 4 fingers
    dots = jp.sum(finger_dirs * thumb_dir[None, :], axis=1)  # (4,)

    # R82: sigmoid替代硬clip，同侧(dots>0)也有梯度引导拇指移向对侧
    # clip(-dots,0,1): dots>0时=0且无梯度 → 策略无法学习移动拇指
    # sigmoid(-5*dots): dots=1→0.007, dots=0→0.5, dots=-1→0.993
    opposition = jax.nn.sigmoid(-5.0 * dots)  # (4,)

    # R78: 软化力门控 — sigmoid 代替硬阈值, 允许微力时也有梯度
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    finger_soft_active = jax.nn.sigmoid(30.0 * (abs_f[:4] - th * 0.5))
    thumb_soft_active = jax.nn.sigmoid(30.0 * (abs_f[4] - th * 0.5))
    # R84: 几何基线提升到50%，给approach阶段更强方向信号（R82=0.3）
    gated_opposition = opposition * (0.5 * finger_soft_active * thumb_soft_active + 0.5)

    # R74: Weighted by finger preference — index(1.0), middle(1.0), ring(0.3), pinky(0.0)
    finger_weights = jp.array([1.0, 1.0, 0.3, 0.0])
    weighted = gated_opposition * finger_weights
    return jp.max(weighted)

  def _reward_soft_contact(self, tip_force: jax.Array) -> jax.Array:
    fmin = self._config.reward_config.soft_contact_fmin
    fmax = self._config.reward_config.soft_contact_fmax
    abs_f = jp.abs(tip_force)
    lower_gate = jax.nn.sigmoid(20.0 * (abs_f - fmin))
    upper_gate = jax.nn.sigmoid(10.0 * (fmax - abs_f))
    return jp.mean(lower_gate * upper_gate)

  def _reward_primary_finger_force(self, tip_force: jax.Array) -> jax.Array:
    """R78: 拇指+食指+中指受力奖励 — 软化公式。

    tip_force 顺序: [index(0), middle(1), ring(2), pinky(3), thumb(4)]
    R74原版用 jp.min() 导致三指中任一无力就全零。
    R78: mean * min_bonus, 允许部分手指先接触也能获得梯度。
    min_bonus 鼓励三指均衡, 但不至于全零。
    """
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    sat = self._config.reward_config.force_contact_saturation
    # 三指力归一化: index(0), middle(1), thumb(4)
    primary_forces = jp.array([abs_f[0], abs_f[1], abs_f[4]])
    normalized = jp.clip((primary_forces - th) / (sat - th + 1e-6), 0.0, 1.0)
    # R81: 收紧 min_bonus 0.4→0.15 — 无拇指只拿15%(原40%), 强制三指参与
    mean_val = jp.mean(normalized)
    min_val = jp.min(normalized)
    min_bonus = jp.where(mean_val > 1e-6, 0.15 + 0.85 * min_val / (mean_val + 1e-6), 0.0)
    return mean_val * min_bonus

  def _reward_release_ready(self, tip_force: jax.Array) -> jax.Array:
    """R102: pre-release grasp readiness for the primary tip triad.

    Support removal is gated on index+middle+thumb tip forces.  Earlier rewards
    mostly activated after support release, which left too little signal for
    learning the release condition itself.
    """
    abs_f = jp.abs(tip_force)
    primary = jp.array([abs_f[0], abs_f[1], abs_f[4]])
    th = self._config.reward_config.finger_active_threshold
    sat = self._config.reward_config.force_contact_saturation
    soft_active = jax.nn.sigmoid(35.0 * (primary - th * 0.5))
    active_mean = jp.mean(soft_active)
    all_active = jp.min(soft_active)
    normalized = jp.clip((primary - th * 0.25) / (sat - th * 0.25 + 1e-6), 0.0, 1.0)
    force_mean = jp.mean(normalized)
    mean_f = jp.mean(primary)
    rel_std = jp.std(primary) / (mean_f + 1e-6)
    balance = jp.clip(1.0 - rel_std, 0.0, 1.0)
    return active_mean * (0.35 + 0.65 * all_active) * (0.5 + 0.5 * balance) * (0.4 + 0.6 * force_mean)

  def _reward_post_release_survival(
      self, cube_pos: jax.Array, cube_linvel: jax.Array,
  ) -> jax.Array:
    z_floor = self._spawn_z - 0.010
    z_safe = jp.clip((cube_pos[2] - z_floor) / 0.010, 0.0, 1.0)
    xy_err = jp.linalg.norm((cube_pos - self._spawn_pos)[:2])
    xy_safe = jp.exp(-30.0 * xy_err)
    slow = jp.exp(-4.0 * jp.linalg.norm(cube_linvel))
    return z_safe * xy_safe * slow

  def _reward_post_release_pose_hold(self, hand_q: jax.Array) -> jax.Array:
    primary_ids = jp.array([0, 1, 2, 3, 8, 9, 10])
    q = jp.take(hand_q, primary_ids)
    target = jp.take(self._pre_grasp_pose, primary_ids)
    err = jp.mean(jp.square(q - target))
    return jp.exp(-10.0 * err)

  def _reward_idle_follow(
      self, hand_q: jax.Array, tip_force: jax.Array,
  ) -> jax.Array:
    active_th = self._config.reward_config.finger_active_threshold
    mcp = jp.take(hand_q, jp.array(_V2_FINGER_MCP_IDS))
    f4 = jp.abs(tip_force[:4])
    is_active = (f4 > active_th).astype(jp.float32)
    n_active = jp.sum(is_active) + 1e-6
    ref_mcp = jp.dot(mcp, is_active) / n_active
    is_idle = 1.0 - is_active
    flex_gap = jp.maximum(ref_mcp - mcp, 0.0)
    follow_score = jp.exp(-5.0 * flex_gap)
    n_idle = jp.sum(is_idle) + 1e-6
    idle_score = jp.dot(is_idle, follow_score) / n_idle
    has_contact = jp.clip(n_active - 1.0, 0.0, 1.0)
    return idle_score * has_contact

  def _reward_height(
      self, cube_pos: jax.Array, palm_pos: jax.Array,
  ) -> jax.Array:
    # R93: spawn_z 处为0分；几何扫描显示+20mm无三指接触解，
    # 因此目标使用可达的约12mm离台高度。
    target_lift = getattr(self._config.reward_config, 'target_lift_m', 0.012)
    lift = jp.clip((cube_pos[2] - self._spawn_z) / target_lift, 0.0, 1.0)
    xy_err = jp.linalg.norm((cube_pos - palm_pos)[:2])
    palm_alignment = jp.exp(-40.0 * jp.square(xy_err))
    return lift * palm_alignment

  # ── 惩罚项 ───────────────────────────────────────────────────────────────

  def _cost_drop_risk(
      self, cube_pos: jax.Array, cube_linvel: jax.Array,
  ) -> jax.Array:
    drop_ref = self._spawn_z
    low_risk = jp.clip(
        (drop_ref - cube_pos[2]) / jp.maximum(drop_ref, 0.005), 0.0, 1.0,
    )
    down_risk = jp.clip((-cube_linvel[2]) / 0.25, 0.0, 1.0)
    return low_risk * (0.4 + 0.6 * down_risk)

  def _cost_post_release_slip(self, cube_linvel: jax.Array) -> jax.Array:
    down = jp.clip((-cube_linvel[2]) / 0.06, 0.0, 2.0)
    lateral = jp.clip(jp.linalg.norm(cube_linvel[:2]) / 0.12, 0.0, 2.0)
    return down + 0.5 * lateral

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array,
  ) -> jax.Array:
    return jp.sum(jp.square(act - last_act))

  def _cost_action_accel(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array,
  ) -> jax.Array:
    jerk = act - 2.0 * last_act + last_last_act
    return jp.sum(jp.square(jerk))

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torques))

  def _cost_force_overload(self, tip_force: jax.Array) -> jax.Array:
    maxf = self._config.reward_config.force_overload_threshold
    soft_w = self._config.reward_config.force_overload_soft_width
    abs_f = jp.abs(tip_force)
    overload = jp.maximum(abs_f - maxf, 0.0)
    cost = jp.square(overload) + jp.power(overload / (soft_w + 1e-6), 3.0)
    return jp.mean(cost)


# ── domain randomization ──────────────────────────────────────────────────

class CubeGraspV2ForceCoacd(CubeGraspV2Force):
  """Experimental V2 grasp task using CoACD convex mesh collision geometry."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=xml_path or consts.GRASP_V2_COACD_XML.as_posix(),
    )


class CubeGraspV2ForceCoacdQbr(CubeGraspV2Force):
  """Cube grasp task using the current QBR-style thumb flex coupling."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config_qbr(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=xml_path or consts.GRASP_V2_COACD_QBR_XML.as_posix(),
    )
    middle_mcp_id = _V2_FINGER_MCP_IDS[1]
    middle_target = 0.62
    thumb_equiv_mcp = float((0.16 * 1.3788) / 0.6666667)
    self._thumb_passive_source_id = _V2_THUMB_MCP_ID
    self._thumb_passive_ratio = 0.6666667
    self._default_ctrl = self._default_ctrl.at[1].set(thumb_equiv_mcp)
    self._default_ctrl = self._default_ctrl.at[3].set(middle_target)
    self._default_pose = self._default_pose.at[middle_mcp_id].set(middle_target)
    self._pre_grasp_pose = self._pre_grasp_pose.at[_V2_THUMB_MCP_ID].set(
        thumb_equiv_mcp
    )
    self._pre_grasp_pose = self._pre_grasp_pose.at[middle_mcp_id].set(
        middle_target
    )
    self._lifted_grasp_pose = self._lifted_grasp_pose.at[_V2_THUMB_MCP_ID].set(
        thumb_equiv_mcp
    )
    self._lifted_grasp_pose = self._lifted_grasp_pose.at[middle_mcp_id].set(
        middle_target
    )
    self._lifted_grasp_ctrl = self._lifted_grasp_ctrl.at[1].set(thumb_equiv_mcp)
    self._lifted_grasp_ctrl = self._lifted_grasp_ctrl.at[3].set(middle_target)
    self._default_pose = self._apply_passive_joint_couplings(self._default_pose)
    self._pre_grasp_pose = self._apply_passive_joint_couplings(self._pre_grasp_pose)
    self._lifted_grasp_pose = self._apply_passive_joint_couplings(
        self._lifted_grasp_pose
    )


class CubeGraspV2ForceCapsuleBottlePalmQbr(CubeGraspV2ForceCoacdQbr):
  """QBR cube policy evaluated on the current capsule hand with bottle palm fit."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config_qbr(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    config = config.copy_and_resolve_references()
    # C63/C69: keep the clean proxy-env triad shaping and the stronger
    # post-release grasp retention that pushed the bottle-palm proxy into the
    # high-20s. C86 showed extra pose-hold weight hurts trainability, so keep
    # the C84 timing line intact and move the next probe to the optimizer.
    config.reward_config.scales.three_finger_proximity = 18.0
    config.reward_config.scales.primary_finger_force = 72.0
    config.reward_config.scales.post_release_grasp = 135.0  # C149: restore C146 reward, lower continuation LR instead
    config.support_config.min_release_active_fingers = 2
    config.reward_config.scales.pre_release_grasp = 35.0
    config.support_config.random_release_min_sec = 1.5
    config.support_config.random_release_max_sec = 2.45
    config.support_config.release_ramp_sec = 0.5
    config.support_config.force_release_after_sec = 3.2
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=xml_path or consts.GRASP_V2_CAPSULE_BOTTLEPALM_CUBE_XML.as_posix(),
    )
    # C58: the current bottle-palm proxy recovered from 2.9s to 20s+ mostly
    # through continuation. Keep the proven middle bias at 0.66 here; the
    # later 0.68 probe stayed clean but underperformed and is not the mainline.
    middle_mcp_id = _V2_FINGER_MCP_IDS[1]
    middle_target = 0.66
    self._default_ctrl = self._default_ctrl.at[3].set(middle_target)
    self._default_pose = self._default_pose.at[middle_mcp_id].set(middle_target)
    self._pre_grasp_pose = self._pre_grasp_pose.at[middle_mcp_id].set(
        middle_target
    )
    self._lifted_grasp_pose = self._lifted_grasp_pose.at[middle_mcp_id].set(
        middle_target
    )
    self._lifted_grasp_ctrl = self._lifted_grasp_ctrl.at[3].set(middle_target)
    self._default_pose = self._apply_passive_joint_couplings(self._default_pose)
    self._pre_grasp_pose = self._apply_passive_joint_couplings(self._pre_grasp_pose)
    self._lifted_grasp_pose = self._apply_passive_joint_couplings(
        self._lifted_grasp_pose
    )


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """V2 灵犀手 domain randomization。"""
  mj_model = CubeGraspV2Force().mj_model
  cube_geom_id = mj_model.geom("cube").id
  cube_body_id = mj_model.body("cube").id
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
  fingertip_geoms = []
  for geom_names in consts.V2_FINGER_CONTACT_GEOMS.values():
    fingertip_geoms.extend(geom_names)
  fingertip_geom_ids = [mj_model.geom(g).id for g in fingertip_geoms]

  nq_hand = consts.V2_NQ

  @jax.vmap
  def rand(rng):
    rng, key = jax.random.split(rng)
    cube_friction = jax.random.uniform(key, (1,), minval=1.0, maxval=2.0)  # C22: 围绕XML标称1.5随机化
    geom_friction = model.geom_friction.at[cube_geom_id : cube_geom_id + 1, 0].set(
        cube_friction,
    )

    fingertip_friction = jax.random.uniform(key, (1,), minval=1.0, maxval=2.0)  # C22: 围绕XML标称1.5随机化
    geom_friction = geom_friction.at[fingertip_geom_ids, 0].set(fingertip_friction)

    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
    cube_mass = model.body_mass[cube_body_id]
    body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
    body_inertia = model.body_inertia.at[cube_body_id].set(
        model.body_inertia[cube_body_id] * dmass,
    )
    dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(
        model.body_ipos[cube_body_id] + dpos,
    )

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
        qpos0[hand_qids]
        + jax.random.uniform(key, shape=(nq_hand,), minval=-0.03, maxval=0.03),
    )

    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=0.8, maxval=1.2,
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(nq_hand,), minval=1.0, maxval=1.05,
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    rng, key = jax.random.split(rng)
    dmass_hand = jax.random.uniform(
        key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1,
    )
    body_mass = body_mass.at[hand_body_ids].set(
        model.body_mass[hand_body_ids] * dmass_hand,
    )

    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
        key, (model.nu,), minval=0.9, maxval=1.1,
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    rng, key = jax.random.split(rng)
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
