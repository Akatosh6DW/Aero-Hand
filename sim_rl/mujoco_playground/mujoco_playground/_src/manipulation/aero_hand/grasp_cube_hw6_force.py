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
"""HW6 force-aware cube grasp task for TetherIA Aero Hand Open."""

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
      # 6 channels in hardware order:
      # [thumb_rot, thumb_flex, index, middle, ring, pinky]
      # scale 设计原则：PPO 使用 tanh_normal 分布，action ∈ (-1, +1)。
      # action_scale 须保证 action=±0.9 可覆盖抓取所需的全行程。
      # thumb_rot: 抓取需 ~1.20 rad → scale=1.3 (act=0.92→ctrl=1.20)
      # thumb_flex: tendon 全程 0.030 → scale=0.030 (act=-0.93→min)
      # index~pinky: tendon 全程 0.052 → scale=0.055 (act=-0.88→min)
      action_scale=[1.3, 0.030, 0.055, 0.055, 0.055, 0.055],
      action_repeat=1,
      episode_length=500,
      early_termination=True,
      history_len=1,
      noise_config=config_dict.create(
          level=0.3,
          force_ema_alpha=0.85,
          scales=config_dict.create(
              hw_pos=0.005,
              hw_force=0.05,
          ),
      ),
        tactile_config=config_dict.create(
          # 将每指 16 路 taxel 做加权平均后再输入策略，降低高维噪声敏感性。
          use_pooled_obs=True,
          # 4x4 权重（行优先）：中心区域更高权重，边缘和角落更低权重。
          taxel_weights=[
            0.7, 1.0, 1.0, 0.7,
            1.0, 1.4, 1.4, 1.0,
            1.0, 1.4, 1.4, 1.0,
            0.7, 1.0, 1.0, 0.7,
          ],
          # ── 实时触觉观测配置 ──────────────────────────────────
          # use_real_tactile=True 时，将 efc_force 提取的每指接触力
          # 经 EMA 滤波 + 饱和归一化后注入策略观测的 5D 触觉位。
          # 默认 False 以兼容旧 checkpoint（V-iter30 的 normalizer
          # 力度维度 mean≈0, std≈0.01，注入真实值会导致归一化后 50x 越界）。
          use_real_tactile=False,
          # 力饱和值(N)：efc 力归一化到 [0,1]，超过此值截断为 1。
          # 3N 与硬件传感器量程对齐。
          force_saturation_n=3.0,
          # 观测力 EMA 系数：f_ema = alpha * f_prev + (1-alpha) * f_new。
          # 0.7 → 约 3 步时间常数，平滑高频接触噪声，保留握力趋势。
          obs_force_ema_alpha=0.7,
        ),
      support_config=config_dict.create(
          release_after_sec=2.2,
          release_ramp_sec=0.0,  # 渐进释放时长(秒)；0=瞬间释放
          support_pos=[-0.066, 0.0, 0.067],
          support_hidden_pos=[0.0, 0.0, -10.0],
          min_release_active_fingers=3,
          min_release_force=0.20,
      ),
      # 为提升可读性，移除 A/B/C 课程采样，改为显式初始位姿配置。
      spawn_config=config_dict.create(
          cube_pos=[-0.050, 0.0, 0.080],
          cube_jitter=[0.0025, 0.0025, 0.0010],
          support_enabled=True,
      ),
        reset_config=config_dict.create(
            # 0.0=home keyframe 姿态；当前 home 已设置为静态伸直。
            hand_open_ratio=0.0,
            # HW6 当前模型拇指两路方向不一致：
            # - 外展(thumb_abd): 下限更接近张开
            # - 弯曲(thumb_flex): 上限更接近张开
            thumb_abd_open_with_lower=True,
            thumb_flex_open_with_lower=False,
          # 初始手型轻微噪声，避免策略过拟合唯一姿态。
          hand_qpos_noise_scale=0.02,
        ),
      reward_config=config_dict.create(
          scales=config_dict.create(
            # ===== 阶段1核心：靠近 → 接触 → 握住不掉 =====
            approach=5.0,        # 稠密引导：最近指尖靠近方块
            contact=2.5,         # 进入接触区额外奖励
            thumb_engage=3.0,    # 拇指单独引导
            closure=3.0,         # 四指整体弯曲（contact_gate门控）
            distal_closure=3.0,  # PIP/DIP远端弯曲，防止"只用MCP托起"（contact_gate门控）
            # ===== 核心目标：稳定持握 =====
            stable_hold=7.0,     # 主目标：≥2指受力 + 方块低速度（最高权重）
            force_contact=2.5,   # 真实力传感器信号正反馈
            hold_position=2.0,   # 方块保持在初始高度（contact_gate门控）
            # ===== 阶段2预留：空闲手指跟随接触手指姿态 =====
            # 激活时：无接触手指MCP向受力手指均值靠拢；无接触时=0防退化
            idle_follow=0.0,
            # ===== 接口保留（阶段2可激活） =====
            finger_synergy=0.0,
            closure_uniformity=0.0,
            height=0.0,
            multi_finger=0.0,
            all_fingers_close=0.0,
            ulnar_engage=0.0,
            force_balance=0.0,
            finger_participation=0.0,
            single_finger_dominance=0.0,
            drop_risk=0.0,
            lift_without_wrap=0.0,
            pinky_missing_on_lift=0.0,
            no_wrap_timeout=0.0,
            # ===== 人手姿态模仿 =====
            human_pose=0.0,
            # ===== 控制约束 =====
            survival=0.05,
            termination=-150.0,
            action_rate=-0.04,
            action_accel=-0.03,  # 略微加大：防止动作突变打飞方块
            torques=-0.001,
            force_overload=-1.0,  # 加大：防止夹碎
            # ===== soft_contact: 安全力区间内持续正反馈 [RMA-prep] =====
            soft_contact=0.0,    # F_min~F_max 区间内给予 bell-shaped 奖励
          ),
            # 指尖力接触奖励起始阈值（单位近似 N）
            force_contact_threshold=0.08,
            # 指尖力奖励饱和点（按 0~3N 量级设置）
            force_contact_saturation=1.2,
            # 指尖力过载阈值（接近 3N 上限）
            force_overload_threshold=2.8,
            # 过载惩罚的"软启动"宽度 (N)，控制 quadratic→cubic 平滑过渡
            force_overload_soft_width=1.5,
            # soft_contact 安全区间下限 (N)：低于此值视为无效接触
            soft_contact_fmin=0.1,
            # soft_contact 安全区间上限 (N)：高于此值开始衰减
            soft_contact_fmax=2.5,
            # 判定某手指“参与受力”的阈值
            finger_active_threshold=0.15,
            # 单指受力占比超过该值后开始惩罚
            single_finger_share_threshold=0.55,
            # 判定“已抬起”的高度阈值（相对当前场景经验值）
            lifted_height_threshold=0.018,
            # 认为“形成包裹”的最小激活手指数
            wrap_active_fingers_threshold=4,
            # 认为“形成包裹”的最小总受力
            wrap_force_threshold=0.38,
            # 连续托举但未包裹超过该时长后，开始加重惩罚
            no_wrap_timeout_sec=0.80,
      ),
  )


def default_config_p() -> config_dict.ConfigDict:
  """P 阶段配置：手心向上（palmup），方块悬于掌上方，由支撑台托住。

  [Round 6 根本修复] FK 分析确认：
    - 所有手指关节上限 [0°, 90°]
    - MCP=90°, PIP=60° 时指尖在 z≈+0.020（世界坐标）
    - 之前 cube_pos.z=-0.022（掌面内部），指尖永远在 cube 上方，几何上不可能接触
    - 修复：cube 移至 z=+0.015，支撑台 z=-0.003 托住方块，启用 support_enabled=True
    - MCP=90°, PIP=60° 时 mf_tip 距方块中心 0.88cm < 1.5cm（可接触！）
  """
  cfg = default_config()
  # ── P-stage 物块初始位置（掌心朝上 palmup 场景，世界坐标）────────────────
  # [Round 6 修复] cube 从 z=-0.022（掌面内部）移到 z=+0.015（掌面上方，指尖可达）
  #   z=+0.015: 方块底面在 z=0.000，恰好在 grasp_site 参考面以上
  #   FK验证：MCP=90°,PIP=60° → mf_tip=(0.102,0.001,0.020)，距方块中心0.88cm < 1.5cm ✓
  cfg.spawn_config.cube_pos = [0.100, 0.008, 0.015]
  cfg.spawn_config.cube_jitter = [0.002, 0.005, 0.005]
  # R6: 启用支撑台，物理托住方块（支撑台 z=-0.003，方块底面 z=0.000）
  cfg.spawn_config.support_enabled = True
  cfg.support_config.support_pos = [0.100, 0.008, -0.003]
  # [R12修复] 恢复默认 2.2s：Bug#3使 grasp_ready 永远=False，9s 导致支撑台直到 18s(step360)才强制释放
  # 2.2s → 强制释放在 4.4s(step88)，episode 后 412 步手指有真实重力感知
  cfg.support_config.release_after_sec = 2.2

  # ── 奖励权重 ─────────────────────────────────────────────────────────────
  # [Round 12] 物理碰撞已修复(Bug#2)，重新聚焦可用梯度：
  #   去掉 force_contact/stable_hold（Bug#3: MJX touch sensor=0，永远无效）
  #   approach 维持禁用（Bug#1: MCP≈45° 局部最优）
  #   加强 closure+distal：物理碰撞生效后，弯曲动作有真实的 cube 阻力反馈
  #   提升 contact 权重：SDF proxy 是当前唯一可靠的接触方向信号
  cfg.reward_config.scales.approach = 0.0         # 维持禁用（MCP≈45° 局部最优）
  cfg.reward_config.scales.contact = 12.0         # [R12] 主导信号：surface SDF 引导
  cfg.reward_config.scales.thumb_engage = 2.0     # 保持
  cfg.reward_config.scales.closure = 6.0          # [R12] 提升：物理碰撞后 MCP 阻力真实
  cfg.reward_config.scales.distal_closure = 6.0   # [R12] 提升：PIP→60° 包裹
  cfg.reward_config.scales.all_fingers_close = 0.0
  cfg.reward_config.scales.stable_hold = 0.0      # [R12] 禁用：MJX touch sensor=0
  cfg.reward_config.scales.force_contact = 0.0    # [R12] 禁用：MJX touch sensor=0
  cfg.reward_config.scales.hold_position = 0.0    # 暂不启用
  cfg.reward_config.scales.termination = -20.0
  return cfg


def default_config_v() -> config_dict.ConfigDict:
  """V 阶段配置：竖直抓取场景，从 P checkpoint 继续微调。

  任务描述：手竖直（举手姿势），掌心朝向方块，支撑台托住方块 2.2s。
  支撑台释放后，手必须靠手指夹持力对抗重力，保持方块不落地。

  奖励设计原则：
    - hold_position 是唯一直接测量"是否真正夹住"的奖励，权重最高，始终部分激活。
    - contact/closure/distal_closure 提供收敛前的密集梯度信号。
    - MJX Bug#3：touch sensor sensordata 全为零，改用 efc_force 提取接触力。
    - [V-iter25] 力反馈已引入观测（手动缩放 EFC 力到 [0,1]）。
    - torques（data.actuator_force）在 MJX 中可用，适度加强作为防暴力夹持代理。
  """
  cfg = default_config()
  # [V-iter11] 方块位置修正：几何分析发现旧位置 [-0.050,0,0.080] 拇指完全不可达
  # （thumb_rot=max 时 thumb_distal.y=-0.061 vs cube.y=0）。
  # 新位置 [-0.072,-0.010,0.072] 经 MuJoCo 接触扫描验证：
  #   thumb_rot≈1.2~1.3 + 全闭合 → 食指(IF)+中指(MF)+拇指(TH)+手掌 同时接触。
  # [V-iter31] 方块从Y=-0.010(偏拇指)移至Y=0.005(掌心中央)
  # 几何分析：4指闭合Y中心=0.012，拇指Y=0.006，中点≈0.009；取0.005偏保守
  cfg.spawn_config.cube_pos = [-0.072, 0.005, 0.072]
  cfg.spawn_config.cube_jitter = [0.004, 0.005, 0.003]  # [V-iter24] 回退±5mm(±6mm三次均失败), 加cube size rand
  cfg.spawn_config.support_enabled = True
  cfg.support_config.support_pos = [-0.072, 0.005, 0.054]
  # V 阶段：支撑 4.0s 后开始渐进释放。
  # [V-iter29] ramp 1.5s / 5cm（V-iter27: 1.0s太快→成功/失败边界太尖锐→震荡）。
  cfg.support_config.release_after_sec = 4.0
  cfg.support_config.release_ramp_sec = 1.5
  cfg.support_config.min_release_active_fingers = 2
  cfg.support_config.min_release_force = 0.15

  # ── V-stage 奖励权重（V-iter25：3D hold + force obs + always-active） ──
  # [V-iter25] 三项关键修复:
  #   1) hold_position: Z-only→3D（旧版方块可横向飞走无惩罚）
  #   2) hold_position: 始终部分激活（support阶段20%，release后100%）
  #   3) 观测引入 EFC 力反馈（旧版 tactile_obs 始终为零）
  # [V-iter18/19] force_contact=1.0 突破: avg~1000, best=1143
  # [V-iter20] stable_hold=2.0 确认有害: 1036→200，与lifting根本冲突
  # [V-iter21/22/23] ±6mm Y jitter 三次均失败(drift/crash), 回退±5mm
  cfg.reward_config.scales.approach = 0.0        # 关闭：新位置已在工作空间内
  cfg.reward_config.scales.contact = 5.0         # [V-iter16] 与 V-iter13 完全相同
  cfg.reward_config.scales.thumb_engage = 8.0    # 拇指对握核心
  cfg.reward_config.scales.closure = 5.0         # 手指闭合
  cfg.reward_config.scales.distal_closure = 5.0  # PIP/DIP 包裹
  cfg.reward_config.scales.all_fingers_close = 0.0
  cfg.reward_config.scales.stable_hold = 0.0     # [V-iter21] 禁用: V-iter17(5.0)和V-iter20(2.0)均退化，与lifting根本冲突
  cfg.reward_config.scales.force_contact = 5.0    # [V-iter29] 提高至5.0（加强支撑阶段握力信号）
  cfg.reward_config.scales.force_overload = 0.0  # 暂不启用
  cfg.reward_config.scales.hold_position = 120.0 # 核心保持奖励
  cfg.reward_config.scales.human_pose = 5.0      # [V-iter16] 与 V-iter13 完全相同
  cfg.reward_config.scales.grip_force = 3.0      # [V-iter29] 启用对握力奖励（contact_gate门控）
  cfg.reward_config.scales.drop_risk = -25.0     # 跌落早期惩罚
  cfg.reward_config.scales.torques = -0.00003    # 微调
  cfg.reward_config.scales.action_rate = -0.01   # 适度平滑
  cfg.reward_config.scales.action_accel = -0.008 # 适度平滑
  cfg.reward_config.scales.survival = 0.5        # [V-iter16] 与 V-iter13 完全相同
  cfg.reward_config.scales.termination = -120.0  # [V-iter27] 更强终止惩罚（配合3cm阈值）
  return cfg


class CubeGraspHW6Force(aero_hand_base.AeroHandEnv):
  """Cube grasp task with 6-channel hardware action/observation semantics."""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
      xml_path: Optional[str] = None,
  ):
    # 允许在不改变动作/观测维度的前提下切换场景 XML，保证 checkpoint 可复用。
    selected_xml = xml_path or consts.GRASP_HW6_XML.as_posix()
    super().__init__(
      xml_path=selected_xml,
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._hand_qids = mjx_env.get_qpos_ids(self.mj_model, consts.JOINT_NAMES)
    self._hand_dqids = mjx_env.get_qvel_ids(self.mj_model, consts.JOINT_NAMES)

    self._support_body_id = self._mj_model.body("cube_support").id
    self._support_mocap_id = int(self._mj_model.body_mocapid[self._support_body_id])
    if self._support_mocap_id < 0:
      raise ValueError("cube_support must be a mocap body.")

    self._support_pos = jp.array(self._config.support_config.support_pos)
    self._support_hidden_pos = jp.array(self._config.support_config.support_hidden_pos)
    self._support_release_steps = max(
        1,
        int(np.round(self._config.support_config.release_after_sec / self.dt)),
    )
    ramp_sec = getattr(self._config.support_config, 'release_ramp_sec', 0.0)
    self._support_ramp_steps = max(
        0, int(np.round(ramp_sec / self.dt)),
    )

    home_key = self._mj_model.keyframe("home")
    self._init_q = jp.array(home_key.qpos)
    home_hand_pose = self._init_q[self._hand_qids]
    lowers_np, uppers_np = self.mj_model.jnt_range[self._hand_qids].T
    self._lowers = jp.array(lowers_np, dtype=jp.float32)
    self._uppers = jp.array(uppers_np, dtype=jp.float32)

    # 构造“张开手型”目标：四指取上限，拇指按外展/弯曲分别处理。
    open_pose = self._uppers
    thumb_abd_open_with_lower = bool(self._config.reset_config.thumb_abd_open_with_lower)
    thumb_flex_open_with_lower = bool(self._config.reset_config.thumb_flex_open_with_lower)
    open_pose = open_pose.at[12].set(self._lowers[12] if thumb_abd_open_with_lower else self._uppers[12])
    thumb_flex_open_value = self._lowers[13] if thumb_flex_open_with_lower else self._uppers[13]
    open_pose = open_pose.at[13:16].set(thumb_flex_open_value)

    open_ratio = jp.array(self._config.reset_config.hand_open_ratio, dtype=jp.float32)
    self._default_pose = home_hand_pose + open_ratio * (open_pose - home_hand_pose)

    # 控制基线也对齐“张开态”，避免 reset 后第一步就因基线偏闭合而挤压方块。
    home_ctrl = jp.array(home_key.ctrl)
    ctrl_lowers = jp.array(self.mj_model.actuator_ctrlrange[:, 0], dtype=jp.float32)
    ctrl_uppers = jp.array(self.mj_model.actuator_ctrlrange[:, 1], dtype=jp.float32)
    open_ctrl = home_ctrl.at[1:].set(ctrl_uppers[1:])
    # HW6 六通道顺序: [thumb_rot, thumb_flex, index, middle, ring, pinky]
    open_ctrl = open_ctrl.at[0].set(ctrl_lowers[0] if thumb_abd_open_with_lower else ctrl_uppers[0])
    open_ctrl = open_ctrl.at[1].set(ctrl_lowers[1] if thumb_flex_open_with_lower else ctrl_uppers[1])
    self._default_tendon = home_ctrl + open_ratio * (open_ctrl - home_ctrl)
    self._spawn_z = jp.array(self._config.spawn_config.cube_pos[2], dtype=jp.float32)
    self._spawn_pos = jp.array(self._config.spawn_config.cube_pos, dtype=jp.float32)
    self._taxel_weights = self._build_taxel_weights()

    # Pre-compute geom IDs for efc_force-based contact force extraction.
    # Touch sensors have 0.8mm cutoff radius → always zero; use efc_force instead.
    self._fingertip_geom_ids_list = [
        int(self.mj_model.geom(n).id) for n in consts.FINGERTIP_NAMES
    ]  # [if_tip, mf_tip, rf_tip, pf_tip, th_tip]
    self._cube_geom_id = int(self.mj_model.geom('cube').id)

  def _build_taxel_weights(self) -> jax.Array:
    weights = jp.array(self._config.tactile_config.taxel_weights, dtype=jp.float32)
    if weights.shape[0] != 16:
      raise ValueError("tactile_config.taxel_weights must contain exactly 16 values.")
    return weights / (jp.sum(weights) + 1e-6)

  def _get_tactile_obs_dim(self) -> int:
    if bool(self._config.tactile_config.use_pooled_obs):
      return 5
    return len(consts.SENSOR_HW6_FORCE_NAMES)

  def reset(self, rng: jax.Array) -> mjx_env.State:
    rng, pos_rng, vel_rng = jax.random.split(rng, 3)
    hand_noise_scale = self._config.reset_config.hand_qpos_noise_scale
    q_hand = jp.clip(
      self._default_pose + hand_noise_scale * jax.random.normal(pos_rng, (consts.NQ,)),
        self._lowers,
        self._uppers,
    )
    v_hand = 0.0 * jax.random.normal(vel_rng, (consts.NV,))

    rng, p_rng, quat_rng = jax.random.split(rng, 3)
    base_pos = jp.array(self._config.spawn_config.cube_pos, dtype=jp.float32)
    jitter = jp.array(self._config.spawn_config.cube_jitter, dtype=jp.float32)
    start_pos = base_pos + jax.random.uniform(
      p_rng,
      (3,),
      minval=-jitter,
      maxval=jitter,
    )
    del quat_rng
    start_quat = jp.array([1.0, 0.0, 0.0, 0.0])
    q_cube = jp.array([*start_pos, *start_quat])
    v_cube = jp.zeros(6)

    qpos = jp.concatenate([q_hand, q_cube])
    qvel = jp.concatenate([v_hand, v_cube])
    mocap_pos = self._initial_mocap_pos()
    use_support = jp.array(bool(self._config.spawn_config.support_enabled))
    support_init = jp.where(use_support, self._support_pos, self._support_hidden_pos)
    mocap_pos = mocap_pos.at[self._support_mocap_id].set(support_init)

    data = mjx_env.make_data(
        self.mj_model,
        qpos=qpos,
        qvel=qvel,
        ctrl=self._default_tendon,
        mocap_pos=mocap_pos,
    )

    info = {
        "rng": rng,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": data.ctrl,
        "last_cube_angvel": jp.zeros(3),
      "filtered_hw_force": jp.zeros((len(consts.SENSOR_HW6_FORCE_NAMES),), dtype=jp.float32),
      "filtered_pooled_force": jp.zeros((5,), dtype=jp.float32),
      "ema_tip_finger_forces": jp.zeros((5,), dtype=jp.float32),
      "obs_tactile_ema": jp.zeros((5,), dtype=jp.float32),
        "no_wrap_lift_steps": jp.array(0, dtype=jp.int32),
        "support_released": jp.array(False),
        "support_timer": jp.array(0, dtype=jp.int32),
        "use_support": use_support,
    }

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())

    # 观测语义对齐硬件：6 维逻辑驱动角 + 80 维(5*16)指尖触觉 + 6 维上一时刻动作。
    state_dim = self.mjx_model.nu + self._get_tactile_obs_dim() + self.mjx_model.nu
    obs_history = jp.zeros(self._config.history_len * state_dim)
    obs = self._get_obs(data, info, obs_history)
    reward, done = jp.zeros(2)
    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    action_scale_custom = jp.array(self._config.action_scale, dtype=jp.float32)
    motor_targets = self._default_tendon + action * action_scale_custom
    data = mjx_env.step(self.mjx_model, state.data, motor_targets, self.n_substeps)

    tip_finger_forces = self._get_contact_forces_efc(data)

    support_timer = state.info["support_timer"] + 1
    ema_tip_finger_forces = (
        0.8 * state.info["ema_tip_finger_forces"] + 0.2 * tip_finger_forces
    )
    state.info["ema_tip_finger_forces"] = ema_tip_finger_forces

    # 单独维护的观测用触觉 EMA（alpha 可独立配置，用于策略输入）。
    obs_alpha = float(self._config.tactile_config.obs_force_ema_alpha)
    obs_tactile_ema = (
        obs_alpha * state.info["obs_tactile_ema"]
        + (1.0 - obs_alpha) * tip_finger_forces
    )
    state.info["obs_tactile_ema"] = obs_tactile_ema

    cube_pos = self.get_cube_position(data)
    no_wrap_lift_flag = self._is_lift_without_wrap(cube_pos, ema_tip_finger_forces)
    no_wrap_lift_steps = jp.where(
      no_wrap_lift_flag,
      state.info["no_wrap_lift_steps"] + 1,
      jp.array(0, dtype=jp.int32),
    )
    state.info["no_wrap_lift_steps"] = no_wrap_lift_steps

    support_released = self._should_release_support(
      state.info["support_released"], support_timer, ema_tip_finger_forces
    )
    # 仅在启用支撑台的场景里生效；否则始终隐藏支撑台。
    data = jax.lax.cond(
      state.info["use_support"],
      lambda d: self._set_support_state(d, support_released, support_timer),
      lambda d: self._set_support_state(d, jp.array(True), support_timer),
      data,
    )

    state.info["motor_targets"] = motor_targets
    state.info["support_released"] = support_released
    state.info["support_timer"] = support_timer

    obs = self._get_obs(data, state.info, state.obs["state"])
    done = self._get_termination(data)

    rewards = self._get_reward(data, action, state.info, state.metrics, done)
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["last_cube_angvel"] = self.get_cube_angvel(data)
    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v

    done = done.astype(reward.dtype)
    return state.replace(data=data, obs=obs, reward=reward, done=done)

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
      self,
      already_released: jax.Array,
      support_timer: jax.Array,
      tip_finger_forces: jax.Array,
  ) -> jax.Array:
    # MJX 中触觉传感器始终返回零（已知 bug），因此改为纯定时释放。
    release_now = support_timer >= self._support_release_steps
    return jp.logical_or(already_released, release_now)

  def _is_grasp_ready_for_release(self, tip_finger_forces: jax.Array) -> jax.Array:
    active_th = self._config.reward_config.finger_active_threshold
    active_count = jp.sum((tip_finger_forces > active_th).astype(jp.int32))
    force_ok = jp.sum(tip_finger_forces) >= self._config.support_config.min_release_force
    finger_ok = active_count >= self._config.support_config.min_release_active_fingers
    return jp.logical_and(force_ok, finger_ok)

  def _set_support_state(
      self, data: mjx.Data, support_released: jax.Array,
      support_timer: jax.Array = None,
  ) -> mjx.Data:
    if self.mj_model.nmocap == 0:
      return data
    if support_timer is not None and self._support_ramp_steps > 0:
      # 渐进释放：support_pos 的 z 在 ramp_steps 内线性下降 5cm（足以脱离接触）。
      # ramp 结束后瞬间移到 hidden_pos。
      elapsed = jp.clip(
          (support_timer - self._support_release_steps).astype(jp.float32),
          0.0, float(self._support_ramp_steps),
      )
      alpha = elapsed / float(self._support_ramp_steps)
      ramp_done = alpha >= 1.0
      # 仅 z 轴下移 0.05m，x/y 不变。
      ramp_target = self._support_pos.at[2].set(self._support_pos[2] - 0.05)
      ramped_pos = (1.0 - alpha) * self._support_pos + alpha * ramp_target
      # ramp 结束后瞬移到远处。
      ramped_pos = jp.where(ramp_done, self._support_hidden_pos, ramped_pos)
      support_pos = jp.where(support_released, ramped_pos, self._support_pos)
    else:
      support_pos = jp.where(support_released, self._support_hidden_pos, self._support_pos)
    mocap_pos = data.mocap_pos.at[self._support_mocap_id].set(support_pos)
    return data.replace(mocap_pos=mocap_pos)

  def _get_termination(self, data: mjx.Data) -> jax.Array:
    cube_z = self.get_cube_position(data)[2]
    spawn_z = jp.array(self._config.spawn_config.cube_pos[2], dtype=jp.float32)
    # 使用与初始高度相关的掉落终止阈值，避免“掉了很久还不终止”。
    # 注意：不加 max(0,...) 截断，spawn_z 可能为负（P-stage palmup = -0.022），
    # 直接相对偏移确保方块下落 4cm 才终止，而非立即触发。
    # [V-iter27] 3cm 阈值（V-iter26: 6cm 太宽容，方块可落很远才终止）。
    drop_z = spawn_z - 0.03
    return cube_z < drop_z

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], obs_history: jax.Array
  ) -> Dict[str, jax.Array]:
    info["rng"], noise_rng = jax.random.split(info["rng"])

    # 位置反馈改为“底层电机逻辑驱动角”，直接采用当前控制目标 data.ctrl。
    hw_pos = jp.asarray(info["motor_targets"], dtype=jp.float32)

    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_hw_pos = (
        hw_pos
        + (2 * jax.random.uniform(noise_rng, shape=hw_pos.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.hw_pos
    )

    hw_force = jp.zeros((len(consts.SENSOR_HW6_FORCE_NAMES),), dtype=jp.float32)

    # ── 触觉观测构建 ──────────────────────────────────────────────────────
    use_real = bool(self._config.tactile_config.use_real_tactile)
    if use_real:
      # 使用 efc_force 提取的每指接触力，经 EMA 滤波 + 饱和归一化。
      # EMA 在 step() 中更新（obs_tactile_ema），此处只做归一化。
      sat = float(self._config.tactile_config.force_saturation_n)
      raw_force = jp.asarray(info["obs_tactile_ema"], dtype=jp.float32)
      tactile_obs = jp.clip(raw_force / sat, 0.0, 1.0)
      # 添加观测噪声（与 hw_pos 相同机制）
      info["rng"], force_noise_rng = jax.random.split(info["rng"])
      tactile_obs = tactile_obs + (
          (2 * jax.random.uniform(force_noise_rng, shape=tactile_obs.shape) - 1)
          * self._config.noise_config.level
          * self._config.noise_config.scales.hw_force
      )
      tactile_obs = jp.clip(tactile_obs, 0.0, 1.0)
    else:
      # 兼容旧 checkpoint：触觉位填零。
      tactile_obs = jp.zeros(5, dtype=jp.float32)

    # Keep rng split for shape consistency with later versions
    info["rng"], noise_rng = jax.random.split(info["rng"])

    state = jp.concatenate([noisy_hw_pos, tactile_obs, info["last_act"]])

    obs_history = jp.roll(obs_history, state.size)
    obs_history = obs_history.at[: state.size].set(state)

    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    cube_pos_error = palm_pos - cube_pos
    cube_quat = self.get_cube_orientation(data)
    cube_angvel = self.get_cube_angvel(data)
    cube_linvel = self.get_cube_linvel(data)
    fingertip_positions = self.get_fingertip_positions(data)
    joint_angles = data.qpos[self._hand_qids]

    privileged_state = jp.concatenate([
        state,
        joint_angles,
        data.qvel[self._hand_dqids],
        data.actuator_force,
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
    del metrics
    cube_pos = self.get_cube_position(data)
    palm_pos = self.get_palm_position(data)
    tip_world = self.get_fingertip_positions(data).reshape(5, 3) + palm_pos
    tip_dists = jp.linalg.norm(tip_world - cube_pos[None, :], axis=1)
    min_tip_dist = jp.min(tip_dists)
    mean_tip_dist = jp.mean(tip_dists)   # 所有指尖均值距离（用于 approach）
    hand_q = data.qpos[self._hand_qids]

    # 奖励由"任务达成 + 稳定接触 + 控制约束"三部分组成：
    # 1) 任务达成：approach(均值)/contact(多指)/closure/distal_closure/all_fingers_close
    # 2) 力反馈质量：force_contact/stable_hold
    # 3) 稳定与安全：action_rate/torques/force_overload/termination
    tip_finger_forces = info["ema_tip_finger_forces"]  # (5,) from efc_force, EMA-smoothed
    cube_linvel = self.get_cube_linvel(data)
    cube_angvel = self.get_cube_angvel(data)
    # proximity_gate：均值距离门（mean_dist < 7cm 时开门），用于 closure（MCP）。
    # 在 V-stage 中手指需先靠近方块才弯曲，以防"空捏"策略。
    # hold_position 用 contact_gate（需要实际接触才激活）。
    proximity_gate = jp.clip(1.0 - mean_tip_dist / 0.07, 0.0, 1.0)
    contact_gate = jp.clip(1.0 - min_tip_dist / 0.06, 0.0, 1.0)
    # near_gate：空间近邻门，用于 closure/distal_closure。
    # 阈值 0.10m（到方块中心）：hand 全伸时指尖约 10~12cm，此时 gate≈0。
    # MCP=90°时指尖约 7cm，gate=0.3，保有梯度（不重现 proximity_gate 死锁）。
    # 作用：强制策略先靠近方块再弯曲，防止「空捏」高分。
    near_gate = jp.clip(1.0 - min_tip_dist / 0.10, 0.0, 1.0)
    # mcp_gate：额外保留 MCP 角度门，用于 distal_closure（PIP/DIP 包裹需 MCP 先弯）。
    mcp_angles = jp.take(hand_q, jp.array([0, 3, 6, 9]))
    mcp_gate = jp.clip(jp.mean(mcp_angles) / 0.8, 0.0, 1.0)
    # released_gate：渐进释放时随 ramp 进度从 0→1，支撑完全移除后才为 1。
    support_released = info.get("support_released", jp.array(True))
    if self._support_ramp_steps > 0:
      support_timer = info.get("support_timer", jp.array(0))
      ramp_elapsed = jp.clip(
          (support_timer - self._support_release_steps).astype(jp.float32),
          0.0, float(self._support_ramp_steps),
      )
      ramp_alpha = ramp_elapsed / float(self._support_ramp_steps)
      released_gate = jp.where(support_released, ramp_alpha, 0.0)
    else:
      released_gate = support_released.astype(jp.float32)
    return {
        "approach": self._reward_approach(min_tip_dist),
        "contact": self._reward_contact(tip_world, cube_pos),  # Box-SDF 软距离奖励（Round 3）
        "thumb_engage": self._reward_thumb_engage(tip_dists),
        "closure": self._reward_closure(hand_q) * near_gate,          # [R12] near_gate：先靠近再弯曲
        "distal_closure": self._reward_distal_closure(hand_q) * near_gate * mcp_gate,  # [R12] 双门控
        "idle_follow": self._reward_idle_follow(hand_q, tip_finger_forces),
        "closure_uniformity": self._reward_closure_uniformity(hand_q),
        "finger_synergy": self._reward_finger_synergy(hand_q),
        # hold_position：support 阶段平台物理固定方块，无需奖励干预；release 后全激活。
        "hold_position": self._reward_hold_position(cube_pos, cube_linvel) * released_gate,
        "human_pose": self._reward_human_pose(hand_q) * near_gate,
        "grip_force": self._reward_grip_force(data.actuator_force) * contact_gate,  # [V-iter29] contact_gate: 手指接近时就激活，不等支撑撤出
        "stable_hold": self._reward_stable_hold(cube_pos, cube_linvel, cube_angvel, tip_finger_forces),
        "force_contact": self._reward_force_contact(tip_finger_forces),
        # 以下项权重为 0，保留接口兼容性
        "height": self._reward_height(cube_pos, palm_pos),
        "multi_finger": self._reward_multi_finger(tip_dists),
        "all_fingers_close": self._reward_all_fingers_close(hand_q),  # 无 gate：五指独立得分
        "ulnar_engage": self._reward_ulnar_engage(tip_finger_forces, hand_q),
        "force_balance": self._reward_force_balance(tip_finger_forces),
        "finger_participation": self._reward_finger_participation(tip_finger_forces),
        "single_finger_dominance": self._cost_single_finger_dominance(tip_finger_forces),
        "drop_risk": self._cost_drop_risk(cube_pos, cube_linvel),
        "lift_without_wrap": self._cost_lift_without_wrap(cube_pos, tip_finger_forces),
        "pinky_missing_on_lift": self._cost_pinky_missing_on_lift(cube_pos, tip_finger_forces),
        "no_wrap_timeout": self._cost_no_wrap_timeout(info["no_wrap_lift_steps"]),
        "survival": self._reward_survival(done),
        "termination": done,
        "action_rate": self._cost_action_rate(action, info["last_act"], info["last_last_act"]),
        "action_accel": self._cost_action_accel(action, info["last_act"], info["last_last_act"]),
        "torques": self._cost_torques(data.actuator_force),
        "force_overload": self._cost_force_overload(tip_finger_forces),
        "soft_contact": self._reward_soft_contact(tip_finger_forces),
    }

  def _get_tip_forces(self, data: mjx.Data) -> jax.Array:
    """读取 80 路指尖触觉力（5 指 * 16 taxels）。"""
    tip_force = jp.zeros((len(consts.SENSOR_HW6_FORCE_NAMES),), dtype=jp.float32)
    for idx, name in enumerate(consts.SENSOR_HW6_FORCE_NAMES):
      v = mjx_env.get_sensor_data(self.mj_model, data, name)
      tip_force = tip_force.at[idx].set(jp.ravel(v)[0])
    return tip_force

  def _aggregate_tip_forces_by_finger(self, tip_taxel_forces: jax.Array) -> jax.Array:
    """把 80 路 taxel 力加权聚合为 5 指单值，用于稳定观测与奖励。

    taxel_weights 是 16 个位置的空间权重（中心权重>边缘），
    使贴近方块的 taxel 贡献更大，提高力觉信号的方向分辨率。
    输出顺序：[thumb, index, middle, ring, pinky]，与其他函数中的 tip_force 对齐。
    """
    # 顺序固定为 thumb/index/middle/ring/pinky，每指 16 个 taxels。
    abs_force = jp.abs(tip_taxel_forces)
    force_4x4 = abs_force.reshape(5, 16)
    return jp.sum(force_4x4 * self._taxel_weights[None, :], axis=1)

  def _get_contact_forces_efc(self, data: mjx.Data) -> jax.Array:
    """Extract per-finger normal contact forces via efc_force + contact.geom.

    Touch sensors in MJX (and native MuJoCo) return all zeros because their
    site cutoff radius (0.0008 m) is far too small.  Instead, we read the
    constraint force array (efc_force) and match contact-geom pairs against
    fingertip ↔ cube to obtain real forces.

    Returns: shape (5,) in FINGERTIP_NAMES order [index, middle, ring, pinky, thumb].
    """
    contact = data.contact
    active = contact.dist < 0.0                             # (ncon,)
    safe_idx = jp.maximum(contact.efc_address, 0)
    forces = jp.abs(data.efc_force[safe_idx])               # normal force
    forces = jp.where(active & (contact.efc_address >= 0), forces, 0.0)

    cube_id = self._cube_geom_id
    per_finger = []
    for gid in self._fingertip_geom_ids_list:
      g0 = (contact.geom[:, 0] == gid) & (contact.geom[:, 1] == cube_id)
      g1 = (contact.geom[:, 1] == gid) & (contact.geom[:, 0] == cube_id)
      mask = (g0 | g1) & active
      per_finger.append(jp.sum(jp.where(mask, forces, 0.0)))
    return jp.stack(per_finger)

  def _reward_height(self, cube_pos: jax.Array, palm_pos: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】抬升奖励（原版本残留）。
    # 设计：lift = clip((z - spawn_z + 0.01) / 0.08, 0, 1)，方块上升 8cm 时满分。
    # 同时乘以水平对齐项 exp(-40 * xy_err^2)，防止把方块弹飞到侧面。
    # 当前任务不要求抬起方块，此项保留接口供将来使用。
    lift = jp.clip((cube_pos[2] - self._spawn_z + 0.01) / 0.08, 0.0, 1.0)
    xy_err = jp.linalg.norm((cube_pos - palm_pos)[:2])
    palm_alignment = jp.exp(-40.0 * jp.square(xy_err))
    return lift * palm_alignment

  def _reward_survival(self, done: jax.Array) -> jax.Array:
    # 每步存活奖励：done=0 时得 1，done=1（回合终止）时得 0。
    # 与 termination 惩罚配合：一方面鼓励活得久，另一方面不要死。
    return 1.0 - done

  def _reward_approach(self, mean_tip_dist: jax.Array) -> jax.Array:
    # 全指均值距离引导：r = exp(-20 * mean_d)。
    # 改用均值后须所有手指靠近方块才能得高分，单指独动不再满足。
    # mean=0 时 r=1，mean=0.05m 时 r≈0.37，mean=0.10m 时 r≈0.14。
    return jp.exp(-20.0 * mean_tip_dist)

  def _reward_contact(self, tip_world: jax.Array, cube_pos: jax.Array) -> jax.Array:
    # [Round 3 重写] Box-SDF 软接触奖励（替代原硬阈值 < 2.5cm）
    # 原因：min_tip_dist ≈ 6.2cm，硬阈值 2.5cm 提供零梯度，无法引导指尖靠拢
    # 新设计：计算每个指尖到方块表面的 L2 距离，软衰减 exp(-5*d)
    # 比 approach（decay=20）更平缓，在 ~20cm 外仍有梯度信号
    # tip_world: (5,3)，cube_pos: (3,)，cube_halfextent: 0.015m
    diff = jp.abs(tip_world - cube_pos[None, :]) - 0.015  # 方块外为正
    surface_dists = jp.linalg.norm(jp.maximum(diff, 0.0), axis=1)  # (5,) 各指到表面距离
    return jp.mean(jp.exp(-5.0 * surface_dists))  # 衰减系数 5（vs approach 的 20）

  def _reward_multi_finger(self, tip_dists: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】多指接近比例（被 stable_hold 替代）。
    # 5根指尖中距方块 < 3cm 的数量，除以 5 归一化到 [0,1]。
    # 曾用于鼓励多指同时靠近，防止"单指戳一下就算接触"策略。
    close_count = jp.sum((tip_dists < 0.03).astype(jp.float32))
    return close_count / 5.0

  def _reward_thumb_engage(self, tip_dists: jax.Array) -> jax.Array:
    # 拇指单独引导：r = exp(-35 * thumb_dist)，tip_dists[4] 为拇指指尖距离。
    # 拇指是对捏夹持的关键——需要到方块对侧而非同侧，
    # 单独给奖励比混在 approach 中更能引导拇指独立运动到位。
    # 比 approach 的衰减系数更大（35 vs 30），鼓励拇指更积极地靠近。
    return jp.exp(-35.0 * tip_dists[4])

  def _reward_closure(self, hand_q: jax.Array) -> jax.Array:
    # 四指+拇指整体弯曲奖励（contact_gate 门控后才激活）。
    # 60% 来自四指 MCP 均值（归一化到 [0,1.2rad]），40% 来自拇指三关节均值。
    # 关节索引：MCP=[0,3,6,9]，拇指=[13,14,15]（abd/flex_mcp/ip）。
    # 鼓励整体屈曲幅度增大，但不区分各指差异（差异性由 distal_closure 补充）。
    finger_mcp = jp.take(hand_q, jp.array([0, 3, 6, 9]))
    finger_close = jp.mean(jp.clip(finger_mcp / 1.2, 0.0, 1.0))
    thumb_close = jp.mean(jp.clip(hand_q[13:16] / 1.0, 0.0, 1.0))
    return 0.6 * finger_close + 0.4 * thumb_close

  def _reward_distal_closure(self, hand_q: jax.Array) -> jax.Array:
    # PIP/DIP 远端关节弯曲奖励（contact_gate 门控后才激活）。
    # 专门奖励中指骨（PIP）和末指骨（DIP）的弯曲，与 closure 的 MCP 层互补。
    # 意义：若只弯 MCP（掌指关节），手指整体像"铲子"把方块托起而非包裹，
    #       PIP/DIP 弯曲才能形成真正的手指包裹接触面，大幅提高抓握稳定性。
    # 80% 权重给四指远端，20% 给拇指 MCP+IP（拇指结构不同，占比较低）。
    # 关节索引：食指[0]=MCP,[1]=PIP,[2]=DIP；中指[3][4][5]；无名[6][7][8]；小指[9][10][11]。
    finger_distal = jp.array([
        jp.mean(jp.clip(hand_q[1:3] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[4:6] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[7:9] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[10:12] / 1.2, 0.0, 1.0)),
    ])
    # 拇指远端 (MCP, IP) 以同样方式计入
    thumb_distal = jp.mean(jp.clip(hand_q[14:16] / 1.0, 0.0, 1.0))
    return 0.8 * jp.mean(finger_distal) + 0.2 * thumb_distal

  def _reward_all_fingers_close(self, hand_q: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】五指总闭合量（被 closure+distal_closure 替代）。
    # 把食指/中指/无名/小指/拇指的所有关节平均闭合度求均值，输出 [0,1]。
    # 曾用于鼓励全手参与，当前已拆分为 closure（MCP） + distal_closure（PIP/DIP）。
    index_close = jp.mean(jp.clip(hand_q[0:3] / 1.2, 0.0, 1.0))
    middle_close = jp.mean(jp.clip(hand_q[3:6] / 1.2, 0.0, 1.0))
    ring_close = jp.mean(jp.clip(hand_q[6:9] / 1.2, 0.0, 1.0))
    pinky_close = jp.mean(jp.clip(hand_q[9:12] / 1.2, 0.0, 1.0))
    thumb_close = jp.mean(jp.clip(hand_q[13:16] / 1.0, 0.0, 1.0))
    return jp.mean(jp.array([index_close, middle_close, ring_close, pinky_close, thumb_close]))

  def _reward_closure_uniformity(self, hand_q: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，阶段2备用】四指闭合均匀性。
    # 设计：(1 - std(四指平均闭合度)) * mean(四指平均闭合度)。
    # 第一项：四指闭合量标准差越小（越均匀）奖励越高。
    # 第二项（门控）：乘以均值，防止"全部伸直时均匀度=1"的退化局部最优。
    # 缺点：抓握时各手指因物体形状本就角度不同，一致性奖励会对抗正常抓握，
    #        因此当前阶段不启用，仅保留接口。
    finger_close = jp.array([
        jp.mean(jp.clip(hand_q[0:3] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[3:6] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[6:9] / 1.2, 0.0, 1.0)),
        jp.mean(jp.clip(hand_q[9:12] / 1.2, 0.0, 1.0)),
    ])
    mean_close = jp.mean(finger_close)
    return jp.clip(1.0 - jp.std(finger_close), 0.0, 1.0) * mean_close

  def _reward_ulnar_engage(self, tip_force: jax.Array, hand_q: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】尺侧（无名指+小指）专项奖励。
    # 设计：60% 来自无名/小指的力接触得分，40% 来自其关节弯曲量。
    # 曾用于解决"只靠食指+中指托举方块"的局部最优，
    # 当前 stable_hold 的 active_count>=2 已隐式覆盖多指参与需求，故不再单独激活。
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.finger_active_threshold
    sat = self._config.reward_config.force_contact_saturation
    ring_contact = jp.clip((abs_f[3] - th) / (sat - th + 1e-6), 0.0, 1.0)
    pinky_contact = jp.clip((abs_f[4] - th) / (sat - th + 1e-6), 0.0, 1.0)
    ulnar_contact = 0.5 * (ring_contact + pinky_contact)

    ring_close = jp.mean(jp.clip(hand_q[6:9] / 1.2, 0.0, 1.0))
    pinky_close = jp.mean(jp.clip(hand_q[9:12] / 1.2, 0.0, 1.0))
    ulnar_close = 0.5 * (ring_close + pinky_close)
    return 0.6 * ulnar_contact + 0.4 * ulnar_close

  def _reward_force_contact(self, tip_force: jax.Array) -> jax.Array:
    # 真实力传感器正反馈：r = mean(clip((|f| - th) / (sat - th), 0, 1))，5指平均。
    # th=0.08（接触阈值），sat=1.2N（饱和点）：超过阈值后线性增长，达1.2N后满分。
    # 意义：补充 stable_hold 的速度稳定性约束，给真实物理接触力显式正反馈，
    #        防止策略只靠关节角度到位但实际不施力（零力悬停）的伪接触策略。
    abs_f = jp.abs(tip_force)
    th = self._config.reward_config.force_contact_threshold
    sat = self._config.reward_config.force_contact_saturation
    normalized = jp.clip((abs_f - th) / (sat - th + 1e-6), 0.0, 1.0)
    return jp.mean(normalized)

  def _reward_force_balance(self, tip_force: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】五指力均衡奖励。
    # 设计：r = clip(1 - std(|f|)/mean(|f|), 0, 1)，相对标准差越小越好。
    # 含义：五根手指受力越均匀奖励越高，防止单指承受全部载荷。
    # 当前阶段优先学会抓住，不要求力均匀，故暂不激活。
    abs_f = jp.abs(tip_force)
    mean_f = jp.mean(abs_f) + 1e-6
    rel_std = jp.std(abs_f) / mean_f
    return jp.clip(1.0 - rel_std, 0.0, 1.0)

  def _reward_finger_participation(self, tip_force: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】多指参与度奖励（被 stable_hold 替代）。
    # 设计：60% * (5指激活比例) + 40% * (非拇指4指激活比例)。
    # "激活"定义：指尖力 > finger_active_threshold(0.15)。
    # 额外加权非拇指比例，防止策略只依靠拇指+1根手指捏持。
    th = self._config.reward_config.finger_active_threshold
    active = (jp.abs(tip_force) > th).astype(jp.float32)
    # 五指激活比例 + 非拇指激活比例，避免只依赖拇指/单指。
    all_ratio = jp.mean(active)
    non_thumb_ratio = jp.mean(active[1:])
    return 0.6 * all_ratio + 0.4 * non_thumb_ratio

  def _cost_single_finger_dominance(self, tip_force: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】单指力独占惩罚。
    # 设计：max_share = max(|f|) / sum(|f|)，超过阈值(0.55)后线性惩罚。
    # 含义：若某根手指承受了55%以上的力，视为"单指独占"，轻微惩罚。
    # 当前降低学习难度暂不启用，待抓握稳定后再激活。
    abs_f = jp.abs(tip_force)
    total = jp.sum(abs_f) + 1e-6
    max_share = jp.max(abs_f) / total
    th = self._config.reward_config.single_finger_share_threshold
    return jp.clip((max_share - th) / (1.0 - th + 1e-6), 0.0, 1.0)

  def _cost_force_overload(self, tip_force: jax.Array) -> jax.Array:
    """非线性过载惩罚：quadratic + cubic 平滑过渡。

    设计：在 F_max 附近用 quadratic，远超 F_max 时加入 cubic 项使梯度更陡峭。
      cost = mean( overload^2 + (overload / W)^3 )
    其中 overload = max(|f| - F_max, 0)，W = force_overload_soft_width。

    效果：
      - overload < W (0.5N)：以 quadratic 为主，惩罚温和
      - overload > W：cubic 项主导，梯度陡增，严厉惩罚极端过载
      - 相比纯 quadratic，对>3.3N 的力有更强抑制
    """
    maxf = self._config.reward_config.force_overload_threshold        # 2.8N
    soft_w = self._config.reward_config.force_overload_soft_width     # 0.5N
    abs_f = jp.abs(tip_force)
    overload = jp.maximum(abs_f - maxf, 0.0)
    # quadratic 基础 + cubic 加速：轻微超标温和，严重超标陡峭
    cost = jp.square(overload) + jp.power(overload / (soft_w + 1e-6), 3.0)
    return jp.mean(cost)

  def _reward_soft_contact(self, tip_force: jax.Array) -> jax.Array:
    """安全区间内的 bell-shaped 有效接触奖励。

    设计：在 [F_min, F_max] 区间内给予持续正反馈，区间外迅速衰减。
      lower_gate = sigmoid(20 * (|f| - F_min))  -- |f| > F_min 时开启
      upper_gate = sigmoid(10 * (F_max - |f|))  -- |f| < F_max 时开启
      reward = mean(lower_gate * upper_gate)

    效果：
      - |f| < 0.1N：几乎为 0（无效接触）
      - 0.1N < |f| < 2.5N：≈1（安全有效区间，持续正反馈）
      - |f| > 2.5N：快速衰减（过载预警，与 force_overload 惩罚互补）
    权重推荐：scale=3.0~6.0，与 force_contact 互补但不冲突。
    """
    fmin = self._config.reward_config.soft_contact_fmin    # 0.1N
    fmax = self._config.reward_config.soft_contact_fmax    # 2.5N
    abs_f = jp.abs(tip_force)
    # 用 sigmoid 实现平滑门控，避免硬阈值导致的梯度不连续
    lower_gate = jax.nn.sigmoid(20.0 * (abs_f - fmin))   # 左侧上升沿
    upper_gate = jax.nn.sigmoid(10.0 * (fmax - abs_f))   # 右侧下降沿
    return jp.mean(lower_gate * upper_gate)


  def _reward_stable_hold(
      self,
      cube_pos: jax.Array,
      cube_linvel: jax.Array,
      cube_angvel: jax.Array,
      tip_force: jax.Array,
  ) -> jax.Array:
    # 稳定持握奖励（scale=7.0，最高权重，核心目标）。
    # 三因子乘积：contact_gate * lin_stable * ang_stable
    #
    # contact_gate：soft-gate，要求至少2根手指有力信号。
    #   active_count=1 时 gate=0，active_count=2 时 gate=0.5，active_count=3 时 gate=1。
    #   意义：单指接触不算稳握，至少2指才开始给分，迫使策略建立多指接触。
    #
    # lin_stable = exp(-6 * |linvel|)：方块线速度越小越好。
    #   |linvel|=0 时满分，0.1m/s时≈0.55，0.3m/s时≈0.16。
    #
    # ang_stable = exp(-4 * |angvel|)：方块角速度越小越好。
    #   |angvel|=0 时满分，0.3rad/s时≈0.30。
    #
    # 三者相乘确保策略必须同时做到"多指有力接触 + 方块几乎不动"才能得高分，
    # 而不能靠其中某一项单独满足就获得高奖励。
    active_th = self._config.reward_config.finger_active_threshold
    active_count = jp.sum((jp.abs(tip_force) > active_th).astype(jp.float32))
    contact_gate = jp.clip((active_count - 1.0) / 2.0, 0.0, 1.0)
    lin_stable = jp.exp(-6.0 * jp.linalg.norm(cube_linvel))
    ang_stable = jp.exp(-4.0 * jp.linalg.norm(cube_angvel))
    return contact_gate * lin_stable * ang_stable


  def _reward_finger_synergy(self, hand_q: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，阶段2备用】四指对应关节角度一致性。
    # 设计：对 MCP/PIP/DIP 分别计算 exp(-5*(max-min))，三者平均后乘弯曲门控。
    # max-min 范围越小 → 四指弯曲越整齐 → 奖励越高（最大约 1.57rad）。
    # 弯曲门控 mcp_mean_flex：全部伸直时门控=0，防止静止退化；
    #   弯曲越深门控越强，训练方向上"先弯起来，再整齐"。
    # 已知缺点：真实抓握时各手指角度本就不同（因物体形状自然适应），
    #   强制一致性会对抗正常抓握，故当前阶段不启用，由 idle_follow 替代。
    # MCP: indices 0, 3, 6, 9
    mcp = jp.array([hand_q[0], hand_q[3], hand_q[6], hand_q[9]])
    # PIP: indices 1, 4, 7, 10
    pip = jp.array([hand_q[1], hand_q[4], hand_q[7], hand_q[10]])
    # DIP: indices 2, 5, 8, 11
    dip = jp.array([hand_q[2], hand_q[5], hand_q[8], hand_q[11]])
    # 用 max-min 范围衡量一致性（范围越小越好，最大可能 ~1.57 rad）
    mcp_sync = jp.exp(-5.0 * (jp.max(mcp) - jp.min(mcp)))
    pip_sync = jp.exp(-5.0 * (jp.max(pip) - jp.min(pip)))
    dip_sync = jp.exp(-5.0 * (jp.max(dip) - jp.min(dip)))
    # 弯曲度门控：MCP 均值越大，姿态奖励权重越高
    mcp_mean_flex = jp.clip(jp.mean(mcp) / 1.2, 0.0, 1.0)
    return (mcp_sync + pip_sync + dip_sync) / 3.0 * mcp_mean_flex

  def _reward_idle_follow(self, hand_q: jax.Array, tip_force: jax.Array) -> jax.Array:
    """阶段2姿态奖励：空闲手指跟随受力手指的弯曲角度。

    受力手指（有真实接触力）：不施加额外约束，自由贴物体。
    空闲手指（无接触力）：MCP 应不低于受力手指均值角度，否则扣分。
    只惩罚"比接触手指更直"，不惩罚"更弯"→ 自然并拢但不干预接触自由度。
    无接触时整体=0，避免静止退化。
    """
    active_th = self._config.reward_config.finger_active_threshold
    # 四指 MCP 角度：index=0, middle=3, ring=6, pinky=9
    mcp = jp.array([hand_q[0], hand_q[3], hand_q[6], hand_q[9]])
    # 对应四指力值（不含拇指）
    f4 = jp.abs(tip_force[1:])  # [index, middle, ring, pinky]
    is_active = (f4 > active_th).astype(jp.float32)
    n_active = jp.sum(is_active) + 1e-6
    # 参考角度：受力手指的 MCP 加权均值
    ref_mcp = jp.dot(mcp, is_active) / n_active
    # 空闲手指"比参考角更直"的差距（越大越差）
    is_idle = 1.0 - is_active
    flex_gap = jp.maximum(ref_mcp - mcp, 0.0)
    follow_score = jp.exp(-5.0 * flex_gap)
    n_idle = jp.sum(is_idle) + 1e-6
    idle_score = jp.dot(is_idle, follow_score) / n_idle
    # 至少1根手指受力才激活
    has_contact = jp.clip(n_active - 1.0, 0.0, 1.0)
    return idle_score * has_contact

  def _reward_human_pose(self, hand_q: jax.Array) -> jax.Array:
    """人手自然抓握姿态模仿奖励。

    目标关节角（弧度）参考人手自然握持 3cm 方块：
      四指 MCP ≈ 70° (1.22 rad), PIP ≈ 55° (0.96 rad), DIP ≈ 40° (0.70 rad)
      拇指 CMC_ABD ≈ 1.25 rad (对握位置，thumb_rot≈1.3 时拇指可达方块)
      拇指 CMC_FLEX ≈ 0.70 rad, MCP ≈ 0.60 rad, IP ≈ 0.50 rad
    用 exp(-k * (q - q_target)^2) 软罚，容许±15°偏差仍有高分。

    [V-iter11] 修改：加入 hand_q[12]（thumb_cmc_abd）目标 1.25 rad，
    引导拇指旋转到对握位置。之前忽略了此关节导致拇指无法对准方块。
    """
    # 四指目标：[MCP, PIP, DIP] × 4
    finger_targets = jp.array([
        1.22, 0.96, 0.70,  # index
        1.22, 0.96, 0.70,  # middle
        1.22, 0.96, 0.70,  # ring
        1.22, 0.96, 0.70,  # pinky
    ])
    # 拇指目标：[abd(12), cmc_flex(13), mcp(14), ip(15)]
    thumb_targets = jp.array([1.20, 0.70, 0.60, 0.50])  # [V-iter12] abd: 1.25→1.20（物理验证最优值）
    # 四指得分
    finger_q = hand_q[:12]
    finger_err = jp.square(finger_q - finger_targets)
    finger_score = jp.mean(jp.exp(-8.0 * finger_err))
    # 拇指得分（包含 abd）
    thumb_q = hand_q[12:16]
    thumb_err = jp.square(thumb_q - thumb_targets)
    thumb_score = jp.mean(jp.exp(-8.0 * thumb_err))
    return 0.6 * finger_score + 0.4 * thumb_score

  def _reward_hold_position(self, cube_pos: jax.Array, cube_linvel: jax.Array) -> jax.Array:
    # [V-iter25] 方块 3D 位置保持奖励（旧版仅测量 Z 高度→方块可横向飞走无惩罚）。
    # 设计：r = exp(-40 * ‖pos - spawn_pos‖_2) * (1 - 0.7 * vel_penalty)。
    # 3D 距离衰减系数 40：偏1mm=0.96 偏5mm=0.82 偏1cm=0.67 偏2cm=0.45
    # vel_penalty：0.15 m/s 满惩，系数 0.7。
    dist_from_spawn = jp.linalg.norm(cube_pos - self._spawn_pos)
    pos_reward = jp.exp(-40.0 * dist_from_spawn)
    vel_penalty = jp.clip(jp.linalg.norm(cube_linvel) / 0.15, 0.0, 1.0)
    return pos_reward * (1.0 - 0.7 * vel_penalty)

  def _cost_drop_risk(self, cube_pos: jax.Array, cube_linvel: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】掉落风险早期惩罚（由 termination 覆盖）。
    # 设计：low_risk * (0.4 + 0.6 * down_risk)，两者取乘积。
    # low_risk：方块低于 spawn_z 的程度（越低越大）。
    # down_risk：方块向下速度（-linvel_z）的程度。
    # 意义：在方块真正落地（触发 termination）前提早给惩罚信号，加快学习收敛。
    # 当前 termination 惩罚已足够，不需要额外早期惩罚，故暂不激活。
    drop_ref = self._spawn_z
    low_risk = jp.clip((drop_ref - cube_pos[2]) / jp.maximum(drop_ref, 0.005), 0.0, 1.0)
    down_risk = jp.clip((-cube_linvel[2]) / 0.25, 0.0, 1.0)
    return low_risk * (0.4 + 0.6 * down_risk)

  def _cost_lift_without_wrap(self, cube_pos: jax.Array, tip_force: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】抬起但未包裹惩罚（当前不要求抬起）。
    # 设计：lifted * insufficient_wrap，两者取乘积。
    # lifted：方块高于 spawn_z 3cm 以上时 =1。
    # insufficient_wrap：激活手指 < 3.5 根时惩罚（鼓励至少4根手指参与包裹）。
    # 意义：防止策略只用1~2根手指托举方块，而不形成稳定包裹抓握。
    # 当前不要求抬起方块，此项无意义，不激活。
    lifted = jp.clip((cube_pos[2] - self._spawn_z) / 0.03, 0.0, 1.0)
    active = jp.sum((jp.abs(tip_force) > self._config.reward_config.finger_active_threshold).astype(jp.float32))
    insufficient_wrap = jp.clip((3.5 - active) / 3.5, 0.0, 1.0)
    return lifted * insufficient_wrap

  def _cost_pinky_missing_on_lift(self, cube_pos: jax.Array, tip_force: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】抬起后小指缺失惩罚。
    # 设计：lifted * (1 - pinky_active)，小指无力时乘以方块高度。
    # 意义：HW6 的小指结构较弱，抬起时若只有3根手指参与，稳定性很差；
    #        显式惩罚推动小指也参与闭合形成"四指包裹"。
    # 当前不要求抬起，不激活。
    lifted = jp.clip((cube_pos[2] - self._spawn_z) / 0.03, 0.0, 1.0)
    th = self._config.reward_config.finger_active_threshold
    pinky_active = (jp.abs(tip_force[4]) > th).astype(jp.float32)
    return lifted * (1.0 - pinky_active)

  def _is_wrap_established(self, tip_force: jax.Array) -> jax.Array:
    active_th = self._config.reward_config.finger_active_threshold
    active_count = jp.sum((jp.abs(tip_force) > active_th).astype(jp.int32))
    min_active = jp.array(self._config.reward_config.wrap_active_fingers_threshold, dtype=jp.int32)
    force_ok = jp.sum(jp.abs(tip_force)) >= self._config.reward_config.wrap_force_threshold
    return jp.logical_and(active_count >= min_active, force_ok)

  def _is_lift_without_wrap(self, cube_pos: jax.Array, tip_force: jax.Array) -> jax.Array:
    lifted = cube_pos[2] >= self._config.reward_config.lifted_height_threshold
    wrapped = self._is_wrap_established(tip_force)
    return jp.logical_and(lifted, jp.logical_not(wrapped))

  def _cost_no_wrap_timeout(self, no_wrap_lift_steps: jax.Array) -> jax.Array:
    # 【当前 scale=0.0，不参与训练】长时间"托举未包裹"超时惩罚。
    # 设计：若连续处于"托举但未包裹"状态超过 no_wrap_timeout_sec(0.8s)，
    #        惩罚随超时步数线性增大，最终 clip 到 1.0。
    # 意义：防止策略停留在"1根手指托着方块晃来晃去"的低效状态。
    # 当前不要求抬起，不激活。
    timeout_steps = max(1, int(np.round(self._config.reward_config.no_wrap_timeout_sec / self.dt)))
    overtime = jp.maximum(no_wrap_lift_steps - timeout_steps, 0)
    return jp.clip(overtime / float(timeout_steps), 0.0, 1.0)

  def _reward_grip_force(self, actuator_force: jax.Array) -> jax.Array:
    """对握力奖励：四指弯曲力与拇指弯曲力同时存在时给正奖励。

    actuator_force 共 7 维：[index, middle, ring, pinky, thumb_abd, thumb1, thumb2]。
    四指腱力(0:4) 为正表示闭合，拇指腱力(5:7) 为正表示闭合。
    当两侧都有闭合力时，说明在形成对握(opposition)夹持。
    用 min(finger_force, thumb_force) 保证双侧同时发力才有奖励。
    """
    # 四指闭合力均值（只取正值=闭合方向）
    finger_f = jp.mean(jp.clip(actuator_force[:4], 0.0, 5.0))
    # 拇指闭合力均值（thumb1, thumb2 腱）
    thumb_f = jp.mean(jp.clip(actuator_force[5:7], 0.0, 5.0))
    # 归一化到 [0,1] 范围
    finger_norm = jp.clip(finger_f / 1.0, 0.0, 1.0)
    thumb_norm = jp.clip(thumb_f / 1.0, 0.0, 1.0)
    # 对握力 = min(双侧力)，确保双方都发力
    return jp.minimum(finger_norm, thumb_norm)

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    # 关节力矩正则惩罚（scale=-0.001，当前激活）。
    # 设计：cost = sum(τ^2)，所有驱动力矩的均方和。
    # 意义：鼓励使用较小的关节力矩完成抓握，减少能耗和机械冲击，
    #        同时防止策略用"暴力夹持"策略（配合 force_overload 双重约束）。
    # 权重极小(-0.001)以避免过度抑制必要的抓握力。
    return jp.sum(jp.square(torques))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    # 动作一阶变化率惩罚（scale=-0.04，当前激活）。
    # 设计：cost = sum((a_t - a_{t-1})^2)，帧间动作差的均方和。
    # 意义：抑制逐帧动作突变，让控制信号更平滑，
    #        降低机械手振动和实物部署时的抖动问题。
    return jp.sum(jp.square(act - last_act))

  def _cost_action_accel(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    # 动作二阶加速度（jerk）惩罚（scale=-0.03，当前激活）。
    # 设计：jerk = a_t - 2*a_{t-1} + a_{t-2}，即动作的二阶差分（离散二阶导）。
    # cost = sum(jerk^2)，防止"来回抽动"的震荡控制策略。
    jerk = act - 2.0 * last_act + last_last_act
    return jp.sum(jp.square(jerk))


def domain_randomize_cube_size(model: mjx.Model, rng: jax.Array):
  """Lightweight domain randomization: cube size only.

  Randomizes cube half-size uniformly in [0.013, 0.017] (2.6-3.4cm edge).
  Designed for V-iter24+: one change at a time from stable V-iter19.
  """
  mj_model = CubeGraspHW6Force().mj_model
  cube_geom_id = mj_model.geom("cube").id

  @jax.vmap
  def rand(rng):
    rng, key = jax.random.split(rng)
    # half-size: default 0.015, randomize ±0.002 (±13%)
    cube_half = jax.random.uniform(key, (3,), minval=0.013, maxval=0.017)
    geom_size = model.geom_size.at[cube_geom_id].set(cube_half)
    return (geom_size,)

  (geom_size,) = rand(rng)

  in_axes = jax.tree_util.tree_map(lambda x: None, model)
  in_axes = in_axes.tree_replace({"geom_size": 0})
  model = model.tree_replace({"geom_size": geom_size})
  return model, in_axes


def domain_randomize(model: mjx.Model, rng: jax.Array):
  """Domain randomization for HW6 force-aware grasp task."""
  mj_model = CubeGraspHW6Force().mj_model
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
    rng, key = jax.random.split(rng)
    cube_friction = jax.random.uniform(key, (1,), minval=0.1, maxval=0.5)
    geom_friction = model.geom_friction.at[cube_geom_id : cube_geom_id + 1, 0].set(cube_friction)

    fingertip_friction = jax.random.uniform(key, (1,), minval=0.5, maxval=1.0)
    geom_friction = model.geom_friction.at[fingertip_geom_ids, 0].set(fingertip_friction)

    rng, key1, key2 = jax.random.split(rng, 3)
    dmass = jax.random.uniform(key1, minval=0.8, maxval=1.2)
    cube_mass = model.body_mass[cube_body_id]
    body_mass = model.body_mass.at[cube_body_id].set(cube_mass * dmass)
    body_inertia = model.body_inertia.at[cube_body_id].set(model.body_inertia[cube_body_id] * dmass)
    dpos = jax.random.uniform(key2, (3,), minval=-5e-3, maxval=5e-3)
    body_ipos = model.body_ipos.at[cube_body_id].set(model.body_ipos[cube_body_id] + dpos)

    rng, key = jax.random.split(rng)
    qpos0 = model.qpos0
    qpos0 = qpos0.at[hand_qids].set(
      qpos0[hand_qids] + jax.random.uniform(key, shape=(16,), minval=-0.03, maxval=0.03)
    )

    rng, key = jax.random.split(rng)
    frictionloss = model.dof_frictionloss[hand_qids] * jax.random.uniform(
      key, shape=(16,), minval=0.8, maxval=1.2
    )
    dof_frictionloss = model.dof_frictionloss.at[hand_qids].set(frictionloss)

    rng, key = jax.random.split(rng)
    armature = model.dof_armature[hand_qids] * jax.random.uniform(
        key, shape=(16,), minval=1.0, maxval=1.05
    )
    dof_armature = model.dof_armature.at[hand_qids].set(armature)

    rng, key = jax.random.split(rng)
    dmass = jax.random.uniform(key, shape=(len(hand_body_ids),), minval=0.9, maxval=1.1)
    body_mass = model.body_mass.at[hand_body_ids].set(model.body_mass[hand_body_ids] * dmass)

    rng, key = jax.random.split(rng)
    kp = model.actuator_gainprm[:, 0] * jax.random.uniform(
      key, (model.nu,), minval=0.9, maxval=1.1
    )
    actuator_gainprm = model.actuator_gainprm.at[:, 0].set(kp)
    actuator_biasprm = model.actuator_biasprm.at[:, 1].set(-kp)

    rng, key = jax.random.split(rng)
    kd = model.dof_damping[hand_qids] * jax.random.uniform(
      key, (16,), minval=0.9, maxval=1.1
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


class CubeGraspHW6ForceP(CubeGraspHW6Force):
  """P 阶段：平躺手场景，训练“先握住”。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config_p(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=consts.GRASP_HW6_PALMUP_XML.as_posix(),
    )


class CubeGraspHW6ForceV(CubeGraspHW6Force):
  """V 阶段：竖直抓取场景，用于承接 P checkpoint 继续训练。"""

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config_v(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=consts.GRASP_HW6_XML.as_posix(),
    )


def default_config_vt() -> config_dict.ConfigDict:
  """VT 阶段配置：竖直场景 + 实时触觉观测。

  在 V-stage 基础上启用真实触觉反馈（efc_force → EMA → 归一化 → 观测）。
  必须从零训练（旧 checkpoint normalizer 不兼容）。

  触觉引入后的奖励调整原则：
    - 策略现在能直接"感受"力，不再需要 closure/distal_closure 等
      纯几何代理信号的高权重，可适度降低以减少信号冗余。
    - force_contact 与 stable_hold 才是核心：触觉反映真实接触质量。
    - 新增 tactile_richness 奖励：鼓励多指均有力反馈（替代几何多指奖励）。
  """
  cfg = default_config_v()
  # ── 启用实时触觉 ─────────────────────────────────────────────────────────
  cfg.tactile_config.use_real_tactile = True
  cfg.tactile_config.force_saturation_n = 3.0   # 与硬件传感器 3N 量程对齐
  cfg.tactile_config.obs_force_ema_alpha = 0.7   # ~3步时间常数，平滑碰撞瞬时尖峰

  # ── VT 特有：episode 延长到 600 步 (30s @ 20Hz) ───────────────────────────
  cfg.episode_length = 600

  # ── VT 特有：方块 Y 轴 jitter 增大（不超出掌心） ─────────────────────────
  # V-stage: ±5mm；VT: ±8mm（掌心宽度~25mm，cube半宽15mm，余量≈10mm安全）
  cfg.spawn_config.cube_jitter = [0.004, 0.008, 0.003]

  # ── VT 奖励权重 ──────────────────────────────────────────────────────────
  # 核心目标奖励（触觉直接驱动）
  cfg.reward_config.scales.hold_position = 150.0  # iter9: 回退至iter5值，方案A无效
  cfg.reward_config.scales.force_contact = 8.0    # ↑ V=5→8：策略能感知力，给更强正反馈
  cfg.reward_config.scales.stable_hold = 15.0     # iter9: 回退至iter5值

  # 几何引导（触觉引入后降低，避免与力信号冲突）
  cfg.reward_config.scales.contact = 3.0          # ↓ V=5→3：SDF代理信号，让位给真实力
  cfg.reward_config.scales.closure = 3.0          # ↓ V=5→3：MCP弯曲引导弱化
  cfg.reward_config.scales.distal_closure = 3.0   # ↓ V=5→3：PIP/DIP弯曲引导弱化
  cfg.reward_config.scales.thumb_engage = 6.0     # ↓ V=8→6：拇指靠近引导弱化

  # 力反馈质量奖励
  cfg.reward_config.scales.grip_force = 4.0       # ↑ V=3→4：对握力，配合触觉提供更丰富梯度
  cfg.reward_config.scales.force_balance = 1.5    # iter9: 回退至iter5值
  cfg.reward_config.scales.finger_participation = 2.0  # ↑ V=0→2：启用多指参与度

  # 控制约束
  cfg.reward_config.scales.force_overload = -1.5  # ↑ V=0→-1.5：启用过载惩罚  # ↑ V=0→-1.5：启用过载惩罚
  cfg.reward_config.scales.action_rate = -0.02    # V=-0.01→-0.02：触觉闭环允许更平滑控制
  cfg.reward_config.scales.action_accel = -0.015  # V=-0.008→-0.015：同上

  # 其他保持与V一致
  cfg.reward_config.scales.human_pose = 3.0       # ↓ V=5→3：触觉可替代部分姿态引导
  cfg.reward_config.scales.survival = 2.0         # iter9: 回退至iter5值
  cfg.reward_config.scales.termination = -120.0
  cfg.reward_config.scales.drop_risk = -10.0     # iter3: -25→-10, 减少z漂移噪声惩罚
  cfg.reward_config.scales.torques = -0.00003
  # [iter12] 降低观测噪声：0.3→0.15，更清晰信号帮助精细控力
  cfg.noise_config.level = 0.15

  return cfg


class CubeGraspHW6ForceVT(CubeGraspHW6Force):
  """VT 阶段：竖直场景 + 实时触觉观测（Vertical-Tactile）。

  与 V 阶段的关键区别：
    1. 策略观测的 5D 触觉位包含真实 efc_force（0~1 归一化）
    2. 奖励函数启用 stable_hold、force_balance 等力反馈相关项
    3. 必须从零训练，不兼容旧 checkpoint 的 normalizer
  """

  def __init__(
      self,
      config: config_dict.ConfigDict = default_config_vt(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        config=config,
        config_overrides=config_overrides,
        xml_path=consts.GRASP_HW6_XML.as_posix(),
    )
