# V2 Autonomous Grasp Iteration Prompt

You are Codex working in `/home/ll/SRTP/Aero-Hand`. Continue the V2 Aero-Hand cube grasp project fully autonomously until either:

1. the policy can continuously hold the cube for 30 seconds, or
2. more than 30 autonomous iterations have been attempted **starting from C09 on 2026-04-20**.

## 2026-04-23 Superseding User Requirements

The current workspace is `/root/autodl-tmp/Aero-Hand`. Treat the following as the new top-priority autonomous workflow:

1. Inspect the latest hand-parameter changes first.
2. Starting from the best existing **cube** checkpoint, adapt the cube policy to the **current hand parameters** and continue autonomous iteration until the cube task is again stable for **30s+**, with **DR enabled**.
3. After the cube task is revalidated, continue autonomous iteration on the **can** task (the previous bottle task has been replaced by can grasping).
4. The can task must satisfy all of the following:
   - stable grasp for **30s+** without dropping,
   - the can must **not be deformed / over-compressed**,
   - when the hand is shaken while grasping the can, the can should still not drop,
   - use the **current script-provided initial pose**,
   - include DR and perturbation robustness, not just fixed-physics success.
5. Do not merely rerun the same training command. After each iteration, read logs/metrics/video, diagnose the failure mode, and modify reward terms, initial-state curriculum, thresholds, or task setup accordingly.
6. Keep cube and can iteration records separate:
   - cube: append to `v2_iteration_docs/changelog.md`
   - can: append to `v2_iteration_docs/can_grasp_changelog.md`
7. Keep task files isolated. Do not break the cube task while iterating on the can task.
8. For every can iteration, before moving on to the next experiment, you must append a can-specific log entry to `v2_iteration_docs/can_grasp_changelog.md` that includes:
   - exact code changes,
   - exact training command / checkpoint source,
   - smoke test status,
   - `contact_duration_sec` first / last / max / best step,
   - metric analysis and comparison vs. previous can runs,
   - why the change was made,
   - expected effect,
   - actual effect,
   - next-step recommendation.
   Do not mix can logs into cube logs.

Use `AeroCubeGraspV2ForceCoacd` with `num_envs=4096` unless a smoke/debug run is explicitly needed before a full run. The current date is 2026-04-20.

## Source Of Truth

Read and obey:

- `/home/ll/SRTP/Aero-Hand/v2_iteration_docs/prompts/V2_current_memory_prompt.md`
- `/home/ll/SRTP/Aero-Hand/v2_iteration_docs/changelog.md`

All iteration records must be appended to:

- `/home/ll/SRTP/Aero-Hand/v2_iteration_docs/changelog.md`

Training stdout logs must be saved to:

- `/home/ll/SRTP/Aero-Hand/v2_iteration_docs/training_logs/`

Training videos must be copied to:

- `/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/TempVideos/`

Do not rely only on reward score. Use metrics and videos to judge whether the policy truly grasps with thumb/index/middle.

## Current Status Snapshot (2026-04-20)

### 30s持握目标已达成 (C21)
- C21 在固定物理下达成 **37.89s** 稳定三指持握
- contact_duration=37.89s, hold_success=151.67, drop_rate=3.9%, palm=0.1, slip=0.00
- 当前进入**鲁棒性增强阶段**: DR + 扰动

### C22 正在训练 — DR + 扰动
- 从C21 checkpoint恢复, 10M步
- 启用Domain Randomization (摩擦/质量/关节)
- 启用扰动: 外力脉冲(0.08N) + 重力倾斜(±0.3rad) + 关节噪声(0.01rad)

### Already Fixed Or Confirmed

- The thumb visual/kinematic mismatch was traced to hidden self-collision, not actuator parameter passing.
- `hw_thumb_rot=1.38` now reaches the expected large thumb abduction when self-collision excludes are applied.
- Added self-collision excludes for:
  - `palm` vs `right_thumb_mid`,
  - `palm` vs `right_ring_distal`,
  - `palm` vs `right_pinky_distal`.
- The default reset/home pose matches the user-requested start:
  `[thumb_rot, thumb_flex, index, middle, ring, pinky] = [1.38, 0.0, 0.55, 0.55, 1.4, 1.4]`.
- Cube/support placement (C08: 方块向食指偏移+靠近手心, 居三指力交点):
  - `cube_pos = [0.025, -0.065, 0.1503]`
  - `support_pos = [0.025, -0.065, 0.1348]`
- Support collision filtering is cube-only:
  `cube_support_geom contype=4 conaffinity=0`.
- Five fingertips each have 4x4 tactile/touch sensors, total 80.
- Actuator targets are now clipped to actuator `ctrlrange` before stepping, so larger residual action ranges are safe.

### Achieved Training Effects So Far

- R100-R106: 从初始bootstrap失败逐步建立三指接触+持握能力
- C08: 初始状态校准 (用户目视碰撞叠加确认)
- C20: 穿模修复 + 奖励函数迭代 + 观测增强 → contact_duration=8.96s, drop=35.9%
- **C21: palm/nonprimary清理 + ring/pinky锁定 → contact_duration=37.89s, drop=3.9% ✓ 30s目标达成**
- C22: DR + 扰动 (训练中)

### Current Main Direction

C21已在固定物理下达成37.89s持握。当前目标是通过DR和扰动增强策略鲁棒性:
- 在DR (摩擦/质量随机化) 下保持稳定持握
- 在外力脉冲和重力倾斜扰动下保持力封闭
- 为后续sim-to-real转移做准备

## Preflight Checks

Before training, confirm these conditions:

- Current training scene is `scene_mjx_grasp_v2_coacd.xml`.
- Current collision model is primitive capsule/box, not high-count CoACD mesh.
- Support platform collides only with the cube, not fingers or palm.
- Initial reset hand target is approximately:
  `[thumb_rot, thumb_flex, index, middle, ring, pinky] = [1.38, 0.0, 0.55, 0.55, 1.4, 1.4]`.
- Cube starts at the three-finger force intersection (三力合一):
  `cube_pos = [0.025, -0.065, 0.1503]`.
- Support starts under the cube:
  `support_pos = [0.025, -0.065, 0.1348]`.
- Each fingertip has a 4x4 tactile grid:
  thumb/index/middle/ring/pinky each has 16 tactile sites and 16 touch sensors, total 80.

Current tactile preflight result:

| Finger | Sites | Touch sensors | Local site bbox | Tip site | Nearby collision |
|---|---:|---:|---|---|---|
| thumb | 16 | 16 | `[-0.010, 0.008, -0.020]` to `[-0.001, 0.020, -0.020]` | `[-0.006, 0.021, -0.020]` | `right_thumb_tip_capsule_*` |
| index | 16 | 16 | `[-0.007, 0.042, 0.005]` to `[-0.001, 0.051, 0.005]` | `[-0.004, 0.052, 0.005]` | `right_index_distal_capsule_*` |
| middle | 16 | 16 | `[-0.007, 0.042, 0.005]` to `[-0.001, 0.051, 0.005]` | `[-0.004, 0.052, 0.005]` | `right_middle_distal_capsule_*` |
| ring | 16 | 16 | `[-0.007, 0.042, 0.005]` to `[-0.001, 0.051, 0.005]` | `[-0.004, 0.052, 0.005]` | `right_ring_distal_capsule_*` |
| pinky | 16 | 16 | `[-0.007, 0.042, 0.005]` to `[-0.001, 0.051, 0.005]` | `[-0.004, 0.052, 0.005]` | `right_pinky_distal_capsule_*` |

## Autonomous Iteration Loop

Run up to 30 continuation iterations from C09 (C08 was initial state calibration). Each iteration must follow this loop:

### Execution Principles

- **不使用脚本反复发送消息**: 训练启动后，使用 sleep 或定期轮询 stdout 日志文件来等待训练完成，不要用外部脚本循环调用。
- **训练步数不固定**: 每轮训练的 `num_timesteps` 不是固定 5M，需要根据日志判断：
  - 如果 reward 在 2M 步时已明显收敛/平台，可以提前终止并分析；
  - 如果 reward 仍在上升，可以延长到 10M 甚至 20M；
  - 首轮建议 5M 作为 baseline，之后根据日志灵活调整。
- **日志驱动决策**: 每轮训练结束后必须读取完整 stdout 日志和 metrics，基于数据做决策，而非凭直觉。
- **最小化改动原则**: 每轮只做一个关键改动（reward 权重 / 阈值 / 新 component / 初始化策略），便于归因。
- **全自主执行**: 从启动训练、等待完成、读取日志、分析失败模式、设计下一轮改动、实施改动到再次训练，全程自主完成，不需要用户介入（除非遇到不可解决的问题）。

### Iteration Steps

1. **Candidate design**
   - Generate several reward/initialization/observation candidates mentally.
   - Implement only the most promising minimal change for that iteration.
   - Do not rewrite the whole reward unless the previous reward is clearly invalid.

2. **Training**
   - Use 4096 envs for main runs.
   - Start with 5M steps, then adjust based on convergence (see execution principles).
   - Render at least one side-view video and, when investigating geometry/contact, also render collision debug.
   - Launch training in background, then poll the stdout log file periodically (e.g., every 60s) to check progress.

3. **Analysis**
   - Read training stdout and metrics.
   - Inspect videos where available.
   - Determine whether failure is:
     - no stable contact,
     - only two-finger contact,
     - thumb/index/middle contact but no lift,
     - lift but no hold,
     - sliding,
     - palm/ring/pinky/support cheating,
     - action jitter or reward hacking,
     - collision/visual mismatch.

4. **Logging**
   Append to the task-specific changelog (cube -> `v2_iteration_docs/changelog.md`, can -> `v2_iteration_docs/can_grasp_changelog.md`) and do this on every iteration before starting the next one (参考已有日志风格，详细记录):
   - iteration id,
   - exact code changes,
   - exact command,
   - checkpoint source / restore path,
   - smoke test result,
   - score and diagnostic metrics (包含各 reward component 的均值/趋势变化),
   - `contact_duration_sec` first / last / max / best_step,
   - 各关键指标的趋势分析 (与前几轮对比，是否改善/恶化/持平),
   - video observations,
   - failure-mode analysis (详细分析当前瓶颈和失败原因),
   - implemented changes (本轮做了什么改动，为什么),
   - expected effect (预期效果),
   - actual effect (实际效果),
   - reward component definitions/weights/units,
   - 涉及的论文/算法思路 (如 EUREKA reward evolution, contact semantics, curriculum learning 等),
   - reward hacking risks,
   - **扰动(perturbation)配置与效果** (C22+必须记录):
     - 外力脉冲: enabled/disabled, magnitude, interval, min_hold_steps, 观测到的效果
     - 重力倾斜: enabled/disabled, max_rad, change_interval, min_hold_steps, 观测到的效果
     - 关节噪声: enabled/disabled, std, 观测到的效果
     - DR配置: 摩擦/质量/关节随机化范围, 观测到的影响
     - 扰动下策略鲁棒性评估: 扰动后持握是否稳定、掉落率变化、力平衡变化
   - next-step recommendation (下一轮最值得尝试的方向).

5. **Stop criteria**
   Stop if a policy holds the cube continuously for 30 seconds in video/metrics.
   Otherwise stop after continuation iteration count exceeds 30 and summarize why the system remains blocked.

## Paper-Inspired Methods To Use When Appropriate

### EUREKA-Style Reward Evolution

Use structured reward reflection:

- Keep reward components explicit and logged.
- Compare components by mean, variance, maximum value, and trigger frequency.
- Mutate weights, thresholds, windows, or activation gates in small steps.
- Prefer changes correlated with actual grasp behavior, not just higher score.

### Contact Semantics Instead Of Raw Tactile

Convert tactile/contact signals into low-dimensional semantics:

- per-finger contact flag,
- per-finger normal force approximation,
- primary three-finger contact count,
- contact duration,
- non-tip primary contact,
- palm/nonprimary contact,
- slip proxy from cube lateral velocity,
- three-finger force balance.

Use raw high-dimensional tactile only if semantic features are insufficient.

### Initialization Curriculum

Start from a feasible but nontrivial hand-object configuration:

- current default starts near the requested grasp pose,
- support holds the cube but does not collide with hand,
- cube jitter is initially disabled for a stable bootstrap,
- gradually reintroduce XY jitter, release randomness, friction/mass randomization, and harder starts only after stable three-finger hold emerges.

### Actor-Critic State Design

Actor should prioritize deployable signals:

- motor targets / joint positions,
- low-dimensional tactile semantics,
- previous action,
- cube relative pose if available in this sim task,
- support phase.

Critic may use privileged simulation signals:

- cube velocity,
- true contact diagnostics,
- lift height,
- slip and contact duration signals,
- domain randomization parameters if added later.

### Online Correction And Action Smoothness

Only add online tactile correction or stronger action smoothing if videos show:

- sliding after contact,
- force imbalance,
- high-frequency finger chatter,
- repeated contact-break-contact cycles.

Prefer simple action rate/accel penalties or EMA-style action smoothing before adding complex filters.

## Default Main Training Command

Use this as the baseline command unless the current iteration log explains a deliberate deviation:

```bash
cd /home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground
/home/ll/miniconda3/envs/aero_rl/bin/python learning/train_jax_ppo.py \
  --env_name=AeroCubeGraspV2ForceCoacd \
  --num_timesteps=5000000 \
  --num_evals=5 \
  --num_envs=4096 \
  --num_eval_envs=128 \
  --episode_length=800 \
  --unroll_length=20 \
  --num_minibatches=4 \
  --num_updates_per_batch=2 \
  --batch_size=4096 \
  --policy_hidden_layer_sizes=128,128 \
  --value_hidden_layer_sizes=128,128 \
  --num_videos=1 \
  --camera=side \
  --render_collision_debug=True
```

For the final 30-second hold check, use an evaluation/video horizon of at least 600 control steps (`30s / 0.05s = 600`).

## Current Baseline Assumptions

- Main environment: `AeroCubeGraspV2ForceCoacd`.
- Current collision: capsule/box primitive hybrid generated in `right_hand_v2_vertical_coacd.xml`.
- Current default hand shape and cube/support positions already implement the user-requested start.
- C21 achieved 37.89s stable hold with near-zero palm/nonprimary contact. DR and perturbations enabled from C22.

---

## 扰动(Perturbation)系统 (C22 实现, 2026-04-20)

### 实现方式
扰动在 `step()` 中实现，仅在支撑释放且持握稳定后才激活：

| 扰动类型 | 实现方式 | 激活条件 | 默认参数 |
|:---|:---|:---|:---|
| 外力脉冲 | `xfrc_applied` 施加在cube body上 | support_released & hold_steps≥100 & timer%40==0 | magnitude=0.08N |
| 重力倾斜 | 等效侧向力 F=m*g*sin(tilt) | support_released & hold_steps≥200 & timer%80==0 | max_rad=0.3 (~17°) |
| 关节噪声 | `_get_obs()` 中注入高斯噪声 | 始终启用 | std=0.01 rad |

### 扰动配置 (`perturbation_config`)
```python
perturbation_config=config_dict.create(
    external_force_enabled=True,
    external_force_magnitude=0.08,   # N
    external_force_interval=40,      # 每40步(2s)施加一次
    external_force_min_hold_steps=100,  # 持握≥5s后
    gravity_perturbation_enabled=True,
    gravity_tilt_max_rad=0.3,         # ~17度
    gravity_tilt_change_interval=80,  # 每80步(4s)改变一次
    gravity_tilt_min_hold_steps=200,  # 持握≥10s后
    joint_obs_noise_enabled=True,
    joint_obs_noise_std=0.01,         # rad
)
```

### 迭代调参指南
- 如果策略在扰动下频繁掉落: 降低 magnitude/max_rad, 增加 min_hold_steps
- 如果策略完全不受扰动影响: 增加 magnitude/max_rad, 减少 interval
- 重力倾斜模拟手腕翻转, 是 sim-to-real 关键维度
- 外力脉冲模拟碰撞/推动, 测试力封闭鲁棒性
- 关节噪声模拟传感器不确定性, 增强 sim-to-real 迁移能力

---

## Domain Randomization (DR) 启用计划（2026-04-20 用户确认）

### 启用条件
DR **不在** C20-C22 迭代中启用。仅当策略在固定物理下能稳定持握 20s+ 后才启用。

### DR 启用前必须修复的问题
1. **cube_friction 范围错误**: domain_randomize() 中 [0.1, 0.5] → 修正为 [1.0, 2.0]（围绕 XML 标称值 1.5）
2. **fingertip_friction 范围错误**: [0.5, 1.0] → 修正为 [1.0, 2.0]（围绕 XML 标称值 1.5）
3. **mass 范围**: 当前 [0.01, 0.05] 围绕 0.02 基本合理，可微调为 [0.015, 0.030]

### DR 分阶段启用顺序
**阶段1 (C23+ 策略持握≥20s后)**:
- 启用 `--domain_randomization`
- 只随机化: 摩擦 + 质量（最低风险）
- 修正后的范围: cube_friction ∈ [1.0, 2.0], fingertip_friction ∈ [1.0, 2.0], mass ∈ [0.015, 0.030]

**阶段2 (策略在DR阶段1下稳定后)**:
- 加入关节延迟随机化
- 加入观测噪声

**阶段3 (策略在DR阶段2下稳定后)**:
- 加入外力扰动 (±0.05N 随机脉冲)
- 加入执行动作噪声

### DR 相关注意事项
- 每次启用新 DR 维度必须从头训练（不从旧checkpoint恢复）
- DR 训练步数应至少 20M（DR 增加探索难度需要更多步数）
- 启用 DR 后必须对比有/无 DR 的关键指标变化

---

## 穿模修复记录（C20, 2026-04-20）

### 问题
用户视频检查发现手指capsule碰撞体与方块存在穿模（互相穿透）。

### 原因分析
1. `solref="0.01 1.2"`: dampratio=1.2 过阻尼，接触响应偏软
2. `solimp="0.95 0.995 0.0005"`: width=0.0005m (0.5mm) 穿透容差过大
3. `iterations=12, ls_iterations=16`: 求解器迭代次数偏少，接触约束求解精度不足
4. 方块无 solref/solimp 设置（使用 MuJoCo 默认值）

### 修复内容
| 参数 | 修改前 | 修改后 | 文件 |
|:---|:---|:---|:---|
| 手指 solref | 0.01 1.2 | 0.005 1.0 | right_hand_v2_vertical_coacd.xml |
| 手指 solimp | 0.95 0.995 0.0005 | 0.97 0.999 0.0001 | right_hand_v2_vertical_coacd.xml |
| 求解器 iterations | 12 | 20 | right_hand_v2_vertical_coacd.xml |
| 求解器 ls_iterations | 16 | 20 | right_hand_v2_vertical_coacd.xml |
| 方块 solref/solimp | 无(默认) | 0.005 1.0 / 0.97 0.999 0.0001 | small_cube.xml |

### 预期效果
- 接触响应时间从 10ms → 5ms（更快响应穿透）
- 穿透容差从 0.5mm → 0.1mm（5倍减少）
- 阻尼从过阻尼(1.2) → 临界阻尼(1.0)（更稳定）
- 求解器精度提升（20次迭代 vs 12次）

### 注意
穿模修复改变了接触物理，之前的 checkpoint 不能直接恢复使用，C20 必须从头训练。

---

## C20 迭代改动摘要 (2026-04-20)

### 奖励函数迭代
| component | 修改前 | 修改后 | 原因 |
|:---|:---|:---|:---|
| progressive_hold | sqrt(steps/50) cap=3.0 | steps/200 cap=5.0 (线性) | 14s→30s梯度不足 |
| sustained_hold_bonus | 不存在 | 新增: 10s/20s/30s阶梯奖励, scale=40 | 长时间持握无milestone激励 |

### 观测空间增强
| 新增obs | 公式 | +维度 |
|:---|:---|:---|
| hold_duration_normalized | clip(stable_hold_steps/600, 0, 1) | +1 |
| force_balance_obs | clip(1-rel_std, 0, 1) * clip(mean_f/0.1, 0, 1) | +1 |

**state_dim**: 44 → 46
