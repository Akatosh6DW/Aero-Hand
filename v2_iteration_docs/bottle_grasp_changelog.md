# Bottle Grasp Iteration Changelog

独立记录 550ml 空矿泉水瓶抓取任务的训练、奖励修改、DR 配置和结果。

## B01: 场景接入 + MJX 兼容修复 (2026-04-21)

### 已完成
- 新建 bottle 场景与物体 XML。
- 新建 bottle 专用环境 `AeroBottleGraspV2Force`。
- 新建 bottle 专用手模 `right_hand_v2_vertical_bottle.xml`。
- palm 不再使用单一大方块，而是基于凸分解结果提取的 4 个定向 box 近似。
- 修复 MJX 碰撞兼容问题：
  - 避免 `mesh` palm 碰撞；
  - 避免 `cylinder` bottle 主体碰撞；
  - bottle 主体改为 `capsule + capsule + sphere` 近似。

### 当前状态
- `registry.load("AeroBottleGraspV2Force")` 可成功加载。
- `reset -> step` 烟雾测试通过。

### 下一步
- 启动第 1 轮 bottle 训练，建立初始基线。

## B02: 初始穿模 / 显存问题修复 (2026-04-21)

### 发现的问题
- `home` 初始姿态下，index/middle/ring/pinky 的 distal 几何直接压进瓶身，接触深度约 8-25mm。
- 第 0 次 eval 出现极大 `force_overload`，`reward=nan`。
- 首轮训练 `B01_bootstrap_4096` 在第一次训练 epoch 因 `512,256,128` 网络过重而 OOM。

### 修复
- 重新打开四指预抓取姿态，并把瓶子/支撑台同步后移、上抬，避免 reset 初始穿模。
- bottle 专用 reward 放宽：
  - `force_overload_threshold: 1.2 -> 2.4`
  - `force_overload_soft_width: 0.5 -> 1.0`
  - `force_overload scale: -12 -> -4`
  - `soft_contact_fmax: 1.2 -> 2.2`
- 初期 DR 稍微放缓：
  - 外力 `0.08 -> 0.05`
  - 重力倾角 `0.35 -> 0.25`
  - 翻转扰动 `1.05 -> 0.85`
- bottle 训练默认网络改为 `(128, 128)`，并切到更轻的 PPO 显存配置。

### 下一步
- 以 B02 为新的可学习基线重新启动训练。
- 稳定拿到 2-5s 级别 contact_duration 后，再逐步把扰动和柔顺约束往上加。

## B03: 先关掉 DR，建立无扰动基线 (2026-04-21)

### 决策
- 先不叠加 DR。
- 原因：
  - 单环境 rollout 有限；
  - 带 DR 的 batched rollout 在第 8 步就出现非有限值；
  - 说明当前更需要先把基础抓取策略学稳，再逐步加 DR。

### 训练
- 运行：
  - `AeroBottleGraspV2Force`
  - `domain_randomization=False`
  - `num_envs=4096`
  - `batch_size=512`
  - `policy/value=(128,128)`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260421-204628-B03_nodr_4096`

### 已有产物
- checkpoint:
  - `000001310720`
  - `000002621440`
  - `000003932160`

### 当前观察
- 训练主循环可持续推进，checkpoint 正常落盘。
- 训练器自带 eval 仍持续打印 `reward=nan`，但 `metrics.csv` 能正常写出行。
- `eval` 里的部分字段可读性较差，尤其：
  - `support_released` 实际更像“释放后的累计步数”
  - `lift_height` 仍是 `nan`
- 训练结束阶段出现 Brax `assert_is_replicated`，属于训练器收尾断言问题，不影响已写出的 checkpoint。

### 下一步
- 用最新 checkpoint 做离线验证，不再只依赖 train 内置 eval。
- 在离线验证确认“能包裹 / 能释放后不立刻掉落”之后，再把 DR 分阶段加回去。

## B04: 稳定性修复与 NaN 防护 (2026-04-21)

### 新增防护
- 在环境 `step` 中加入非有限状态检测：
  - `qpos/qvel/ctrl`
  - `cube_pos/cube_linvel`
- 一旦检测到非有限状态：
  - 当步直接 `done=1`
  - 返回上一个有限 `state.data/state.obs`
  - 给予额外 termination 惩罚
  - 记录 `diagnostic/nonfinite_state=1`

### 控制侧收紧
- bottle 动作尺度收小：
  - `[0.16, 0.38, 0.40, 0.40, 0.22, 0.22]`
  - → `[0.08, 0.18, 0.22, 0.22, 0.14, 0.14]`
- 新增动作幅度裁剪：
  - `max_abs_action=0.65`
- 新增相邻控制目标变化限幅：
  - `motor_delta_clip=[0.03, 0.05, 0.06, 0.06, 0.05, 0.05]`

### 目的
- 即使当前策略还不会抓，也不要再把整批 rollout 污染成 `NaN`。
- 让训练能在“失败但有限”的分布上继续学习。

### B04 训练结果
- 运行：
  - `domain_randomization=False`
  - `num_envs=4096`
  - `batch_size=512`
  - `policy/value=(128,128)`
  - `num_timesteps=1_000_000`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260421-213819-B04_stabilityfix_nodr_4096`

### 关键结果
- 训练全程未再出现环境级 `reward=nan` 崩坏。
- 第 0 次 eval：
  - `reward=-53.33`
  - `contact_duration_sec=0.299`
  - `nonfinite_state=0.0117`
- 第 655360 步 eval：
  - `reward=-46.19`
  - `contact_duration_sec=0.439`
  - `nonfinite_state=0.0`
- 后续 eval：
  - `1310720: reward=-59.33`
  - `1966080: reward=-58.17`

### 当前判断
- 稳定性修复有效，训练已重新进入“可迭代”状态。
- 但抓取能力仍弱，尚未形成可靠抬起和释放后保持。
- 下一轮应继续在 no-DR 下微调初始姿态 / release 条件 / 早期抓握奖励，而不是立刻加 DR。
