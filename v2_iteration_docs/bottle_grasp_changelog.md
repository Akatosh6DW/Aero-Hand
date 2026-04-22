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

## B05: 包裹手型 + 延后释放 (2026-04-22)

### 修改
- 将 bottle 的预抓取手型从 cube pinch 改成更接近圆柱包裹：
  - `index/middle/ring = 0.75`
  - `pinky = 0.65`
  - `thumb_abd = 1.25`
  - `thumb_mcp = 0.20`
- 将 `pre_grasp_fraction` 提高到 `0.70`，并暂时关闭 `lifted_grasp_fraction`。
- 将支撑释放改为更晚、更严格：
  - `release_after_sec: 3.0`
  - `force_release_after_sec: 4.2`
  - `random_release: 2.8-3.8s`
  - `min_release_active_fingers: 3`
  - `min_release_force: 0.16`
- 提高 `finger_participation / thumb_opposition / primary_finger_force / pre_release_grasp`。

### 训练
- 续训自：
  - `AeroBottleGraspV2Force-20260421-213819-B04_stabilityfix_nodr_4096/checkpoints/000001966080`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-143321-B05_wrap_release_nodr_4096`

### 结果
- `655360` 时是本轮最佳：
  - `reward=-43.32`
  - `contact_duration_sec=0.427`
  - `lift_success=12.18`
  - `nonfinite_state=0.023`
- 后续在 `1310720`、`1966080` 均明显回落。
- 离线评估（8 episodes）显示：
  - `lift > 2cm: 12.5%`
  - `post-release hold >= 0.5s: 0.0%`
  - `dropped after release: 100%`

### 结论
- 方向正确：抓到、抬起的概率明显好于 B04。
- 但仍然是“支撑在时能抬，支撑释放后马上掉”，说明还没有学会 unsupported hold。

## B06: 重新对位瓶身到包裹手型 (2026-04-22)

### 修改
- 将 bottle 和 support 的初始位置向手内侧前移、微降：
  - `cube_pos: [0.028, -0.075, 0.169]`
  - `support_pos: [0.028, -0.075, 0.1325]`
- 继续延后支撑释放：
  - `release_after_sec: 3.2`
  - `force_release_after_sec: 4.6`
  - `random_release: 3.0-4.0s`
  - `min_release_force: 0.20`
- 进一步提高：
  - `force_contact`
  - `thumb_opposition`
  - `primary_finger_force`
  - `pre_release_grasp`
  - `post_release_grasp / post_release_survival / post_release_pose_hold`

### 训练
- 续训自：
  - `AeroBottleGraspV2Force-20260422-143321-B05_wrap_release_nodr_4096/checkpoints/000000655360`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-144620-B06_align_releasehold_nodr_4096`

### 结果
- `655360` 时：
  - `reward=-43.65`
  - `contact_duration_sec=0.445`
  - `lift_success=11.14`
  - `nonfinite_state=0.016`
- 相比 B05，接触时长略长，但整体没有拉开明显优势。
- 离线评估与 B05 基本等价：
  - `lift > 2cm: 12.5%`
  - `post-release hold >= 0.5s: 0.0%`
  - `dropped after release: 100%`

### 结论
- “对位”本身是有帮助的，但问题不只是几何对位。
- 更深层的瓶颈是 bottle 仍然沿用 cube 的三指 pinch 判据。

## B07: Bottle 专用 wrap 判据 (2026-04-22)

### 修改
- 只在 `grasp_bottle_v2_force.py` 中加入 bottle 专用逻辑，不改 cube 文件：
  - `_reward_thumb_opposition`: 提高 ring 在对握中的权重；
  - `_reward_primary_finger_force`: 从三指 pinch 改为 `thumb+index+middle+ring` wrap；
  - `_reward_release_ready`: 改为更偏全手包裹的释放前判据；
  - `_is_grasp_ready_for_release`: 要求 thumb 在位，且 `index/middle/ring/thumb` 至少 3 指 active，合力更高才释放。

### 训练
- 续训自：
  - `AeroBottleGraspV2Force-20260422-143321-B05_wrap_release_nodr_4096/checkpoints/000000655360`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-145841-B07_wrapcriterion_nodr_4096`

### 结果
- `655360` 时：
  - `reward=-47.26`
  - `contact_duration_sec=0.436`
  - `lift_success=11.67`
  - `nonfinite_state=0.008`
- 后续依旧回落到 `reward≈-56~-59`。

### 结论
- bottle 专用 wrap 判据方向是对的，至少没有把训练搞炸。
- 但目前它还没有把 unsupported hold 从 0 拉起来。
- 当前最值得保留的 checkpoint 仍然是：
  - `B05_wrap_release_nodr_4096/checkpoints/000000655360`

## 当前阶段总结

- bottle 任务已经摆脱了“训练即 NaN”的阶段。
- 目前最佳候选是：
  - `logs/AeroBottleGraspV2Force-20260422-143321-B05_wrap_release_nodr_4096/checkpoints/000000655360`
- 当前客观状态：
  - 能形成接触并偶发抬起；
  - 但支撑释放后仍未出现可靠保持；
  - 还不能算“抓稳水瓶策略已完成”。

## B08: bottle 专用 hold 逻辑首次生效 (2026-04-22)

### 修改
- 在 `grasp_bottle_v2_force.py` 中首次完整接管 bottle 的：
  - `step` 里的 `stable_hold_steps`
  - `_get_reward`
  - `_get_diagnostics`
- 不再沿用 cube 的三指 pinch 逻辑，改为 index+middle+ring+thumb 的 wrap 逻辑。
- `palm_contact` / `nonprimary_contact` 从惩罚改为中性或轻正向信号。

### 训练
- 续训自：
  - `AeroBottleGraspV2Force-20260422-143321-B05_wrap_release_nodr_4096/checkpoints/000000655360`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-155517-B08_wrapholdlogic_nodr_4096`

### 结果
- `0 step`:
  - `reward=-16.11`
  - `contact_duration_sec=0.579`
  - `lift_success=12.27`
- `655360`:
  - `reward=-14.19`
  - `contact_duration_sec=0.599`
  - `lift_success=13.49`
- 后续到 `1310720+` 有明显回落。

### 结论
- 这是 bottle 线上一次实质性台阶。
- wrap-based hold logic 明显优于沿用 cube 的 pinch 逻辑。

## B09: 更晚 release + 低学习率保守续训 (2026-04-22)

### 修改
- 延后支撑释放：
  - `release_after_sec=4.2`
  - `force_release_after_sec=6.0`
  - `random_release=4.0-5.0s`
- 增大：
  - `pre_grasp_fraction=0.75`
  - `lifted_grasp_fraction=0.30`
- 提高 `stable_hold / progressive_hold / sustained_hold_bonus / post_release_survival / post_release_pose_hold`
- 降低学习率和熵，尽量避免后半段漂移。

### 训练
- 续训自：
  - `AeroBottleGraspV2Force-20260422-155517-B08_wrapholdlogic_nodr_4096/checkpoints/000000655360`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-161207-B09_late_release_wrapcurr_nodr_4096`

### 结果
- `0 step`:
  - `reward=-14.19`
  - `contact_duration_sec=0.582`
- `655360`:
  - `reward=-14.61`
  - `contact_duration_sec=0.589`
- `1966080`:
  - `reward=-26.34`
  - `contact_duration_sec=0.616`
- 仍然没有非零 `hold_success`。

### 结论
- B09 是目前“包裹行为最稳”的 checkpoint 基座。
- 但仍然没有穿过真正 unsupported hold 的门槛。

## B10: unsupported start curriculum 首次生效 (2026-04-22)

### 修改
- bottle reset 中，当 `lifted_reset=True` 时，直接把 support 隐藏：
  - `use_support=False`
  - `support_released=True`
- 这使 lifted curriculum 真正变成 unsupported curriculum。

### 训练
- 续训自：
  - `B09_late_release_wrapcurr_nodr_4096/checkpoints/000001966080`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-162304-B10_unsupportedstart_nodr_4096`

### 结果
- `0 step`:
  - `reward=-5.83`
  - `contact_duration_sec=0.711`
  - `lift_success=13.51`
- `655360`:
  - `reward=-11.73`
  - `contact_duration_sec=0.655`
- 后续继续回落。

### 结论
- unsupported curriculum 是目前第二个真正有效的大改动。
- 这一步把接触时长直接拉到当前最高一档。

## B11: 放宽 bottle 的 hold/termination 容差 (2026-04-22)

### 修改
- bottle 专用：
  - `reward_hold_position` 更宽松
  - `reward_post_release_survival` 更宽松
  - `_get_termination` 放宽到 `spawn_z - 0.06`

### 训练
- 续训自：
  - `B09_late_release_wrapcurr_nodr_4096/checkpoints/000001966080`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-163507-B11_relaxedhold_unsupported_nodr_4096`

### 结果
- `0 step`:
  - `reward=-4.71`
  - `contact_duration_sec=0.712`
- `655360` 后仍开始回落。

### 结论
- 容差放宽是对的，但单独不够。

## B12: hold buffer / palm-assisted hold (2026-04-22)

### 修改
- `stable_hold_steps` 不再一掉就清零：
  - 非 holding 时改为 `-2` 衰减而不是直接归零
- `hold_ready` 改为：
  - `wrap_count>=3`
  - 或 `thumb + >=2 wrap fingers + palm_contact`
- 诊断中的 `contact_duration_sec / three_finger_contact / hold_success` 也同步改为 bottle hold 判据。

### 训练
- 续训自：
  - `B09_late_release_wrapcurr_nodr_4096/checkpoints/000001966080`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-164848-B12_holdbuffer_nodr_4096`

### 结果
- `0 step`:
  - `reward=-5.03`
  - `contact_duration_sec=0.741`
  - `lift_success=13.42`
- `655360`:
  - `reward=-12.08`
  - `contact_duration_sec=0.665`
- 仍未出现非零 `hold_success`。

### 结论
- 这是目前按训练器统计最好的 bottle 接触时长：
  - `contact_duration_sec ≈ 0.74`
- 但仍明显距离 `30s hold` 很远。

## B13: 纯 unsupported hold 阶段 (2026-04-22)

### 修改
- 将当前 bottle 任务暂时切成纯 hold 课：
  - `support_enabled=False`
  - `pre_grasp_fraction=0.0`
  - `lifted_grasp_fraction=1.0`
  - `lifted_grasp_noise_scale=0.02`
- 即：所有 episode 都从已包裹、无支撑状态开始。

### 训练
- 续训自：
  - `B09_late_release_wrapcurr_nodr_4096/checkpoints/000001966080`
- 日志目录：
  - `logs/AeroBottleGraspV2Force-20260422-165804-B13_pure_hold_stage_nodr_4096`

### 结果
- `0 step`:
  - `reward=-5.00`
  - `contact_duration_sec=0.742`
  - `lift_success=13.43`
- 继续训练后依旧回落到 `0.65s` 左右，没有出现非零 `hold_success`。

### 结论
- 纯 hold 阶段没有把行为拉坏，但也没有让它自己跨过“长时稳握”这道坎。

## 最新结论

- 目前 bottle 线最有效的两个改动是：
  1. **B08: bottle 专用 wrap hold 逻辑**
  2. **B10: 真正的 unsupported start curriculum**
- 当前最强基座 checkpoint 仍建议使用：
  - `logs/AeroBottleGraspV2Force-20260422-161207-B09_late_release_wrapcurr_nodr_4096/checkpoints/000001966080`
- 在当前 bottle 环境代码下，这个 checkpoint 的起始评估大约能达到：
  - `reward≈-5`
  - `contact_duration_sec≈0.74`
  - `lift_success≈13.4`
- 但截至目前，**仍未训练出稳定 30s+ 的水瓶握持策略**。
