# Can Grasp Current Memory

整理时间：2026-04-24

这份文档用于收束当前 can 抓取线的有效记忆，方便后续继续自动迭代时直接接上，不再重复排查环境、配置漂移和已证伪方向。

## 目标与约束

- 任务目标：在当前确认过的 can 初始状态上，实现稳定 `30s+` unsupported hold。
- 初始状态约束：沿用当前 scene / pose，不要随意改动初始手型、物体位姿、支撑位置。
- 记录约束：can 线单独维护，不与 cube 或 bottle 线混写。
- 工作方式：开始训练后连续自主迭代，优先依据 `metrics.csv`、stdout、评估结果和失败模式做小步修正。

## 训练工作流约束

- Python 环境必须使用 `aero_rl`：`/root/miniconda3/envs/aero_rl/bin/python`
- 必须设置本地包路径：`PYTHONPATH=/root/autodl-tmp/Aero-Hand/sim_rl/mujoco_playground`
- 训练型无头迭代统一使用 `--num_videos=0`，避免 render/OpenGL 问题打断训练
- 解释器正确和导入本地 `mujoco_playground` 是两个独立条件，缺一不可

## 当前保留的代码状态

当前仓库最终保留的是 `CAN62` 状态，不保留 `CAN63/CAN64/CAN65` 的失败试验改动。

### 主控制文件

- 环境文件：`sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_can_v2_force.py`
- 训练入口：`sim_rl/mujoco_playground/learning/train_jax_ppo.py`

### 当前有效逻辑

- 保留 honest post-release hold count
- `is_holding` 必须包含 `support_released`
- `stable_hold_steps` 只在 `support_released=True` 后累计
- `progressive_hold` 与 `sustained_hold_bonus` 只在 released phase 发放
- `default_config()` 已修复缩进损坏，当前可被真实导入验证

### 当前默认配置关键值

这些值对应 `CAN59` 主线 handover config，也是 `CAN62` 继续使用的值：

- `release_after_sec = 10.2`
- `release_ramp_sec = 1.8`
- `force_release_after_sec = 14.0`
- `min_release_force = 0.125`
- `post_release_joint_palm_hold = 980.0`
- `cradle_lock = 760.0`
- `pre_release_grasp = 160.0`
- `post_release_grasp = 340.0`
- `post_release_survival = 1560.0`

## 已验证的 continuation 基线

- 基准 checkpoint：`sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-044958-CAN59_soft_cradle_sliplock/checkpoints/000002621440`
- 当前最好且最终保留的续训切片：`sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-101845-CAN62_honest_hold_can59cfg`

## 本轮实验结论

### CAN61b

- 改动：只把 post-release hold count 改得更诚实
- 结果：`contact_duration_sec = 10.65`
- 结论：单独改 hold count 不会自动提升 unsupported hold，而且当时还混入了 config 漂移

### CAN62

- 改动：在恢复 `CAN59` 默认配置后，保留 honest hold count
- 结果：`contact_duration_sec = 11.059`
- 结论：honest hold count 本身不是核心退化源，但也没有带来 30s 突破

### CAN63

- 改动：增强 post-release anti-slip 权重
- 结果：`contact_duration_sec = 11.056`
- 结论：只加大 anti-slip 强度无法抬出 `11s` 平台

### CAN64

- 改动：让 slip penalty 在 released phase 持续激活
- 结果：`contact_duration_sec = 11.057`
- 结论：release phase 持续 slip 惩罚也没有改善时长

### CAN65

- 改动：撤回失败 reward 改动，只测试 release 后动作降幅 `0.88`
- 结果：`contact_duration_sec = 10.978`
- 结论：release 后动作 damping 会伤害当前 retention，表现更差

## 当前判断

- can 线目前稳定平台大约在 `11s` unsupported hold
- 距离 `30s+` 目标仍然有明显差距
- 继续只在现有 reward 权重附近微调，边际收益已经很低
- 现阶段最稳妥的保留策略是：代码保留 `CAN62`，训练从 `CAN59` 基准 checkpoint 继续

## 已证伪方向

- 只把 hold count 改诚实
- 只提高 post-release anti-slip 权重
- 让 slip penalty 在整个 released phase 持续激活
- release 后动作收敛 / 动作降幅

## 排障经验

- 对这个仓库，真实 import 检查比编辑器错误提示更可靠，特别是缩进和局部语法损坏
- `Env 'AeroCanGraspV2Force' not found in default configs.` 往往说明导入到了外部安装版 `mujoco_playground`
- render / EGL 问题不要和训练评估绑死，训练型迭代优先走 `--num_videos=0`
- 如果一次试验更差，不要把失败试验代码继续挂在主线里，最终应回到最佳已验证切片

## 下一轮建议

- 不要继续重复 reward 小权重搜索
- 优先考虑更结构性的突破口：
  - handover / release curriculum
  - release 后控制动态而非单纯 reward 加权
  - 针对 `11s` 附近失败模式做离线切片分析，区分是托持丢失、包裹松脱还是 release 后动作自激导致滑落

## 对应记录文件

- 迭代日志：`v2_iteration_docs/can_grasp_changelog.md`
- repo 级简记：`/memories/repo/aero_hand_can_training.md`
