# Aero-Hand Current Memory 2026-04-28

这份文档用于保存当前最重要的工作记忆，方便后续继续迭代时直接接手。

## 1. 当前工作范围

- 工作区根目录：`/home/ll/SRTP/Aero-Hand`
- 当前并行关心的三条线：
  1. `cube` 抓取 RL 主线
  2. `can` 抓取 RL 历史主线
  3. 手部碰撞模型，尤其是 `thumb tip` 与掌心有效接触面

## 2. 方块任务当前主线

### 2.1 关键环境文件

- 训练环境主文件：
  - [grasp_cube_v2_force.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py)
- 当前训练用手部 XML：
  - [right_hand_v2_vertical_bottle.xml](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand_v2_vertical_bottle.xml)
- 方块迭代日志：
  - [v2_iteration_docs/changelog.md](/home/ll/SRTP/Aero-Hand/v2_iteration_docs/changelog.md)

### 2.2 当前最优已验证 run

- 当前 `cube` 主线最佳仍是 `C124`
- logdir:
  - `logs/AeroCubeGraspV2ForceCapsuleBottlePalmQbr-20260428-013906-C124_long_dr_lr7p25e5_upd3_force3p2_pose50_from_C106best_relmax2p45_postgrasp135_triad18_force72_capsule_bottlepalm_2048`
- 关键指标：
  - `first = 24.1273s`
  - `last = 25.1686s`
  - `max = 25.1686s`
  - `best_step = 2949120`
- 真实成功指标始终只看：
  - `eval/episode_diagnostic/contact_duration_sec`

### 2.3 当前主线有效配置结论

`CubeGraspV2ForceCapsuleBottlePalmQbr` 有类级 override，很多默认配置改动不一定真正生效。当前必须以类内有效值为准。

当前确认有效的关键值：

- `three_finger_proximity = 18.0`
- `primary_finger_force = 72.0`
- `post_release_grasp = 135.0`
- `pre_release_grasp = 35.0`
- `post_release_pose_hold = 50.0`
- `hold_position = 45.0`
- `stable_hold = 185.0`
- `force_balance = 28.0`
- `random_release_min_sec = 1.5`
- `random_release_max_sec = 2.45`
- `release_ramp_sec = 0.5`
- `force_release_after_sec = 3.2`
- `finger_active_threshold = 0.08`
- `force_contact_threshold = 0.06`

### 2.4 当前重要结论

最近几轮的结论已经比较明确：

- `C133`：
  - 把 `clean_gate` 从 `1.0 - cheat_contact` 放松到 `1.0 - 0.5 * cheat_contact`
  - 负样本，不能留在主线
- `C134`：
  - `min_release_active_fingers: 2 -> 3`
  - 全程接触形态干净，但 `last = 24.4577s`
  - 低于 `C124`
  - 结论：更硬的 “三指都活跃再 release” 没有带来提升
- `C135`：
  - 回到 `min_release_active_fingers = 2`
  - `min_release_force: 0.10 -> 0.11`
  - 全程接触形态干净，但 `last = 24.8409s`
  - 仍低于 `C124`
  - 结论：问题不主要在 release force gate

### 2.5 当前平台判断

当前 `cube` 线已经在 **DR 开启** 条件下进入平台区。

平台特征：

- `palm_contact / nonprimary_contact / non_tip_primary_contact / slip_event / drop` 基本长期为 `0`
- 奖励组件如：
  - `post_release_survival`
  - `stable_hold`
  - `progressive_hold`
  - `sustained_hold_bonus`
  已经很高
- 但真实指标仍卡在 `24~25s`

当前判断：

- 不是单纯 reward 太小
- 不是脏接触作弊
- 不是简单 release gate 太松或太紧
- 更像是：
  - `release 后真实受力闭合不够强`
  - `食指/中指压掌心的承托结构还不够实`
  - `策略更像会“干净抓住”，但不会“在 DR 扰动下长期稳住”`

## 3. 方块任务视频状态

最新 `C135` 已生成检查视频，放在 `tempvideo`：

- [C135_side_stl.mp4](/home/ll/SRTP/Aero-Hand/tempvideo/C135_side_stl.mp4)
- [C135_side_collision.mp4](/home/ll/SRTP/Aero-Hand/tempvideo/C135_side_collision.mp4)
- [C135_palm_stl.mp4](/home/ll/SRTP/Aero-Hand/tempvideo/C135_palm_stl.mp4)
- [C135_palm_collision.mp4](/home/ll/SRTP/Aero-Hand/tempvideo/C135_palm_collision.mp4)

说明：

- 环境里现成 camera 是 `side` 和 `palm`
- `palm` 视角是当前最适合看“食指把物体压向掌心、大拇指下托”的检查视角

## 4. Can 任务当前记忆

### 4.1 关键文件

- can 环境主文件：
  - [grasp_can_v2_force.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/grasp_can_v2_force.py)
- can 日志：
  - [v2_iteration_docs/can_grasp_changelog.md](/home/ll/SRTP/Aero-Hand/v2_iteration_docs/can_grasp_changelog.md)

### 4.2 can 当前记忆摘要

根据用户较早明确说明与已有日志：

- 慢 `release ramp` 已被证伪，不要沿那条线继续
- 曾经有效的一条结构改动是：
  - `can 初始位` 和 `support 位` 同时下调 `2mm`
- 用户对 can 的目标始终是：
  - 先稳定 unsupported 明显超过 `20~30s`
  - 再启用 DR 做鲁棒性
- 用户明确强调：
  - 不要把“thumb/index 不接触”写成硬约束
  - 形态目标是：
    - 食指主导把圆柱压向掌心
    - 大拇指在下方托举

### 4.3 can 最近已记录的历史结论

`can_grasp_changelog.md` 当前尾部保留的较新记录里：

- `CAN190`
  - `first = 2.1572`
  - `last = 1.9301`
- `CAN191`
  - 轻微缩短 `probe_hold_sec`
  - 负样本
- `CAN192`
  - 增加 `release_ready_sec`
  - 负样本

这说明当时那条 can 主线仍远未接近目标时长。

## 5. 手部碰撞模型当前状态

### 5.1 当前训练/查看都在用的 XML

- [right_hand_v2_vertical_bottle.xml](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand_v2_vertical_bottle.xml)

### 5.2 掌心与摩擦

当前 XML 里：

- 掌心使用 `box + ellipsoid` 混合拟合
- 掌心与相关接触面摩擦已经提高过
- 这部分是为了让“物体被压向掌心”的接触更真实、更容易稳定握持

### 5.3 大拇指最新修正

最新已经直接修改了 `right_thumb_tip` 这一组碰撞体：

- `right_thumb_tip_capsule_000`
- `right_thumb_tip_capsule_001`
- `right_thumb_tip_capsule_002`
- `right_thumb_tip_capsule_003`
- `right_thumb_tip_capsule_004`
- 新增 `right_thumb_tip_capsule_005`

本次调整目的：

- 不再让 thumb tip 掌侧接触面过短、过碎
- 把真正握持时会碰到物体的掌侧接触面连成更连续的一段
- 解决用户反复指出的：
  - “大拇指指尖以下还是空的”
  - “接触面不在真正有用的位置”

### 5.4 当前编辑器入口

编辑器：

- [edit_grasp_initial_state.py](/home/ll/SRTP/Aero-Hand/edit_grasp_initial_state.py)

当前支持：

- `F6` 切换 STL 显示
- `F7` 切换 collision 显示
- 新增快速检查入口：
  - `--inspect-thumb-tip`

直接检查 thumb tip overlay：

```bash
cd /home/ll/SRTP/Aero-Hand
/home/ll/miniconda3/envs/aero_rl/bin/python edit_grasp_initial_state.py --inspect-thumb-tip
```

这会自动：

- `scene = can_330ml`
- `mode = overlay`
- `camera = palm`
- `show_sites = True`

## 6. 当前重要脚本与自动化

### 6.1 方块自动迭代脚本

- [autonomous_cube_qbr_iteration.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/scripts/autonomous_cube_qbr_iteration.py)

它目前已经支持：

- 跳过已记录的 run
- 自动：
  - 改单变量
  - smoke
  - 发训练
  - 等 `metrics.csv`
  - 读 `first / last / max / best_step`
  - 记入 changelog

### 6.2 checkpoint 评估脚本

- [eval_aero_jax_checkpoint.py](/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/scripts/eval_aero_jax_checkpoint.py)

## 7. 当前最建议的后续方向

### 对 cube

不要继续优先做这些：

- 更硬的 `min_release_active_fingers`
- 更硬的 `min_release_force`
- 继续给已有 long-hold 奖励加码
- 放松 `clean_gate`

更值得做的方向：

1. 继续看视频，确认是否仍然是拇指主导、食指/中指承托不足
2. 优先修正有效接触几何，而不是继续加 reward
3. 若继续改 reward / gate，优先选更直接影响 `post-release force closure` 的窄结构项

### 对 can

如果后面重新接 can 线：

1. 先看最新几何与碰撞模型是否更适合“食指压掌心、拇指托举”
2. 先不要急着重新开大规模 DR
3. 先在无 DR 下把真实 unsupported 做长

## 8. 硬约束回顾

- 真实成功指标优先看 `contact_duration_sec`
- 不能用总 reward 代替成功
- 方块日志写入：
  - [v2_iteration_docs/changelog.md](/home/ll/SRTP/Aero-Hand/v2_iteration_docs/changelog.md)
- can 日志写入：
  - [v2_iteration_docs/can_grasp_changelog.md](/home/ll/SRTP/Aero-Hand/v2_iteration_docs/can_grasp_changelog.md)
- 不要把 can 与 cube 的日志混写

