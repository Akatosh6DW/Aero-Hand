# Can Grasp Iteration Changelog

独立记录 330ml Sleek Can（摩登罐）抓取任务的需求、场景修改、后续训练设想与实验结果。
此文件与方块线、矿泉水瓶线分开维护，不混写。

## CAN01: 330ml Sleek Can 场景接入 (2026-04-22)

### 用户需求
- 抓取对象从水瓶切换为 330ml 摩登罐（Sleek Can）。
- 几何规格：
  - 高度：146 mm
  - 罐身直径：57 mm
  - 顶盖直径：52 mm
- 抓取姿势依旧参考人手的圆柱包裹式抓握，而不是方块 pinching。
- 由于易拉罐容易变形，策略后续应避免过大的法向挤压力。
- 当前阶段先把 scene 改好，方便用户用 `view.py` / `edit_grasp_initial_state.py` 检查并手动设初始参数。

### 本轮已完成
- 新建 can 物体 XML：
  - `xmls/empty_can_330ml_sleek.xml`
- 新建 can 场景 XML：
  - `xmls/scene_mjx_grasp_can_330ml.xml`
- 新建 can 专用环境入口：
  - `AeroCanGraspV2Force`
- `view.py` / `edit_grasp_initial_state.py` 已加入 `can_330ml` 场景入口，供后续可视化与初始姿态编辑。

### 当前建模假设
- 罐体继续沿用现有通用命名 `cube` / `cube_freejoint`，以兼容 reward、编辑器和通用工具链。
- 物理几何采用“主 capsule + 视觉顶盖/底盖/标签带”的轻量近似。
- 为了体现“罐体容易变形、不能大力捏”，当前接触参数比 bottle 更软，force overload 阈值也更低。
- 用户尚未给定空罐/满罐重量，因此此版先按 30 g 轻质场景近似，用于几何与初始抓取检查。

### 后续待确认
- 若目标是满罐 330 ml 饮料，需要把质量和惯量单独改为约 0.34-0.37 kg 量级重新评估。
- 若后续训练发现 rigid + soft-contact 仍不足以表达“易变形不能大力捏”，再考虑加入更强的 force penalty / release gate / tactile over-compression shaping。
- 用户检查完初始位姿后，再确定 `spawn/support/home keyframe` 的最终版本。

## CAN02: 改为 40mm x 60mm 小圆柱，并复核拇指 flex 代理 (2026-04-23)

### 用户需求
- 当前圆柱太大，手抓不住。
- 将待抓取物改为：
  - 直径：40 mm
  - 高度：60 mm
- 进一步检查手模里拇指 `flex` 自由度/系数是否符合 `handinformation` 里的数据手册、URDF/STL。

### 本轮已完成
- `empty_can_330ml_sleek.xml` 里的主物体从 146 mm capsule 改为横放 `40 mm x 60 mm` cylinder 代理。
- `scene_mjx_grasp_can_330ml.xml` 的 `home` 位姿、统计中心和抓取 band 已同步缩到小圆柱尺寸。
- `grasp_can_v2_force.py` 的 spawn 高度、接触包围盒和初始 pre-grasp pose 已同步到新尺寸。

### 拇指 flex 复核结论
- `handinformation/五指灵巧手用户手册.pdf` 给出的拇指弯曲段范围是 `30°/45°` 量级，而不是由拇指外展角直接决定。
- `handinformation/URDF/右/qbr/urdf/qbr.urdf` 与 `qbr.csv` 里的 `j2/j3` 也表明拇指弯曲链是独立于 `thumb_rot` 的一条链。
- `right_hand_v2_vertical_bottle.xml` 之前使用了训练近似 `thumb_cmc_flex = 0.16 * thumb_cmc_abd`，这更偏向策略工程近似，不够像真实手。
- 当前 can/bottle 共用手模已改为更接近硬件语义的代理：
  - `hw_thumb_flex` 仍驱动 `right_thumb_mcp`
  - `right_thumb_cmc_flex` 改为被动联动 `≈ (30° / 45°) * right_thumb_mcp = 0.6667 * right_thumb_mcp`

### 风险说明
- 这次拇指 flex 修正优先服务当前 can/bottle 场景与初始姿态编辑，更贴近 `handinformation`。
- 老的 V2 cube/coacd 训练线历史上围绕 `0.16 * abd` 调过策略；若后续要直接复用那些旧 checkpoint，需要重新核对兼容性。

## CAN03: 记录用户确认的圆柱初始位姿 (2026-04-23)

### 用户提供的圆柱位姿
- `cube pos = [0.013113, -0.040712, 0.132458]`
- `quat_wxyz = [0.707913, -0.007449, 0.706260, 0.000628]`

### 本轮已完成
- 将这组圆柱位姿写回 `scene_mjx_grasp_can_330ml.xml` 的 `home` keyframe。
- 同步更新了 scene 里的 `cube_support` 基准位置与 `mpos`，使支撑台继续在圆柱正下方。
- 同步更新 `grasp_can_v2_force.py` 的 baseline `spawn_config.cube_pos` 与 `support_config.support_pos`。
- 圆柱任务的 baseline 手型沿用当前 scene 中已经保存的那组手指角度：
  - `index = [0.3400, 0.3145]`
  - `middle = [0.3520, 0.3256]`
  - `ring = [0.3380, 0.3127]`
  - `pinky = [0.3000, 0.2775]`
  - `thumb = [abd=1.18, cmc_flex=0.12, mcp=0.18]`

### 说明
- 这次我只更新了 baseline 起点，没有额外改 `cube_jitter` / `pre_grasp_noise_scale` 等训练噪声项。
- 如果你希望“每次 reset 都严格从这组姿态开始”，下一步我可以把这些 noise/jitter 一并清零。

## CAN04: 用户确认新的最终训练目标 (2026-04-23)

### 新目标
- 在当前脚本提供的圆柱 / 易拉罐初始姿态上继续训练，不再回到旧矿泉水瓶任务。
- 训练目标不是短时接触，而是：
  - **稳定持握 30s+ 不掉落**
  - **不能靠过大法向挤压造成易拉罐变形**
  - **握住后在晃动/扰动下仍不掉落**
  - **包含 DR 与扰动鲁棒性，不只是固定物理成功**
- 训练过程必须自主迭代：
  - 每轮读 `metrics.csv` / stdout / 视频或离线评估
  - 按失败模式修改 reward、release 逻辑、初始状态课程、扰动时序
  - 不能只是重复同一条训练命令

### 当前状态判断
- `AeroCanGraspV2Force` 已完成场景、初始位姿、拇指 flex 代理与更软接触参数接入。
- 但截至当前，还**没有**一个真正训练过并达到 `30s+` 的 can checkpoint。
- 后续 can 训练要与 cube 线严格分开记录，不改坏 cube 主线。

## CAN05: cube 完成后切入当前 pose 的 can/cylinder 基线 (2026-04-23)

### 前置条件
- cube 的当前手参数适配线已完成 `30s+`：
  - `AeroCubeGraspV2ForceCoacdQbr`
  - `C52b_qbr_middle_bias_8192`
- 按用户流程，cube 达标后自动切换到当前圆柱 / 易拉罐任务。

### 本轮策略
- 保持当前脚本里确认过的圆柱初始姿态不变。
- 先读取当前 `AeroCanGraspV2Force` 的基线表现，再按失败模式决定是：
  - 先收紧“易变形不能大力捏”的 force shaping
  - 还是先改 unsupported hold / release / wrap reward
- 训练与日志继续和 cube 线隔离记录。

### CAN05 结果
- `CAN05_currentpose_baseline`
  - 直接用“unsupported start + 已开 DR”的当前 can 默认配置起跑。
  - 首个 eval:
    - `avg_episode_length ≈ 100`
    - `contact_duration = 0`
    - `drop = 100%`
    - `lift_success ≈ 0.78`
  - 判断:
    - 这不是一个可学的起点；
    - 说明当前圆柱任务不能一开始就让策略在无支撑 + 扰动条件下学。

## CAN06: support + no-DR baseline (2026-04-23)

### 改动
- `spawn.support_enabled: False -> True`
- `reset.pre_grasp_fraction: 0.0 -> 1.0`
- `reset.lifted_grasp_fraction: 1.0 -> 0.0`
- `support.random_release: True -> False`
- `support.release_after_sec: 3.8 -> 4.2`
- `support.force_release_after_sec: 5.2 -> 6.0`
- 第一阶段关闭 DR：
  - `external_force_enabled = False`
  - `gravity_perturbation_enabled = False`
  - `orientation_flip_enabled = False`
- 将 `palm_contact / nonprimary_contact` 从正奖励改回惩罚，避免奖励函数鼓励掌心蹭住物体。
- `force_overload` 惩罚加重，继续维护“易拉罐不能大力捏”的目标。

### 结果
- `CAN06_support_nodr`
  - 首个 eval:
    - `avg_episode_length ≈ 143`
    - `contact_duration = 0`
    - `drop = 100%`
    - `lift_success ≈ 44.9`
    - `palm_contact ≈ 64.5`
    - `normal_force_mean ≈ 0.52`
- 判断:
  - 比 `CAN05` 前进了一步：策略已经学到“靠掌心去兜住圆柱并尝试抬起”。
  - 但这条线**明显在走 palm cheat**，还没有形成我们要的轻力包裹抓握。
  - 下一步不该继续原配方长跑，而要继续改：
    - 更强抑制 palm cheating
    - 强化真正 wrap hold 的判据/奖励
    - 让主要接触从“掌心兜”转为“手指包裹”

  ## CAN61b: honest post-release hold count (2026-04-24)

  ### 改动
  - 将 `stable_hold_steps` 的累计改为只在 `support_released=True` 后生效。
  - 将 `progressive_hold / sustained_hold_bonus` 改为只在 support release 后发放。
  - 保持当前 can 初始状态不变，继续从 `CAN59_soft_cradle_sliplock` checkpoint 续训。

  ### 训练与环境修复
  - 训练环境固定为 `aero_rl`：
    - `/root/miniconda3/envs/aero_rl/bin/python`
  - 强制使用本地仓库包，避免误导入外部安装版 `mujoco_playground`：
    - `PYTHONPATH=/root/autodl-tmp/Aero-Hand/sim_rl/mujoco_playground`
  - 为无头环境补齐 OpenGL 依赖，并确认 `train_jax_ppo.py` 在 MuJoCo/Brax import 前设置渲染环境变量。

  ### 结果
  - `reward`: `34829.891 -> 30029.562`
  - `contact_duration_sec`: `11.05 -> 10.65`

  ### 结论
  - 单独把 hold count 改诚实，并不会自动把 unsupported hold 往上推。
  - 这一轮同时暴露出一个混杂因素：当时默认 can config 已经漂到更接近 `CAN60`，不能直接拿来判断 `CAN59` 主线是否失效。

  ## CAN62: honest hold count + restored CAN59 config (2026-04-24)

  ### 改动
  - 修复 `grasp_can_v2_force.py` 里 `default_config()` 的缩进损坏，恢复可导入状态。
  - 将 can 默认 release / reward 配置恢复到 `CAN59` checkpoint 对应值：
    - `release_after_sec = 10.2`
    - `release_ramp_sec = 1.8`
    - `force_release_after_sec = 14.0`
    - `min_release_force = 0.125`
    - `post_release_joint_palm_hold = 980.0`
    - `cradle_lock = 760.0`
    - `pre_release_grasp = 160.0`
    - `post_release_grasp = 340.0`
    - `post_release_survival = 1560.0`
  - 保留 `CAN61b` 的 honest post-release hold count 逻辑。
  - 为避免无头渲染再次打断训练，本轮使用 `--num_videos=0` 只跑训练与评估。

  ### 结果
  - 日志目录：
    - `sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-101845-CAN62_honest_hold_can59cfg`
  - `reward`: `31326.922`
  - `contact_duration_sec`: `11.059`
  - `hold_success`: `0.0`
  - `support_released`: `38.891`
  - `slip_event`: `1.070`
  - `post_release_survival_reward`: `4104.167`

  ### 结论
  - honest hold count 在真正 `CAN59` handover config 下基本与 `CAN59` 持平，说明它本身不是主要退化源。
  - 但时长仍卡在 `11s` 左右，30s 目标没有任何实质突破。

  ## CAN63: stronger post-release anti-slip weights on CAN59 release (2026-04-24)

  ### 改动
  - 保持 `CAN59` release 时序不变，只把若干 post-release 权重向 `CAN60` 靠拢：
    - `post_release_joint_palm_hold = 1040.0`
    - `cradle_lock = 820.0`
    - `pre_release_grasp = 150.0`
    - `post_release_grasp = 360.0`
    - `post_release_survival = 1650.0`

  ### 结果
  - 日志目录：
    - `sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-103122-CAN63_postrelease_antislip_can59release`
  - `reward`: `31294.102`
  - `contact_duration_sec`: `11.056`
  - `hold_success`: `0.0`
  - `support_released`: `38.891`
  - `slip_event`: `1.102`

  ### 结论
  - 单纯提高 post-release anti-slip 权重，没有把时长抬出 `CAN62`/`CAN59` 平台。
  - 问题更像是 reward 激活形态或控制动态，而不是“同样信号强度还不够大”。

  ## CAN64: keep slip penalty active through release phase (2026-04-24)

  ### 改动
  - 将 `post_release_slip` 从 `released_gate * hold_gate` 改为只乘 `released_gate`，避免物体一开始滑落时惩罚同步衰减。
  - 其他保持 `CAN63` 配置不变。

  ### 结果
  - 日志目录：
    - `sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-104427-CAN64_releasephase_slippenalty`
  - `reward`: `31240.365`
  - `contact_duration_sec`: `11.057`
  - `hold_success`: `0.0`
  - `support_released`: `38.961`
  - `slip_event`: `1.000`
  - `post_release_survival_reward`: `4309.780`
  - `cradle_lock`: `459.141`
  - `post_release_joint_palm_hold`: `842.793`

  ### 结论
  - 把 slip penalty 提前持续激活也没有带来时长提升。
  - 这个方向不足以解释当前 `11s` 平台。

  ## CAN65: release-phase action damping (2026-04-24)

  ### 改动
  - 先撤回 `CAN63/CAN64` 的失败 reward 改动，回到 `CAN62` 的 reward 基线。
  - 仅新增一项 release 后动作收敛实验：
    - `support_released` 后将动作幅度乘以 `0.88`

  ### 结果
  - 日志目录：
    - `sim_rl/mujoco_playground/logs/AeroCanGraspV2Force-20260424-105959-CAN65_release_action_damp`
  - `reward`: `31056.953`
  - `contact_duration_sec`: `10.978`
  - `hold_success`: `0.0`
  - `support_released`: `38.875`
  - `slip_event`: `0.977`
  - `post_release_survival`: `3587.697`
  - `cradle_lock`: `382.671`
  - `post_release_joint_palm_hold`: `713.552`

  ### 结论
  - release 后动作降幅会直接削弱当前已经学到的 unsupported retention 形态，表现比 `CAN62` 更差。
  - 因此仓库最终保留 `CAN62` 状态：
    - honest post-release hold count 保留
    - `CAN59` default config 保留
    - 不保留 `CAN63/CAN64/CAN65` 的失败试验改动

  ## 当前判断

  ### 已证伪的方向
  - 只把 hold count 改诚实，不会自动提升 unsupported hold。
  - 只提高 post-release anti-slip 权重，不会突破 `11s` 平台。
  - 让 slip penalty 在整个 released phase 持续激活，也没有带来时长提升。
  - release 后动作收敛会伤害当前 grasp retention。

  ### 当前最好可继续主线
  - 代码状态：`CAN62`
    - honest post-release hold count
    - `CAN59` handover / reward config
  - 当前最好续训切片：
    - 从 `CAN59_soft_cradle_sliplock` checkpoint 继续
    - 使用上面的 `CAN62` 代码状态

  ### 当前结论
  - can 线已经稳定达到约 `11s` unsupported hold，但离 `30s+` 仍有很大差距。
  - 继续只在现有 reward 权重附近微调，边际收益已经很低；下一轮若还要冲 30s，需要换更结构性的突破口，而不是重复当前局部权重搜索。
