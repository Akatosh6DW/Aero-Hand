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

  ## CAN66: release curriculum with staged support retreat (2026-04-24)

  ### 改动
  - 保持 `CAN62` reward 主线不变，第一次尝试在 support 侧做结构改动：
    - support release 后不再立刻视为 fully unsupported
    - support 采用两段式 retreat，先轻退、再完全清退
  - 目标是给策略更多“handover”缓冲，而不是继续改 reward 权重。

  ### 结果
  - 训练确实得到更高总 reward，但后验核查发现：
    - released 相关奖励仍可能在 support 尚未完全移走时被计入
    - 因此这轮存在明显的假 post-release 膨胀

  ### 结论
  - `CAN66` 不能作为真实 unsupported 突破依据。
  - 它暴露出的真正问题是：必须把“support release 开始”和“support 完全清退”严格分开统计。

  ## CAN67: honest support-cleared gating (2026-04-24)

  ### 改动
  - 新增 `_support_is_cleared()`，只有 support 完全退出后才允许：
    - `stable_hold_steps` 累计
    - `released_gate` 打开
    - `contact_duration_sec` 进入真实 unsupported 统计
  - release 侧加入更严格的质量 / 运动门控，防止太早放手。

  ### 结果
  - `contact_duration_sec ≈ 11.45s`
  - 但真实 clear 后的 `post_release_*` 子项几乎为零。

  ### 结论
  - `CAN67` 证明之前确实有“假 unsupported 奖励膨胀”。
  - 同时也说明当前策略在真正 clear 后几乎接不住物体，结构瓶颈转向 handover / clear 后控制。

  ## CAN68: fast clear + true unsupported metric (2026-04-24)

  ### 改动
  - 缩短 `release_ramp_sec`，并把 `contact_duration_sec` 明确改成只统计真实 unsupported 接触时长。

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.21s`
  - `support_released ≈ 20.86` steps

  ### 结论
  - 策略已经能进入 clear phase，但一旦真正失去支撑就会很快掉落。
  - 问题不再是“不敢 clear”，而是“clear 后控制根本接不住”。

  ## CAN69: probe handover (2026-04-24)

  ### 改动
  - 将 support handover 改成两阶段：
    - 先进入 probe 卸载
    - probe 稳住后再 full clear
  - 在 `info` 中显式维护：
    - `probe_ready_steps`
    - `support_clear_started`
    - `support_clear_timer`

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.366s`
  - clear 后的 `post_release_joint_palm_hold / cradle_lock / post_release_survival` 全部抬升。

  ### 结论
  - 两阶段 handover 是有效方向。
  - 它第一次把真实 unsupported 指标从 `0.2s` 量级推到 `0.3s+`。

  ## CAN70: deeper probe handover (2026-04-24)

  ### 改动
  - probe 下探更深、probe 停留更长，试图在 full clear 前做更重的载荷转移测试。

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.367s`

  ### 结论
  - 单独加深 probe 的收益很有限，说明下一步不能只靠更长 probe，而需要让 full clear 的触发条件更像真实 cradle。

  ## CAN71: hard cradle-gated clear (2026-04-24)

  ### 改动
  - full clear 不再只看 wrap + 低速，而是要求 probe 期间同时满足较高的：
    - `joint_palm_clamp`
    - `ulnar_wrap`
    - `thumb_under`

  ### 结果
  - `support_released ≈ 2.9` steps
  - 真实 unsupported `contact_duration_sec ≈ 0`

  ### 结论
  - 方向本身不是错的，但门槛开得过严，直接把 clear 卡死了。

  ## CAN72: soft cradle gate (2026-04-24)

  ### 改动
  - 将 CAN71 的硬阈值改成中强度 cradle score：
    - 保留 `joint_palm_clamp` 最低下限
    - 用 `joint_palm_clamp + ulnar_wrap + thumb_under` 的加权分数决定 full clear

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.472s`
  - `support_released ≈ 22.76` steps

  ### 结论
  - soft cradle gate 成功保留了 clear 暴露，同时把 clear 后 retention 明显推高。
  - 这条线成为新的主线结构。

  ## CAN73: soft cradle gate continuation (2026-04-24)

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.561s`
  - `support_released ≈ 23.74` steps

  ### 结论
  - 同结构续训仍单调上升，说明 `CAN72` 不是一次性偶然跳升，而是可继续学习的主线。

  ## CAN74: earlier release window fix (2026-04-24)

  ### 发现
  - 当时的配置：
    - `release_after_sec = 10.2`
    - `probe_hold_sec = 0.38`
    - `release_ramp_sec = 0.45`
  - 在 `episode_length = 800`、`ctrl_dt = 0.05` 下，最大真实 unsupported 窗口只有：
    - `40.0 - 10.2 - 0.38 - 0.45 = 28.97s`
  - 这意味着用户要求的 `30s+` 在旧时间预算下其实不可达。

  ### 改动
  - 将 release / force-release 整体前移，使真实 unsupported 窗口先变成可达：
    - `release_after_sec = 8.8`
    - `force_release_after_sec = 9.4`

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.642s`
  - `support_released ≈ 24.65` steps

  ### 结论
  - 30s 目标在配置层面终于变成可达。
  - 提前 release 后，这条 soft cradle 主线依然继续上升，没有被破坏。

  ## CAN75: window-fix continuation (2026-04-24)

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.675s`
  - `support_released ≈ 24.99` steps

  ### 结论
  - 时间窗修复后的主线继续增长，但增幅开始放缓。

  ## CAN76: more clear exposure (2026-04-24)

  ### 改动
  - 再次前移 release，并缩短 probe 停留，增加每个 episode 的真实 unsupported 暴露时间：
    - `release_after_sec = 7.8`
    - `force_release_after_sec = 8.4`
    - `probe_hold_sec = 0.30`
  - 对应最大真实 unsupported 窗口：
    - `40.0 - 7.8 - 0.30 - 0.45 = 31.45s`

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.825s`
  - `support_released ≈ 27.81` steps
  - `post_release_joint_palm_hold ≈ 764.24`
  - `cradle_lock ≈ 415.79`
  - `post_release_survival ≈ 5332.04`

  ### 结论
  - 当前最佳主线已经从早期约 `11s` 的“名义 unsupported / 实际统计混杂”状态，转到真正按 clear 后接触统计的 `0.825s`。
  - 虽然离 `30s+` 还很远，但方向已经明确：
    - 更早 clear 暴露
    - soft cradle-gated handover
    - clear 后 joint-palm cradle retention
  - 这条线在 `CAN76` 的四个 eval 点仍是单调上升，尚未完全平台。

  ## CAN77: more clear exposure continuation (2026-04-24)

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.863s`
  - `support_released ≈ 28.15` steps
  - `post_release_joint_palm_hold ≈ 845.34`
  - `cradle_lock ≈ 459.94`
  - `post_release_survival ≈ 5722.62`

  ### 结论
  - `CAN77` 继续沿 `CAN76` 主线单调上升，说明更早 clear 暴露仍在产生正向学习信号。
  - 但相对 `CAN76` 的提升已开始放缓，说明这条线正在接近新的局部平台。

  ## CAN78: more clear exposure continuation (2026-04-24)

  ### 结果
  - 真实 unsupported `contact_duration_sec ≈ 0.894s`
  - `support_released ≈ 29.00` steps
  - `post_release_joint_palm_hold ≈ 939.62`
  - `cradle_lock ≈ 510.99`
  - `post_release_survival ≈ 6205.80`

  ### 结论
  - `CAN78` 仍有提升，但增幅比 `CAN76 -> CAN77` 更小。
  - 这说明当前主线还没有完全失效，但如果再续一轮只得到极小增量，就需要重新回到结构改动，而不是继续长跑。

  ## 当前主线判断 (2026-04-24 晚)

  ### 当前最好真实指标
  - 当前已完整验证的最好真实 unsupported 抓持时长：
    - `CAN78 contact_duration_sec ≈ 0.894s`

  ### 当前主线结构
  - 代码状态：`soft cradle gate + earlier release + shorter probe`
  - 核心思想：
    - 先 probe 卸载
    - 用中等强度 cradle score 决定 full clear
    - 给每个 episode 留出 `31s+` 的真实 unsupported 训练窗口

  ### 当前未完成事项
  - 用户目标仍是 `30s+` 真实 unsupported hold。
  - 这条主线虽然在持续上升，但目前距离目标仍差两个数量级以上。
  - 当前最合理策略是：
    - 先再验证一轮同结构续训是否正式进入平台
    - 若增量继续缩小，则切回新的后 clear 控制结构改动

## CAN149-CAN172: 2026-04-25 autonomous narrow-structure probes

### 当前回填说明
- 这一批实验全部属于 can 任务，和 cube 日志分开记录。
- 真实成败指标统一只看 `eval/episode_diagnostic/contact_duration_sec`。
- 基线 checkpoint：
  - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- 这一批实验的一个关键新发现是：6 维 action 顺序实际是
  - `[thumb_rot, thumb_flex, index, middle, ring, pinky]`
  之前有过误把第 2/6 维口头当成别的手指的情况，因此后续分析以这个真实映射为准。

### CAN149: long continuation from CAN147
- 代码改动：无，直接在 `CAN147` 主线上长训。
- 命令要点：`num_timesteps=2097152, num_evals=4`
- 修改原因：
  - 验证 `CAN147` 是否已经足够强，可以直接长续训。
- 预期效果：
  - 若主线已具可训练性，长训应继续把 unsupported hold 抬高。
- 实际结果：
  - `first=2.114452124, last=1.542773128, max=2.114452124, best_step=0`
  - 训练后明显退化。
- 分析：
  - 说明“从好 checkpoint 直接续很长”不是当前瓶颈的解法。
  - 之后所有长训都需要先有短 probe 证据支撑。

### CAN150-CAN154: 食指/拇指形态方向受控 probe
- 共同动机：
  - 用户明确指出当前问题仍是“食指参与少、拇指参与多”，希望更像“食指把圆柱压在手掌上，大拇指在下方托举”。
  - 因此先在 action/control nominal pose 附近做小步试探，而不直接重写 reward。

#### CAN150
- 代码改动：
  - `action_scale: [0.07, 0.255, 0.22, 0.22, 0.24, 0.082] -> [0.07, 0.245, 0.235, 0.22, 0.24, 0.082]`
- 原因：
  - 明显降 `thumb_flex`，明显增 `index`。
- 预期：
  - 更接近食指压掌、拇指托举。
- 实际：
  - `first=2.090624094, last=2.039452314, max=2.090624094, best_step=0`
- 分析：
  - 负向，拇指 flex 不能一下砍太多。

#### CAN151
- 代码改动：
  - `action_scale -> [0.07, 0.252, 0.225, 0.22, 0.24, 0.082]`
- 原因：
  - 轻降 `thumb_flex`，轻增 `index`。
- 预期：
  - 保住托举骨架的同时多给食指 room。
- 实际：
  - `first=2.106249094, last=1.896288395, max=2.106249094, best_step=0`
- 分析：
  - 比 CAN150 好一点，但训练后仍退化。

#### CAN152
- 代码改动：
  - `action_scale -> [0.07, 0.255, 0.225, 0.22, 0.24, 0.082]`
- 原因：
  - 只做 index-only 微增。
- 预期：
  - 检查“食指 room”本身是否有效。
- 实际：
  - `first=2.100194454, last=2.000780582, max=2.100194454, best_step=0`
- 分析：
  - 比 CAN151 更稳，但仍未超过 CAN147。

#### CAN153
- 代码改动：
  - 更明显改 nominal pose / default ctrl，让食指更压、拇指更松。
- 原因：
  - 直接从初始抓形逼近用户想要的手型。
- 预期：
  - 让策略从更对的 morphology 出发。
- 实际：
  - `first=1.923827410, last=1.833983660, max=1.923827410, best_step=0`
- 分析：
  - 明显负向，姿态层改太大破坏了主线。

#### CAN154
- 代码改动：
  - 恢复 pre-grasp pose，只轻推 `default_ctrl`
- 原因：
  - 测试“更轻的形态偏置”是否可行。
- 预期：
  - 不破坏入手几何的情况下略推食指压掌。
- 实际：
  - `first=2.021288157, last=1.899218082, max=2.021288157, best_step=0`
- 分析：
  - 仍弱于 CAN147，说明这批手型偏置不是主突破口。

### CAN155-CAN158: handover timing / probe-depth probes

#### CAN155
- 代码改动：
  - `probe_hold_sec: 0.30 -> 0.35`
- 原因：
  - 给浅支撑 probe handover 多 50ms。
- 预期：
  - 圆柱能更充分坐进掌心。
- 实际：
  - `first=2.111327171, last=1.859569550, max=2.111327171, best_step=0`
- 分析：
  - 负向，probe 停得更久反而伤 trainability。

#### CAN156
- 代码改动：
  - `release_ready_sec: 0.20 -> 0.18`
- 原因：
  - 让已经成形的 grasp 更早进入 release。
- 预期：
  - 减少“晚半拍”的 handover。
- 实际：
  - `first=2.104881763, last=1.619335532, max=2.104881763, best_step=0`
- 分析：
  - 明显负向，过早放开会伤后续 hold。

#### CAN157
- 代码改动：
  - `probe_drop_m: 0.018 -> 0.016`
- 原因：
  - 试更浅的 probe 卸载。
- 预期：
  - 减少 release 后前滚。
- 实际：
  - `first=2.122264862, last=1.963671088, max=2.122264862, best_step=0`
- 分析：
  - `step 0` 非常接近 CAN147，是一个有信号的方向，但训练后掉回去。

#### CAN158
- 代码改动：
  - `probe_drop_m: 0.018 -> 0.017`
- 原因：
  - 补中间值。
- 预期：
  - 在 0.016 与 0.018 之间取平衡。
- 实际：
  - `first=2.113280296, last=1.965624213, max=2.113280296, best_step=0`
- 分析：
  - 不如 0.016，高点更偏向更浅那边，但仍不具 trainability。

### CAN159-CAN171: clear-drop geometry line

#### CAN159
- 代码改动：
  - `clear_drop_m: 0.060 -> 0.050`
- 原因：
  - 怀疑 clear 阶段掉太深，导致 handover 被打断。
- 预期：
  - 更浅的 clear 能减少 post-release 前滚。
- 实际：
  - `first=2.114647627, last=2.033202410, max=2.114647627, best_step=0`
- 分析：
  - 没破 CAN147 峰值，但训练后保留性是这条线里第一次明显变好。

#### CAN161
- 代码改动：
  - `clear_drop_m: 0.050 -> 0.055`
- 原因：
  - 在 0.050 与 0.060 之间补点。
- 预期：
  - 同时兼顾高初值和训练稳定性。
- 实际：
  - `first=2.121288300, last=2.054296017, max=2.121288300, best_step=0`
- 分析：
  - 这是当前最好新支线；虽然还没破 CAN147 峰值，但 trainability 最好。

#### CAN162
- 代码改动：
  - 无额外代码改动，从 CAN161 最优点长训。
- 原因：
  - 验证 `clear_drop_m=0.055` 是否具备可持续性。
- 实际：
  - `first=2.032030582, last=1.890038252, max=2.032030582, best_step=0`
- 分析：
  - 长续训仍退化，说明它更像短 probe 提升，而不是稳定 continuation 主线。

#### CAN163
- 代码改动：
  - `clear_drop_m: 0.055 -> 0.057`
- 实际：
  - `first=2.113866329, last=2.004686832, max=2.113866329, best_step=0`
- 分析：
  - 不如 0.055。

#### CAN164
- 代码改动：
  - `clear_drop_m: 0.055 -> 0.054`
- 实际：
  - `first=2.102928638, last=2.008983612, max=2.102928638, best_step=0`
- 分析：
  - 也不如 0.055，说明 clear_drop 邻域当前最好点就是 0.055 左右。

#### CAN165
- 代码改动：
  - 保持 `clear_drop_m=0.055`，从 CAN161 best 继续短训。
- 实际：
  - `first=2.035350800, last=2.088866472, max=2.088866472, best_step=1064960`
- 分析：
  - continuation 不能重现 CAN161 的高初值，说明那次更像结构改动带来的新起点，而不是 checkpoint 内部有稳定“可续训”好策略。

#### CAN166
- 代码改动：
  - `clear_drop_m=0.055 + probe_drop_m=0.016`
- 原因：
  - 把两个各自有信号的几何小改动叠加。
- 实际：
  - `first=2.106444359, last=1.876171231, max=2.106444359, best_step=0`
- 分析：
  - 不叠加，说明 `probe_drop_m=0.016` 并不是 `clear_drop_m=0.055` 的互补项。

#### CAN167
- 代码改动：
  - `clear_drop_m=0.055` 保持不变
  - `learning_rate: 3e-4 -> 1e-4`
- 原因：
  - 直接针对 “短 probe 好、续训学坏” 的 trainability 问题。
- 实际：
  - `first=2.114647627, last=1.940038323, max=2.114647627, best_step=0`
- 分析：
  - 降学习率没有救 trainability，这条优化器线可以先停。

#### CAN168
- 代码改动：
  - `clear_drop_m=0.055` 保持
  - `release_ramp_sec: 0.45 -> 0.40`
- 原因：
  - 在当前最好 clear 深度上，让 clear 动作再稍微利落一点。
- 预期：
  - 如果问题是 clear 动作在 0.055 深度下还稍慢，这会进一步改善 handover。
- 实际：
  - `first=2.157225609, last=1.970311642, max=2.157225609, best_step=0`
- 分析：
  - 这是目前新的全局最好 `step 0`，已经超过 CAN147 (`2.127342939`)。
  - 但训练后还是掉回去，说明结构方向是对的，但 trainability 仍没解。

#### CAN169
- 代码改动：
  - 从 CAN168 的 best checkpoint 继续
  - `learning_rate=1e-4`
- 实际：
  - `first=1.951171041, last=1.954491377, max=1.954491377, best_step=1064960`
- 分析：
  - 从 CAN168 的训练后 checkpoint 已经拿不到那个 `2.1572` 的好起点，说明 CAN168 的高点不是可直接续训保持的内部态。

#### CAN170
- 代码改动：
  - `clear_drop_m: 0.055 -> 0.056`
- 实际：
  - `first=2.113866329, last=2.007616520, max=2.113866329, best_step=0`
- 分析：
  - 仍不如 0.055，进一步坐实 `0.055` 是当前 clear_drop 最优点。

#### CAN171
- 代码改动：
  - `release_after_sec: 7.8 -> 7.6`
  - `force_release_after_sec: 8.4 -> 8.2`
  - 其他保持 `clear_drop_m=0.055`
- 原因：
  - 检查“整体 handover 更早一点”是否有用。
- 实际：
  - `first=2.120897770, last=1.986522675, max=2.120897770, best_step=0`
- 分析：
  - 负向，且 index/thumb/joint_palm 诊断一起掉，不值得继续。

### CAN172: min_release_force loosen under clear_drop=0.055
- 代码改动：
  - `min_release_force: 0.125 -> 0.120`
- 原因：
  - 在最好新支线下，轻微放松 release 资格，测试是否存在“手型已经对了但 release 还卡半拍”的情况。
- 预期效果：
  - 可能更早触发有效 handover，提升 unsupported 时长。
- 实际结果：
  - `first=2.099999189, last=1.960936785, max=2.099999189, best_step=0`
  - best点诊断：
    - `support=57.32421875`
    - `index=169.26171875`
    - `thumb=233.015625`
    - `joint_palm=215.58203125`
    - `slip=0.0`
- 实际效果：
  - 没有形成新的高点，训练后还明显退化。
- 分析：
  - 当前信号弱于 `CAN161/CAN168`，不是新主线。
  - 轻微放松 `min_release_force` 没有解决“release 晚半拍”问题，反而让训练后保留性更差。

### 当前阶段结论 (2026-04-25 夜)
- 当前最好**训练后保留性**支线是：
  - `clear_drop_m=0.055`
  - 对应 `CAN161: first=2.121288300, last=2.054296017`
- 当前最好**单点评估峰值**是：
  - `CAN168: first=max=2.157225609`
  - 结构为 `clear_drop_m=0.055 + release_ramp_sec=0.40`
- 但截至当前，仍然没有任何新线在“max 或训练后 last”上同时稳定超过 `CAN147` 主线并具可持续 continuation。
- 因此当前 can 线的真实瓶颈已经从“找不到更高 step-0 几何”转为：
  - 如何保住新结构带来的高起点，不让 PPO 在后续更新中把它学坏。

### CAN173: loosen `min_release_active_fingers` under best clear-drop branch
- 代码改动：
  - `min_release_active_fingers: 4 -> 3`
  - 其他保持当前最好新支线：
    - `clear_drop_m=0.055`
    - `probe_drop_m=0.018`
    - `release_after_sec=7.8`
    - `release_ramp_sec=0.45`
    - `min_release_force=0.125`
- 训练命令：
  - `CAN173_probe_min_release_active_3_clear055_512`
  - base checkpoint:
    - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- smoke test：
  - CPU smoke passed
- 修改原因：
  - 当前 can 线里 palm+thumb+triad 形状信号已经很强，怀疑“四指全齐”这个门偶尔会把已经足够稳的包裹 handover 卡掉。
- 预期效果：
  - release 资格略放松，handover 更早更平滑，提升 unsupported hold。
- 实际效果（当前已写出 step 0）：
  - `first=2.103319645, last=2.103319645, max=2.103319645, best_step=0`
  - best点诊断：
    - `support=57.40625`
    - `index=169.69921875`
    - `thumb=233.27734375`
    - `joint_palm=215.83984375`
    - `slip=0.00390625`
- 分析：
  - 首个 eval 已弱于 `CAN161 (2.1213)`、`CAN168 (2.1572)` 和 `CAN159 (2.1146)`。
  - 放松离散 release-finger gate 没有提供新的正信号，当前看不像值得继续的方向。
  - 完整 short probe 跑完后，结果进一步确认该方向为负：
    - `last=1.964452386`
    - 两个 eval 点从 `2.1033 -> 1.9645` 明显下滑。
  - 下一步应回到 `clear_drop_m=0.055` 主线，继续只调 release/clear 动力学，而不是继续放松离散 release gate。

### CAN174: keep clear055 and slightly soften release ramp
- 代码改动：
  - `release_ramp_sec: 0.45 -> 0.425`
  - 其他保持当前最优邻域：
    - `clear_drop_m=0.055`
    - `probe_drop_m=0.018`
    - `release_after_sec=7.8`
    - `force_release_after_sec=8.4`
    - `min_release_active_fingers=4`
    - `min_release_force=0.125`
- 训练命令：
  - `CAN174_probe_clear055_ramp0425_512`
  - base checkpoint:
    - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- smoke test：
  - CPU smoke passed
- 修改原因：
  - `CAN168` 说明更利落的 clear ramp 有明显正信号，但 `0.40` 在训练后掉得偏多；因此回退半步到 `0.425`，看能否保留高峰值同时改善训练后保留性。
- 预期效果：
  - 接近 `CAN168` 的高 step-0 unsupported hold，同时比 `0.40` 略稳一些。
- 实际效果：
  - `first=2.155272484, last=2.011327267, max=2.155272484, best_step=0`
  - best点诊断：
    - `index_wrap_contact=169.40625`
    - `middle_wrap_contact=228.2109375`
    - `ring_wrap_contact=218.2421875`
    - `thumb_wrap_contact=233.16015625`
    - `joint_palm_contact=215.875`
    - `index_palm_press=50.14325714111328`
    - `post_release_joint_palm_hold=5472.4189453125`
    - `post_release_force_support=830.09521484375`
    - `post_release_grasp=895.6739501953125`
    - `post_release_survival=31369.12109375`
    - `post_release_slip=-567.69873046875`
- 分析：
  - `CAN174` 基本保住了 `CAN168` 的高点：
    - `2.1553` vs `2.1572`
  - 同时训练后 `last=2.0113` 也略高于 `CAN168` 的 `1.9703`。
  - 这说明最有希望的窄结构线仍然是：
    - `clear_drop_m=0.055`
    - `release_ramp_sec` 在 `0.40 ~ 0.425` 附近
  - 但训练后仍明显丢失：
    - `index_palm_press`
    - `joint_palm_clamp`
    - `post_release_joint_palm_hold`
    - `post_release_survival`
  - 也就是说，当前瓶颈更像“clear 启动后，食指压掌骨架没有被稳定保住”，而不是“需要再更早 release”。
- 下一轮建议：
  - 保持 `clear_drop_m=0.055` 和 `release_ramp_sec≈0.425`
  - 优先尝试让 support 从 probe 进入 clear 前，对 `index_palm_press_release` 和 `probe_cradle_score` 更严格一点，检查能否保住训练后 retention。

### CAN175: tighten clear-entry gate to wait for stronger index-palm press
- 代码改动：
  - 在 `grasp_can_v2_force.py` 中把 clear 前门槛显式参数化：
    - `clear_probe_joint_palm_clamp_min = 0.24`
    - `clear_probe_index_palm_press_min = 0.20`
    - `clear_probe_cradle_score_min = 0.44`
  - `probe_clear_ready` 从固定阈值改为读取上述 config。
  - 其他保持 `CAN174` 主线：
    - `clear_drop_m=0.055`
    - `release_ramp_sec=0.425`
    - `probe_drop_m=0.018`
    - `release_after_sec=7.8`
    - `min_release_active_fingers=4`
    - `min_release_force=0.125`
- 训练命令：
  - `CAN175_probe_clear055_ramp0425_cleargate_indexpress020_512`
  - base checkpoint:
    - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- smoke test：
  - CPU smoke passed
  - 输出确认：
    - `clear_drop_m=0.055`
    - `release_ramp_sec=0.425`
    - `clear_probe_joint_palm_clamp_min=0.24`
    - `clear_probe_index_palm_press_min=0.20`
    - `clear_probe_cradle_score_min=0.44`
- 修改原因：
  - `CAN168/CAN174` 的高 step-0 支线训练后主要丢失的是：
    - `index_palm_press`
    - `joint_palm_clamp`
    - `post_release_joint_palm_hold`
    - `post_release_survival`
  - 因此尝试在 clear 真正开始前，要求“食指压掌 + 掌部夹持”更充分，以保住用户希望的“食指压掌、拇指托举”形态。
- 预期效果：
  - step-0 接近 `CAN174`
  - 训练后 retention 高于 `CAN174 last=2.011327267`
- 实际效果：
  - `first=2.150584936, last=1.732616663, max=2.150584936, best_step=0`
  - best点诊断：
    - `index_wrap_contact=169.59375`
    - `middle_wrap_contact=228.36328125`
    - `ring_wrap_contact=218.33203125`
    - `thumb_wrap_contact=233.375`
    - `joint_palm_contact=215.79296875`
    - `index_palm_press=52.47510528564453`
    - `post_release_joint_palm_hold=5495.90576171875`
    - `post_release_force_support=830.8779296875`
    - `post_release_grasp=896.6170043945312`
    - `post_release_survival=31476.3828125`
    - `post_release_slip=-563.315673828125`
  - final点诊断：
    - `index_wrap_contact=168.0078125`
    - `middle_wrap_contact=111.90234375`
    - `ring_wrap_contact=217.37890625`
    - `thumb_wrap_contact=233.54296875`
    - `joint_palm_contact=204.65234375`
    - `index_palm_press=30.863248825073242`
    - `post_release_joint_palm_hold=4961.802734375`
    - `post_release_force_support=812.7642211914062`
    - `post_release_grasp=848.199951171875`
    - `post_release_survival=29286.232421875`
    - `post_release_slip=-546.5225830078125`
- 分析：
  - 这轮是明确负样本。
  - step-0 只比 `CAN174` 略低：
    - `2.1506` vs `2.1553`
  - 但训练后大幅退化到 `1.7326`，远弱于：
    - `CAN174 last=2.0113`
    - `CAN161 last=2.0543`
  - 最显著的坏信号是 `middle_wrap_contact` 在 final 点崩到 `111.9`，而不是仅仅食指没保住。
  - 说明把 clear-entry gate 收得更紧，会破坏原本的中指/掌部联动，不是当前 retention 问题的正确解。
- 下一轮建议：
  - 回退 `CAN175` 的 clear-entry gate 收紧改动。
  - 保留 `clear_drop_m=0.055 + release_ramp_sec=0.425` 主线。
  - 下一步不要继续抬 clear-entry 的离散门槛，而应尝试“更柔和地延后 clear 动作”：
    - 优先探索更短步长的 `release_ramp_sec` 邻域或 probe/clear 过渡形状，
    - 不再继续走更苛刻 gate 这条线。

### CAN176: half-step back toward slower clear ramp
- 代码改动：
  - 回退 `CAN175` 的 clear-entry gate 收紧改动，恢复：
    - `joint_palm_clamp_release > 0.24`
    - `index_palm_press_release > 0.18`
    - `probe_cradle_score > 0.42`
  - 然后仅做一处窄改动：
    - `release_ramp_sec: 0.425 -> 0.4375`
  - 其他保持 `CAN174` 主线：
    - `clear_drop_m=0.055`
    - `probe_drop_m=0.018`
    - `release_after_sec=7.8`
    - `force_release_after_sec=8.4`
    - `min_release_active_fingers=4`
    - `min_release_force=0.125`
- 训练命令：
  - `CAN176_probe_clear055_ramp04375_512`
  - base checkpoint:
    - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- smoke test：
  - CPU smoke passed
  - 输出确认：
    - `clear_drop_m=0.055`
    - `release_ramp_sec=0.4375`
    - `probe_drop_m=0.018`
- 修改原因：
  - `CAN174` 已说明 `0.425` 比 `0.40` 更稳一些。
  - `CAN175` 说明更苛刻 clear-entry gate 会把中指联动打坏。
  - 因此回到最小扰动策略，在 ramp 邻域对 `0.425` 做半步回退，验证是否存在更好的 retention 点。
- 预期效果：
  - 虽然 step-0 峰值可能略低于 `CAN174`，但希望 training-after-probe retention 更高。
- 实际效果：
  - `first=2.103514671, last=1.884178996, max=2.103514671, best_step=0`
  - best点诊断：
    - `index_wrap_contact=169.515625`
    - `middle_wrap_contact=228.43359375`
    - `ring_wrap_contact=218.3515625`
    - `thumb_wrap_contact=233.328125`
    - `joint_palm_contact=215.86328125`
    - `index_palm_press=37.6035270690918`
    - `post_release_joint_palm_hold=5283.4853515625`
    - `post_release_force_support=807.473388671875`
    - `post_release_grasp=874.9246826171875`
    - `post_release_survival=30430.3984375`
    - `post_release_slip=-564.5389404296875`
  - final点诊断：
    - `index_wrap_contact=168.36328125`
    - `middle_wrap_contact=222.0546875`
    - `ring_wrap_contact=214.90625`
    - `thumb_wrap_contact=228.64453125`
    - `joint_palm_contact=210.77734375`
    - `index_palm_press=18.770301818847656`
    - `post_release_joint_palm_hold=4303.7919921875`
    - `post_release_force_support=636.380615234375`
    - `post_release_grasp=778.5713500976562`
    - `post_release_survival=26027.552734375`
    - `post_release_slip=-547.1220703125`
- 分析：
  - 这轮也是负样本，但性质和 `CAN175` 不同。
  - 相比 `CAN174`：
    - `first: 2.1035 < 2.1553`
    - `last: 1.8842 < 2.0113`
  - 相比 `CAN175`：
    - `last: 1.8842 > 1.7326`
  - 说明：
    - `0.4375` 比“更苛刻 gate”要温和得多，
    - 但仍不如 `0.425`，所以当前 ramp 邻域最好点仍然更靠近 `0.425` 一侧，而不是回向 `0.45`。
  - final点里 `index_palm_press`、`post_release_joint_palm_hold`、`post_release_force_support`、`post_release_survival` 全部显著低于 `CAN174`，说明这次更慢半步的 clear ramp 没能保住 release 后骨架。
- 下一轮建议：
  - 把 `release_ramp_sec` 回到 `0.425`。
  - 如果继续扫 ramp 邻域，更有价值的是朝 `0.40` 方向做半步：
    - 优先尝试 `0.4125`
  - 也可以考虑保持 `0.425` 不动，转去 probe/clear 过渡形状的连续变量，而不是继续往 `0.45` 回。

### CAN177: half-step from 0.425 toward the faster-clear side
- 代码改动：
  - 仅做一处窄改动：
    - `release_ramp_sec: 0.425 -> 0.4125`
  - 其他保持当前最佳主线：
    - `clear_drop_m=0.055`
    - `probe_drop_m=0.018`
    - `release_after_sec=7.8`
    - `force_release_after_sec=8.4`
    - `min_release_active_fingers=4`
    - `min_release_force=0.125`
- 训练命令：
  - `CAN177_probe_clear055_ramp04125_512`
  - base checkpoint:
    - `/home/ll/SRTP/Aero-Hand/logs/AeroCanGraspV2Force-20260425-190032-CAN147_probe_thumb_support_0p082_512/checkpoints/000001064960`
- smoke test：
  - CPU smoke passed
  - 输出确认：
    - `clear_drop_m=0.055`
    - `release_ramp_sec=0.4125`
    - `probe_drop_m=0.018`
- 修改原因：
  - `CAN176` 说明朝 `0.45` 回半步会同时伤高点和 retention。
  - 因此改为向 `0.40` 方向做半步，验证 `0.425` 与 `0.40` 之间是否存在更优的折中点。
- 预期效果：
  - 保住甚至略超 `CAN174` 的高 step-0 unsupported hold，
  - 同时使 final retention 不低于 `CAN174 last=2.011327267`。
- 实际效果：
  - `first=2.165428877, last=2.028905392, max=2.165428877, best_step=0`
  - best点诊断：
    - `index_wrap_contact=169.421875`
    - `middle_wrap_contact=228.828125`
    - `ring_wrap_contact=218.5`
    - `thumb_wrap_contact=233.5546875`
    - `joint_palm_contact=216.1484375`
    - `index_palm_press=51.23951721191406`
    - `post_release_joint_palm_hold=5554.341796875`
    - `post_release_force_support=840.5755615234375`
    - `post_release_grasp=901.9327392578125`
    - `post_release_survival=31695.11328125`
    - `post_release_slip=-565.845703125`
  - final点诊断：
    - `index_wrap_contact=168.671875`
    - `middle_wrap_contact=224.51953125`
    - `ring_wrap_contact=215.47265625`
    - `thumb_wrap_contact=229.22265625`
    - `joint_palm_contact=212.6796875`
    - `index_palm_press=22.143959045410156`
    - `post_release_joint_palm_hold=4708.00830078125`
    - `post_release_force_support=683.173583984375`
    - `post_release_grasp=828.6070556640625`
    - `post_release_survival=28001.4296875`
    - `post_release_slip=-561.1676025390625`
- 分析：
  - 这是当前 ramp 邻域里最有价值的新正样本。
  - 相比 `CAN174`：
    - `first: 2.1654 > 2.1553`
    - `last: 2.0289 > 2.0113`
  - 相比 `CAN168`：
    - `first: 2.1654 > 2.1572`
    - `last: 2.0289 > 1.9703`
  - 相比 `CAN161`：
    - `first: 2.1654 > 2.1213`
    - `last: 2.0289 < 2.0543`
  - 也就是说：
    - `0.4125` 已成为当前**单点评估峰值**最优点，
    - 同时训练后 retention 也略优于 `CAN174`，
    - 但还没有超过 `CAN161` 的 best-last。
  - 当前阶段最合理结论是：
    - ramp 最优点已从“可能在 `0.425`”收窄为更可能在 `0.4125~0.425` 之间，
    - 其中 `0.4125` 是当前最值得继续验证 trainability 的新候选主线。
- 下一轮建议：
  - 在 `0.4125` 配置上直接做长训验证，而不是立刻再扫更多邻域点。
  - 如果长训仍明显学坏，再考虑继续细扫 `0.40625` 或回到 `0.425`。
