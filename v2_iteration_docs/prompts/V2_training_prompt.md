# V2 灵犀手抓握训练 — 新对话提示词

> 复制此文件内容到新聊天的第一条消息中，AI即可继承全部上下文继续迭代。

---

## 一、项目目标

训练 V2 灵犀手（11关节、6通道）完成 **指尖精密捏取**（thumb+index+middle 三指对握）抓取 2.5cm 方块。
**不是侧面掌压夹持**，必须是指尖精密对握。

## 2026-04-23 新增执行要求

1. 先检查当前手部参数改动，再从当前最优方块 checkpoint 继续迭代，训练出**适配新手参**的方块模型。
2. 方块任务完成后，再自主迭代易拉罐任务（原瓶子任务已切换为易拉罐 / 小圆柱代理）。
3. 易拉罐任务必须同时满足：
   - 持握 30s+ 不掉落；
   - 不允许靠过大挤压力造成“易拉罐变形”；
   - 持握时加入晃动/扰动后仍不掉落；
   - 使用当前脚本里保存的初始姿态；
   - 包含 DR 与扰动鲁棒性。
4. 不能只重复训练。每轮必须根据 `metrics.csv`、stdout、视频或离线评估结果修改 reward、阈值、初始化课程、扰动时序等。
5. 易拉罐任务相关日志/记录要与方块任务严格区分。

## 二、严格规则

1. **从 R87 开始计算，本轮迭代次数不超过 7 次**
2. **每一次完成训练都要生成视频，新的视频不能把老的视频覆盖了**
3. **分析日志后如果有问题需要更改奖励结构，禁止单纯重复运行**
4. **严禁连续两轮只改 LR 不改奖励结构**

## 三、技术栈

- **MuJoCo MJX + JAX + Brax PPO**，RTX 4060 Laptop 8GB
- Conda 环境: `aero_rl`
- 工作目录: `/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground`

## 四、关键文件路径

| 文件 | 路径 |
|------|------|
| 环境代码 | `mujoco_playground/_src/manipulation/aero_hand/grasp_cube_v2_force.py` |
| V2 手 XML | `mujoco_playground/_src/manipulation/aero_hand/xmls/right_hand_v2_vertical.xml` |
| 场景 XML | `mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml` |
| 常量 | `mujoco_playground/_src/manipulation/aero_hand/aero_hand_constants.py` |
| PPO 参数 | `mujoco_playground/config/manipulation_params.py` (搜 `AeroCubeGraspV2Force`) |
| 训练脚本 | `learning/train_jax_ppo.py` |
| 迭代日志 | `/home/ll/SRTP/Aero-Hand/V2_iteration_changelog.md` |
| 训练输出 | `logs/AeroCubeGraspV2Force-{timestamp}/` |

## 五、V2 手架构

### 关节 (11个)
```
qpos 顺序:
  index_mcp=0, index_pip=1,
  middle_mcp=2, middle_pip=3,
  ring_mcp=4, ring_pip=5,
  pinky_mcp=6, pinky_pip=7,
  thumb_cmc_abd=8, thumb_cmc_flex=9, thumb_mcp=10
```

### 执行器 (6通道)
```
act[0] hw_thumb_rot  → joint[8] right_thumb_cmc_abd  (kp=5)
act[1] hw_thumb_flex → joint[10] right_thumb_mcp     (kp=3)
act[2] hw_index      → joint[0] right_index_mcp      (kp=3)
act[3] hw_middle     → joint[2] right_middle_mcp     (kp=3)
act[4] hw_ring       → joint[4] right_ring_mcp       (kp=3)
act[5] hw_pinky      → joint[6] right_pinky_mcp      (kp=3)
```

### 等式约束
- 4指 PIP = 0.925 × MCP
- **拇指 CMC_FLEX = 0.16 × CMC_ABD** (R86从0.38改为0.16，核心修复)

### 关节范围
- thumb_cmc_abd: [0, 79°] (1.3788 rad)
- thumb_cmc_flex: [0, 30°] (0.5236 rad)
- thumb_mcp: [0, 45°] (0.7854 rad)
- finger MCP: [0, 80°] (1.3963 rad)
- finger PIP: [0, 74°] (1.2915 rad)

### 坐标系
- Z = 手指延伸方向 (+Z 指尖)
- Y = 开合方向 (-Y 远离手心/抓握方向)
- X = 横跨手指 (+X 拇指侧)

## 六、关键发现：拇指耦合约束

**R85 失败根因分析**:
- 原始耦合 `flex = 0.38 * abd` 导致拇指指尖被拉离手指方向
- 不耦合: 指尖Y达-0.088 (超过食指-0.075)
- 0.38耦合: 指尖Y仅达-0.049 (差25mm)
- **0.16耦合: 指尖Y达-0.077, 指尖距离10.8mm, Y间隙1.7mm** ← 最优

**耦合系数扫描关键数据**:
| coeff | abd° | flex° | dist_mm | Y_gap_mm |
|-------|------|-------|---------|----------|
| 0.00 | 79.0 | 0.0 | 21.4 | 12.5 |
| 0.10 | 79.1 | 7.0 | 13.9 | 4.9 |
| **0.16** | **77.7** | **10.3** | **10.8** | **1.7** |
| 0.18 | 41.9 | 7.0 | 37.0 | 10.6 (分岔!) |
| 0.38 | 77.2 | 29.4 | 29.6 | 24.6 |

**abd不到79°的原因**: kp=5 产生 +0.1194 N·m 驱动力，耦合约束产生 -0.1194 N·m 阻力，在77.7°平衡。提高kp反而有害(更多abd→更多forced flex→指尖反而偏移)。

## 七、训练参数

```bash
cd /home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground && conda run -n aero_rl python learning/train_jax_ppo.py \
  --env_name=AeroCubeGraspV2Force \
  --num_timesteps=40000000 \
  --num_evals=14 \
  --num_envs=4096 \
  --discounting=0.99 \
  --normalize_observations=True \
  --learning_rate=3e-4 2>&1
```

**关键**: `--num_envs=4096` 必须显式指定，默认8192会OOM！

- 训练时间: ~93-97分钟 + ~47秒 JIT
- 网络: policy (512,256,128), value (512,256,128)
- policy_obs_key="state" (42D), value_obs_key="privileged_state" (~98D)
- Episode: 800步 (40s @ 20Hz), 方块2.5cm (half=0.0125), 20g
- 随机释放: 1.5s-4.0s, release_ramp_sec=0.3
- Brax PPO: batch_size=256, entropy_cost=1e-2, unroll_length=40, num_updates_per_batch=4, num_minibatches=32, discounting=0.99

## 八、观测空间 (42D state)

```
motor_targets(6) + tactile_ema(5) + last_action(6) + 
cube_pos_error(3) + cube_vel_scaled(3) + cube_quat(4) + 
fingertip_to_cube_scaled(15) = 42
```

privileged_state (~98D): state + joint_angles + joint_vels + actuator_force + fingertip_rel + cube_pos_error + cube_quat + cube_angvel + cube_linvel

## 九、当前奖励结构 (R86)

### 权重
```python
approach=12, closure=6, contact=10, thumb_engage=20,
hold_position=18, stable_hold=20, progressive_hold=4,
force_contact=15, grip_force=15, force_balance=25,
finger_participation=5, thumb_opposition=30,
primary_finger_force=35, soft_contact=5,
human_pose=2, pip_closure=3,
height=5, survival=1, termination=-80,
drop_risk=-8, action_rate=-0.01, action_accel=-0.005,
torques=-0.00003, force_overload=0, idle_follow=0
```

### 分层门控
```
形状引导 (无门控):     approach, contact, thumb_engage
近距门控 (near_gate):  closure
MCP门控:               pip_closure
released_gate:         grip_force, hold_position*, stable_hold*, force_contact, finger_participation, primary_finger_force
primary_gate (三指有力): hold_position, stable_hold, progressive_hold
无released_gate:       thumb_opposition (R83: 支撑阶段也要学对立)
```

### 核心奖励函数摘要
- **approach**: `exp(-15*mean_tip_dist)` (R86: -20→-15)
- **thumb_engage**: `exp(-12*thumb_tip_dist)` (R81: -35→-12)
- **thumb_opposition**: sigmoid(-5*dots) 替代硬clip, 几何基线50% (R84), finger_weights=[1,1,0.3,0]
- **primary_finger_force**: mean(normalized[idx,mid,thumb]) × min_bonus(0.15+0.85*min/mean)
- **grip_force**: 仅 index+middle+thumb 三指 (R78)
- **force_balance**: 仅 index+middle+thumb 的力标准差/均值
- **hold_position**: exp(-20*dist_from_spawn) × (1-0.5*vel_penalty)
- **stable_hold**: contact_gate × lin_stable × ang_stable
- **drop_risk**: low_risk × (0.4+0.6*down_risk), drop_z=spawn_z-0.04

### 方块位置 (R86)
```
cube_pos = [0.035, -0.056, 0.155]  (R86: Y从-0.065回调到-0.056)
support_pos = [0.035, -0.056, 0.1395]
```

### action_scale
```
[thumb_rot=1.4, thumb_flex=0.8, index=1.4, middle=1.4, ring=1.4, pinky=1.4]
```

## 十、标准化迭代流程 (6步)

### Step 1: 启动训练
- 40M steps, 4096 envs, 14 evals, γ=0.99
- ~93分钟 + 45s JIT

### Step 2: 指标分析
- 从 `metrics.csv` 提取全部奖励分项
- 核心三指: `primary_finger_force`, `thumb_opposition`, `grip_force`
- 持有质量: `hold_position`, `stable_hold`, `progressive_hold`
- 计算效率 = 实际贡献 / (权重 × 平均episode长度)
- 与上轮对比变化率，检查曲线趋势

### Step 3: 诊断 + 奖励修改
| 情况 | 判断依据 | 行动 |
|------|---------|------|
| 曲线仍上升 | 末2次eval增长>3% | 可降LR续训 |
| 有瓶颈组件 | 效率<10%或=0 | **必须修改奖励结构** |
| 退化崩溃 | reward下降>10% | 回退+分析+修改 |

修改原则:
- Rajeswaran 2017: 力封闭 > 运动学约束
- OpenAI 2018/2019: 观测完整+随机化打破过拟合
- 检查奖励漏洞 (不用拇指也能拿分？)

### Step 4: 视频检查
- 确认三指对握 vs 侧面夹持
- 视频保存到各自 logdir，不覆盖

### Step 5: 记录 → 更新 `V2_iteration_changelog.md`

### Step 6: 立即开始下一轮

## 十一、历史成绩

| 轮次 | Peak Reward | EpLen | 备注 |
|------|-------------|-------|------|
| R78 | 2421.4 | 661 | 首次三指门控 |
| R84 | 2926.3 | 739 | **历史最佳** (cube Y=-0.045) |
| R85 | 168.1 | 111 | cube Y=-0.065 崩溃 → 发现耦合根因 |
| R86 | ~280 (13M步中断) | ~71 | coupling=0.16 + Y=-0.056, 训练未完成 |

## 十二、当前状态

- **R86 训练未完成** (仅到13M步/40M步，进程已终止)
- 最新 log 目录: `logs/AeroCubeGraspV2Force-20260418-195224/`
- XML 已改: coupling=0.16, cube Y=-0.056
- 代码已改: approach -20→-15, release_ramp_sec=0.3, drop_risk=-8, drop_z=spawn_z-0.04

**下一步**: 重新启动 R86 训练 (或确认中断原因后决策)

## 十三、Domain Randomization

已启用的随机化:
- 方块摩擦 [0.1, 0.5]，指尖摩擦 [0.5, 1.0]
- 方块质量 ×[0.8, 1.2]
- 初始关节姿态 ±0.03 rad
- 关节摩擦损耗 ×[0.8, 1.2]
- 关节 armature ×[1.0, 1.05]
- 手体质量 ×[0.9, 1.1]
- 执行器 kp ×[0.9, 1.1]
- 关节阻尼 ×[0.9, 1.1]
