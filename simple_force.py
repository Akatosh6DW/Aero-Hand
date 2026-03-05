# =============================================================================
# simple_force.py  Aero Hand 仿真抓取控制器（竖握版本）
#
# 【MuJoCo 力的机制说明】(详见各处注释)
#   MuJoCo 物理引擎每步调用 mj_step() 时，会：
#   1. 计算接触碰撞  存入 data.contact[]
#   2. 计算接触约束力  存入 data.cfrc_ext[body_id]
#   3. 计算执行器广义力  存入 data.qfrc_actuator[dof_id]
#
#   我们用以下两种力来做抓取控制（test_hand.py 也按此逻辑重写）：
#    data.cfrc_ext[box_id, 3:6]   方块受到的接触合力(N)，最直观的"抓没抓住"
#    data.qfrc_actuator[dof_id]   执行器输出的广义力(方向取决于DOF)，
#      注意：qfrc_actuator 按 DOF 索引，不是按执行器编号！
#
# =============================================================================
import time
import os
import sys
import math
import numpy as np
import mujoco
import mujoco.viewer

# 
# 0. 单位转换常数 (与 test_hand.py 完全一致，便于对照参数传递链路)
# 
# ctrl[0~3] : 四指肌腱长度 (m)，actrange 0.058520~0.110387
# ctrl[4]   : 拇指外展关节角 (rad)，actrange -0.1~1.75
# ctrl[5]   : 拇指屈曲肌腱 (m)，actrange 0.026152~0.038389
# ctrl[6]   : 拇指 IP 肌腱 (m)，actrange 0.081568~0.112138
_DEG2RAD        = math.pi / 180.0
_PULLEY_R_M     = 9.0 / 1000.0   # 绳轮半径 9 mm  转换系数

_FINGER_CTRL_MIN = 0.058520
_FINGER_CTRL_MAX = 0.110387
_TH1_CTRL_MIN    = 0.026152
_TH1_CTRL_MAX    = 0.038389
_TH2_CTRL_MIN    = 0.081568
_TH2_CTRL_MAX    = 0.112138
_THUMB_ABD_MIN   = -0.1
_THUMB_ABD_MAX   = 1.75


def sdk_to_mujoco_ctrl(raw):
    """
    SDK 输出 raw[7](度)  MuJoCo data.ctrl[7](米 / 弧度)

    SDK 输出顺序: [ThumbAbd, ThumbFlex, ThumbTendon, Index, Middle, Ring, Pinky]
    MuJoCo ctrl 顺序: [Index, Middle, Ring, Pinky, ThumbAbd, Th1, Th2]

    肌腱控制逻辑：
      ctrl_tendon(m) = ctrlrange_max - θ(deg)  (π/180)  滑轮半径(m)
       θ 越大（弯曲越多），肌腱越短，ctrl 值越小
    拇指外展逻辑：
      ctrl_abd(rad) = θ(deg)  (π/180)
       直接关节角控制，无绳轮
    """
    ctrl_index  = _FINGER_CTRL_MAX - raw[3] * _DEG2RAD * _PULLEY_R_M
    ctrl_middle = _FINGER_CTRL_MAX - raw[4] * _DEG2RAD * _PULLEY_R_M
    ctrl_ring   = _FINGER_CTRL_MAX - raw[5] * _DEG2RAD * _PULLEY_R_M
    ctrl_pinky  = _FINGER_CTRL_MAX - raw[6] * _DEG2RAD * _PULLEY_R_M
    ctrl_th_abd = raw[0] * _DEG2RAD
    ctrl_th1    = _TH1_CTRL_MAX - raw[1] * _DEG2RAD * _PULLEY_R_M
    ctrl_th2    = _TH2_CTRL_MAX - raw[2] * _DEG2RAD * _PULLEY_R_M
    return [
        np.clip(ctrl_index,  _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_middle, _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_ring,   _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_pinky,  _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_th_abd, _THUMB_ABD_MIN,   _THUMB_ABD_MAX),
        np.clip(ctrl_th1,    _TH1_CTRL_MIN,    _TH1_CTRL_MAX),
        np.clip(ctrl_th2,    _TH2_CTRL_MIN,    _TH2_CTRL_MAX),
    ]


def hand_open_ctrl():
    """
    返回手完全张开时的 ctrl 向量。
    肌腱最长（= ctrlrange_max），弯曲角为 0，弹簧力将手指推开。
    """
    return [
        _FINGER_CTRL_MAX, _FINGER_CTRL_MAX,
        _FINGER_CTRL_MAX, _FINGER_CTRL_MAX,
        0.0, _TH1_CTRL_MAX, _TH2_CTRL_MAX,
    ]


def angle_to_ctrl(angle_deg, converter):
    """
    统一抓握接口：整体弯曲程度 angle_deg（0~90） MuJoCo ctrl

    【重要】直接传度数给 SDK，不要先转成弧度！
    SDK hand_actuations() 期望的输入单位是 **度**。
    原 Gemini 代码写成 math.radians(angle) 传入，值缩小了 57 倍。
    """
    target_joints = [
        angle_deg * 0.1,   # 拇指 Abd：轻微外展，避免过张
        angle_deg * 0.1,   # 拇指 Flex：多弯曲以包裹
        angle_deg * 0.1,   # 拇指 Tendon：主力弯曲
        angle_deg * 0.1,   # 拇指 IP：稍次一级的弯曲
        angle_deg, angle_deg, angle_deg,   # 食指 [MCP, PIP, DIP]（PIP=DIP 由 equality 约束）
        angle_deg, angle_deg, angle_deg,   # 中指
        angle_deg, angle_deg, angle_deg,   # 无名指
        angle_deg, angle_deg, angle_deg,   # 小指
    ]
    raw = converter.hand_actuations(target_joints)  # 输出单位：度
    # 再经过单位转换，得到 MuJoCo 需要的米/弧度
    return sdk_to_mujoco_ctrl(raw)


# 
# 1. 加载模型和场景
# 
project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "sdk", "src"))
from aero_open_sdk.joints_to_actuations import JointsToActuationsModel

converter = JointsToActuationsModel()
xml_path  = os.path.join("sim_rl", "simulation", "mujoco", "scene_right.xml")
model     = mujoco.MjModel.from_xml_path(xml_path)
data      = mujoco.MjData(model)

#
# 2. 预查圆柱体 body id（只查一次，提高效率）
#
_OBJ_BODY_ID = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "target_cylinder")
print(f"[setup] target_cylinder body_id = {_OBJ_BODY_ID}")


def get_object_contact_force():
    """
    统计『手-圆柱体』接触力总和（N），排除平台/地板支撑力。

    说明：
    - 不能直接用 data.cfrc_ext[obj]，它包含平台对圆柱的支撑反力。
    - 这里遍历 data.contact，仅累计涉及目标物体且另一方不是 worldbody(id=0) 的接触。
    """
    total = 0.0
    for i in range(data.ncon):
        c = data.contact[i]
        b1 = model.geom_bodyid[c.geom1]
        b2 = model.geom_bodyid[c.geom2]
        if b1 == _OBJ_BODY_ID or b2 == _OBJ_BODY_ID:
            other = b2 if b1 == _OBJ_BODY_ID else b1
            if other != 0:
                f = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, f)
                total += float(np.linalg.norm(f[:3]))
    return total


def get_object_z():
    """读取圆柱体中心的世界坐标 z（m）。"""
    obj_jnt_id = model.body_jntadr[_OBJ_BODY_ID]
    obj_qadr = int(model.jnt_qposadr[obj_jnt_id])
    return float(data.qpos[obj_qadr + 2])


def main():
    # ================================================================
    # 查找 mount 关节和执行器：
    #   mount_z    : 手整体升降
    #   mount_roll : 抬起后翻腕（手心向上）
    # ================================================================
    mount_z_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_z")
    mount_z_qadr = int(model.jnt_qposadr[mount_z_jnt_id])
    mount_z_dofadr = int(model.jnt_dofadr[mount_z_jnt_id])
    mount_z_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mount_z_pos")

    mount_roll_jnt_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_roll")
    mount_roll_qadr = int(model.jnt_qposadr[mount_roll_jnt_id])
    mount_roll_dofadr = int(model.jnt_dofadr[mount_roll_jnt_id])
    mount_roll_act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mount_roll_pos")

    print(f"[setup] mount_z: qadr={mount_z_qadr} act={mount_z_act_id}")
    print(f"[setup] mount_roll: qadr={mount_roll_qadr} act={mount_roll_act_id}")
    print(f"[setup] obj_id={_OBJ_BODY_ID} nq={model.nq} nu={model.nu}")

    # ================================================================
    # 任务参数（单位：米 / 弧度 / 牛顿）
    #
    # 这组变量决定“手从哪里开始、下降到哪里、抬到哪里、以多快速度夹紧”。
    # 你可以把它们理解成一个离散状态机的调参面板：
    #   P1(初始化) -> P2(下降) -> P3(接触) -> P4(稳握) -> P5(抬升) -> P6(保持)
    # ================================================================
    # INIT_HEIGHT：手基座初始高度（P1 使用）
    # - 太低：起步就可能碰杯/碰桌，造成不稳定。
    # - 太高：下降路径更长，抓取变慢。
    INIT_HEIGHT = 0.24
    # GRASP_HEIGHT：抓取工作高度（P2 下降终点 / P3~P4 保持高度）
    # - 目标是在杯身中上部形成包夹，不是压杯口或按桌面。
    GRASP_HEIGHT = 0.178
    # LIFT_HEIGHT：成功抓住后希望提升到的目标高度（P5 终点）
    # - 与 GRASP_HEIGHT 的差值决定“提起幅度”。
    LIFT_HEIGHT = 0.240

    # ================================================================
    # GRASP_ROLL：抓取时腕部滚转角（绕 mount_roll 轴）
    # ================================================================
    GRASP_ROLL = -np.pi/2+0.01

    # DESCEND_STEP：每个仿真步的下降增量（P2）
    # - 越大下降越快，但更容易错过最佳抓取窗口或产生碰撞冲击。
    DESCEND_STEP = 0.0011
    # LIFT_STEP：每个仿真步的上升增量（P5）
    # - 越大提得越快，但容易在接触不稳时把杯子甩掉。
    LIFT_STEP = 0.0009

    # APPROACH_SPEED：P3“快速闭合”时，手指角度每步增加量（单位：度）
    APPROACH_SPEED = 0.1
    # GRASP_SPEED：P4“慢速夹紧”时，手指角度每步增加量（单位：度）
    # - 用于接触后细调握力，避免一把捏爆接触稳定性。
    GRASP_SPEED = 0.15
    # MAX_GRASP_ANGLE：手指最大闭合角（度）
    # - 限制上限可避免关节接近极限时产生过大约束力。
    MAX_GRASP_ANGLE = 75.0
    # CONTACT_THRESH：判定“接触存在”的最小手-杯接触力（N）
    # - 低于此值视为无有效接触。
    CONTACT_THRESH = 0.05
    # HOLD_FORCE_THRESH：判定“握力足够稳定”的阈值（N）
    # - P4 中达到该力并持续一段时间才允许进入 P5。
    HOLD_FORCE_THRESH = 1.2
    # MIN_CONTACT_ANGLE：最小闭合角门槛（度）
    # - 防止“刚触碰就误判抓住”，要求至少闭合到一定角度。
    MIN_CONTACT_ANGLE = 15.0
    # HOLD_COUNT_STEPS：握力连续满足阈值的步数要求
    # - 相当于时间去抖（debounce），抗瞬时抖动误判。
    HOLD_COUNT_STEPS = 10
    # OBJ_LIFT_DZ_THRESH：判定“真的提起来了”的物体最小抬升高度（m）
    # - 不是只看接触力，而是看杯子 z 是否相对静置高度真的上升。
    OBJ_LIFT_DZ_THRESH = 0.020

    # ── 初始化 mount 状态（第一次 mj_step 前）──
    data.qpos[mount_z_qadr] = INIT_HEIGHT
    data.qpos[mount_roll_qadr] = GRASP_ROLL
    data.ctrl[:7] = hand_open_ctrl()
    data.ctrl[mount_z_act_id] = INIT_HEIGHT
    data.ctrl[mount_roll_act_id] = GRASP_ROLL
    mujoco.mj_forward(model, data)

    # target_z / target_roll：当前阶段希望基座追踪的目标位姿
    target_z = INIT_HEIGHT
    target_roll = GRASP_ROLL
    # current_angle：当前手指整体闭合角（经 angle_to_ctrl 映射到 7 路 ctrl）
    current_angle = 0.0
    # hold_counter：P4 中“连续稳定握持”的计数器
    hold_counter = 0
    # rest_obj_z：P1 结束时记录的杯子静置高度（作为后续“是否抬起”基线）
    rest_obj_z = None
    # phase：状态机编号
    # 1=高位张开稳定, 2=下降接近, 3=快速闭合找接触,
    # 4=慢速夹紧稳握, 5=抬升验证, 6=最终保持
    phase = 1

    # 轨迹日志：定期记录圆柱体状态，复盘“被打飞”瞬间
    LOG_EVERY = 5  # 每隔多少仿真步记录一行
    log_rows = []

    print(">>> 仿真启动：竖握姿态抓杯 -> 确认握紧 -> 提升")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        step_count = 0
        while viewer.is_running():
            loop_start = time.time()
            # 基座关节改为“小步长运动学跟踪”：
            # 这里用每步微小位姿更新来保证稳定，同时抓取真假仍由接触力与物体抬升判定。
            data.qpos[mount_z_qadr] = target_z
            data.qvel[mount_z_dofadr] = 0.0
            data.qpos[mount_roll_qadr] = target_roll
            data.qvel[mount_roll_dofadr] = 0.0
            data.ctrl[mount_z_act_id] = target_z
            data.ctrl[mount_roll_act_id] = target_roll

            mujoco.mj_forward(model, data)
            mujoco.mj_step(model, data)

            sim_time = data.time
            mount_z_now = data.qpos[mount_z_qadr]
            mount_roll_now = data.qpos[mount_roll_qadr]
            contact_force = get_object_contact_force()
            obj_z = get_object_z()
            step_count += 1

            # 采样圆柱体状态：位置/线速度/角速度 + 接触力 + 状态机信息
            if step_count % LOG_EVERY == 0:
                pos = data.xpos[_OBJ_BODY_ID]
                vel = np.zeros(6)
                # 取世界系速度：前 3 个是线速度，后 3 个是角速度
                mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_BODY, _OBJ_BODY_ID, vel, 0)
                log_rows.append([
                    sim_time,
                    pos[0], pos[1], pos[2],
                    vel[0], vel[1], vel[2],
                    vel[3], vel[4], vel[5],
                    contact_force,
                    current_angle,
                    mount_z_now,
                    mount_roll_now,
                    phase,
                ])

            # P1: 高位张开等待稳定
            # 进入条件：初始化后默认 phase=1
            # 退出条件：sim_time >= 0.5s，记录 rest_obj_z 后切到 P2
            if phase == 1:
                data.ctrl[:7] = hand_open_ctrl()
                data.ctrl[mount_z_act_id] = INIT_HEIGHT
                data.ctrl[mount_roll_act_id] = GRASP_ROLL
                print(f"  [P1] 稳定 sim={sim_time:.2f}s z={mount_z_now:.3f} roll={mount_roll_now:.2f} "
                      f"F={contact_force:.3f}N", end="\r")
                if sim_time >= 0.5:
                    rest_obj_z = obj_z
                    phase = 2
                    print(f"\n>>> Phase 2: 手向下接近圆柱体... (obj_rest_z={rest_obj_z:.3f})")

            # P2: 下降到抓取高度
            # 行为：target_z 以 DESCEND_STEP 逐步逼近 GRASP_HEIGHT，手保持张开。
            # 退出条件：到达抓取高度附近（+0.002 容差）后切到 P3。
            elif phase == 2:
                target_z = max(GRASP_HEIGHT, target_z - DESCEND_STEP)
                data.ctrl[:7] = hand_open_ctrl()
                data.ctrl[mount_z_act_id] = target_z
                data.ctrl[mount_roll_act_id] = GRASP_ROLL
                print(f"  [P2] 下降 z={mount_z_now:.3f}->{target_z:.3f} F={contact_force:.3f}N", end="\r")
                if target_z <= GRASP_HEIGHT + 0.01:
                    phase = 3
                    print("\n>>> Phase 3: 快速闭合手指接触圆柱体...")

            # P3: 快速闭合直到接触
            # 行为：以 APPROACH_SPEED 增大闭合角，优先快速建立接触。
            # 退出条件：角度达到 MIN_CONTACT_ANGLE 且接触力超过 CONTACT_THRESH。
            elif phase == 3:
                current_angle = min(current_angle + APPROACH_SPEED, MAX_GRASP_ANGLE)
                data.ctrl[:7] = angle_to_ctrl(current_angle, converter)
                data.ctrl[mount_z_act_id] = GRASP_HEIGHT+0.002  # 稍微抬高一点，避免过早压碎接触
                data.ctrl[mount_roll_act_id] = GRASP_ROLL
                print(f"  [P3] 接近 angle={current_angle:.1f}° F={contact_force:.3f}N", end="\r")
                if current_angle >= MIN_CONTACT_ANGLE and contact_force > CONTACT_THRESH:
                    hold_counter = 0
                    phase = 4
                    print(f"\n>>> Phase 4: 已接触，慢速夹紧到 {HOLD_FORCE_THRESH:.1f}N...")

            # P4: 慢速力控夹紧
            # 行为：
            # - 若握力不足：继续以 GRASP_SPEED 缓慢增加闭合角；
            # - 若握力达标：累计 hold_counter（连续稳定帧数）。
            # 退出条件：hold_counter >= HOLD_COUNT_STEPS，进入 P5 开始抬升。
            elif phase == 4:
                if contact_force < HOLD_FORCE_THRESH:
                    hold_counter = 0
                    current_angle = min(current_angle + GRASP_SPEED, MAX_GRASP_ANGLE)
                    print(f"  [P4] 夹紧 angle={current_angle:.1f}° F={contact_force:.3f}N", end="\r")
                else:
                    hold_counter += 1
                    print(f"  [P4] 稳定抓持 F={contact_force:.3f}N ({hold_counter}/{HOLD_COUNT_STEPS})", end="\r")
                    if hold_counter >= HOLD_COUNT_STEPS:
                        phase = 5
                        print("\n>>> Phase 5: 抓稳，开始提升...")
                data.ctrl[:7] = angle_to_ctrl(current_angle, converter)
                data.ctrl[mount_z_act_id] = GRASP_HEIGHT
                data.ctrl[mount_roll_act_id] = GRASP_ROLL

            # P5: 提升圆柱体
            # 行为：target_z 以 LIFT_STEP 上升到 LIFT_HEIGHT，同时保持当前握姿。
            # 失败回退：
            # - 接触力掉到 CONTACT_THRESH 以下，视为脱手，回 P3；
            # - 到了 LIFT_HEIGHT 但物体 z 未超过 rest_obj_z + OBJ_LIFT_DZ_THRESH，
            #   视为“假抓住（没提起）”，回 P3。
            # 成功条件：到达高位且物体真实抬升，进入 P6。
            elif phase == 5:
                target_z = min(LIFT_HEIGHT, target_z + LIFT_STEP)
                data.ctrl[:7] = angle_to_ctrl(current_angle, converter)
                data.ctrl[mount_z_act_id] = target_z
                data.ctrl[mount_roll_act_id] = GRASP_ROLL
                print(f"  [P5] 提升 z={mount_z_now:.3f}->{target_z:.3f} obj_z={obj_z:.3f} F={contact_force:.3f}N", end="\r")
                if contact_force < CONTACT_THRESH:
                    phase = 3
                    print("\n>>> [WARN] 提升时失去接触，返回 Phase 3 重新抓取...")
                elif target_z >= LIFT_HEIGHT - 0.002:
                    if rest_obj_z is not None and obj_z >= rest_obj_z + OBJ_LIFT_DZ_THRESH:
                        phase = 6
                        print("\n>>> Phase 6: 已抬起，保持竖握姿态。")
                    else:
                        phase = 3
                        print("\n>>> [WARN] 未实际提起圆柱体，返回 Phase 3 重新抓取...")

            # P6: 最终保持（竖握+高位）
            # 行为：维持握角 + 维持抬升高度，作为稳定抓持终态。
            else:
                data.ctrl[:7] = angle_to_ctrl(current_angle, converter)
                data.ctrl[mount_z_act_id] = LIFT_HEIGHT
                data.ctrl[mount_roll_act_id] = GRASP_ROLL
                print(f"  [P7] 保持 z={mount_z_now:.3f} roll={mount_roll_now:.2f} "
                      f"obj_z={obj_z:.3f} F={contact_force:.3f}N", end="\r")

            viewer.sync()

            # 锁帧：仿真以实时速率运行
            elapsed   = time.time() - loop_start
            remaining = model.opt.timestep - elapsed
            if remaining > 0:
                time.sleep(remaining)

    # 关闭窗口后把日志落盘
    if log_rows:
        header = "time,x,y,z,vx,vy,vz,wx,wy,wz,contact,current_angle,mount_z,mount_roll,phase"
        np.savetxt("cylinder_log.csv", np.array(log_rows), delimiter=",", header=header, comments="")
        print(f"\n[log] saved {len(log_rows)} rows to cylinder_log.csv")


if __name__ == "__main__":
    main()
