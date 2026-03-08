import time
import os
import sys
import math
import numpy as np
import mujoco
import mujoco.viewer

# ============================================================================
# test_hand.py  关节参数快速自测脚本（不跑场景、不看接触力）
# ----------------------------------------------------------------------------
# - 只加载裸手模型 right_hand.xml；无需桌面/物体/力学判定。
# - 通过参数区直接改角度/幅值/频率，观察各关节运动。
# - SDK 负责“16 关节角(度) → 7 路 ctrl(米/弧度)”；注释标明映射关系。
# ============================================================================

# 0) 单位与量程（对应 sim_rl/simulation/mujoco/right_hand.xml actuator ctrlrange）
#    ctrl[0~3] : 四指肌腱长度 (m)   范围 0.058520~0.110387
#    ctrl[4]   : 拇指外展关节角 (rad) 范围 -0.1~1.75（直接关节角）
#    ctrl[5]   : 拇指屈曲肌腱 (m)   范围 0.026152~0.038389
#    ctrl[6]   : 拇指 IP 肌腱 (m)   范围 0.081568~0.112138
_DEG2RAD = math.pi / 180.0
_PULLEY_R_M = 9.0 / 1000.0  # MOTOR_PULLEY_RADIUS = 9 mm → 转角(度)×π/180×半径(m) = 缩短量(m)

_FINGER_CTRL_MIN = 0.058520
_FINGER_CTRL_MAX = 0.110387
_TH1_CTRL_MIN = 0.026152
_TH1_CTRL_MAX = 0.038389
_TH2_CTRL_MIN = 0.081568
_TH2_CTRL_MAX = 0.112138
_THUMB_ABD_MIN = -0.1
_THUMB_ABD_MAX = 1.75


# SDK hand_actuations() 输出顺序: [ThumbAbd, ThumbFlex, ThumbTendon, Index, Middle, Ring, Pinky] (单位: 度)
# MuJoCo ctrl 顺序:             [Index, Middle, Ring, Pinky, ThumbAbd, Th1, Th2]
# 这里完成单位转换 + 通道重排。
def sdk_to_mujoco_ctrl(raw):
    ctrl_index = _FINGER_CTRL_MAX - raw[3] * _DEG2RAD * _PULLEY_R_M
    ctrl_middle = _FINGER_CTRL_MAX - raw[4] * _DEG2RAD * _PULLEY_R_M
    ctrl_ring = _FINGER_CTRL_MAX - raw[5] * _DEG2RAD * _PULLEY_R_M
    ctrl_pinky = _FINGER_CTRL_MAX - raw[6] * _DEG2RAD * _PULLEY_R_M
    ctrl_th_abd = raw[0] * _DEG2RAD  # 外展直接是关节角，不走滑轮
    ctrl_th1 = _TH1_CTRL_MAX - raw[1] * _DEG2RAD * _PULLEY_R_M
    ctrl_th2 = _TH2_CTRL_MAX - raw[2] * _DEG2RAD * _PULLEY_R_M
    return [
        np.clip(ctrl_index, _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_middle, _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_ring, _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_pinky, _FINGER_CTRL_MIN, _FINGER_CTRL_MAX),
        np.clip(ctrl_th_abd, _THUMB_ABD_MIN, _THUMB_ABD_MAX),
        np.clip(ctrl_th1, _TH1_CTRL_MIN, _TH1_CTRL_MAX),
        np.clip(ctrl_th2, _TH2_CTRL_MIN, _TH2_CTRL_MAX),
    ]


def angle_to_ctrl(angle_deg_list, converter):
    """
    输入 16 个关节角(度) → SDK → 7 路 ctrl（米/弧度）。
    angle_deg_list 顺序必须符合 AeroHandConstants.joint_names:
      [thumb_abd, thumb_flex, thumb_tendon, thumb_ip,
       index_mcp, index_pip, index_dip,
       middle_mcp, middle_pip, middle_dip,
       ring_mcp, ring_pip, ring_dip,
       pinky_mcp, pinky_pip, pinky_dip]
    """
    raw = converter.hand_actuations(angle_deg_list)
    return sdk_to_mujoco_ctrl(raw)


# 1) 让你能快速改的参数区 ----------------------------------------------------
# 基座位置（mount_z / mount_roll）；仅设定静态值，不做状态机
MOUNT_Z = 0.15      # 手整体高度 (m)，right_hand.xml mount_z range 0~0.25
MOUNT_ROLL = np.pi   # 翻腕角 (rad)，0=默认姿态，pi=手心朝上

# 拇指四个关节角（度），SDK 期望 4 路：Abd, Flex, Tendon, IP
TH_ABD_DEG = 90.0      # 外展（独立关节）
TH_FLEX_DEG = 90.0     # flex 路径（会进 Th1 肌腱）
TH_TENDON_DEG = 90.0   # tendon 路径（会进 Th2 肌腱）
TH_IP_DEG = 20.0       # 拇指 IP 关节角（SDK 第 4 路）

# 四指统一的运动： base + amp*sin(2π f t)
FINGER_BASE_DEG = 20.0   # 基准弯曲角 (度)
FINGER_AMP_DEG = 30.0    # 正弦幅值 (度)
FINGER_FREQ_HZ = 0.5     # 频率 (Hz)

# 可单独开关每根指头的缩放，1=跟随，0=不动，介于其间=半幅
FINGER_SCALE = {
    "index": 1.0,
    "middle": 1.0,
    "ring": 1.0,
    "pinky": 1.0,
}

# 如果想要静态姿态，把 FINGER_AMP_DEG 设 0；想测试极限，把 BASE/AMP 改大，但注意 ctrlrange


# 2) 加载模型与 SDK ----------------------------------------------------------
project_root = os.getcwd()
sys.path.append(os.path.join(project_root, "sdk", "src"))
from aero_open_sdk.joints_to_actuations import JointsToActuationsModel

converter = JointsToActuationsModel()
xml_path = os.path.join("sim_rl", "simulation", "mujoco", "right_hand2.xml")
model = mujoco.MjModel.from_xml_path(xml_path)
data = mujoco.MjData(model)


def build_joint_targets(t):
    """按当前时间 t 构造 16 关节角(度)。"""
    wave = FINGER_BASE_DEG + FINGER_AMP_DEG * math.sin(2 * math.pi * FINGER_FREQ_HZ * t)
    idx = wave * FINGER_SCALE["index"]
    mid = wave * FINGER_SCALE["middle"]
    ring = wave * FINGER_SCALE["ring"]
    pinky = wave * FINGER_SCALE["pinky"]
    return [
        TH_ABD_DEG, TH_FLEX_DEG, TH_TENDON_DEG, TH_IP_DEG,  # 拇指 4 路
        idx, idx, idx,                                     # 食指 MCP/PIP/DIP (PIP=DIP 由 equality 约束)
        mid, mid, mid,                                     # 中指
        ring, ring, ring,                                  # 无名指
        pinky, pinky, pinky,                               # 小指
    ]


def main():
    # 基座初始位姿
    data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_z")] = MOUNT_Z
    data.qpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "mount_roll")] = MOUNT_ROLL
    mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        start = time.time()
        while viewer.is_running():
            t = time.time() - start

            # 关节角 → ctrl
            joint_angles = build_joint_targets(t)
            data.ctrl[:7] = angle_to_ctrl(joint_angles, converter)

            # 固定基座 ctrl
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mount_z_pos")] = MOUNT_Z
            data.ctrl[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "mount_roll_pos")] = MOUNT_ROLL

            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(1/120)  # 120 FPS 预留


if __name__ == "__main__":
    main()
