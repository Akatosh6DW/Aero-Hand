"""诊断脚本：打印 home pose 下各指尖、方块、掌心的世界坐标。"""
import mujoco
import numpy as np
import os

XML_PATH = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"

xml_dir = os.path.dirname(XML_PATH)
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(XML_PATH))
data = mujoco.MjData(model)

# Load home keyframe
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
if key_id >= 0:
    nq = model.nq
    data.qpos[:] = model.key_qpos[key_id * nq : (key_id + 1) * nq]
    nu = model.nu
    if nu > 0:
        data.ctrl[:] = model.key_ctrl[key_id * nu : (key_id + 1) * nu]
    if model.nmocap > 0:
        data.mocap_pos[:] = model.key_mpos[key_id * model.nmocap : (key_id + 1) * model.nmocap]
        data.mocap_quat[:] = model.key_mquat[key_id * model.nmocap : (key_id + 1) * model.nmocap]

mujoco.mj_forward(model, data)

print("=" * 60)
print("Home Pose 世界坐标诊断")
print("=" * 60)

# Cube position from freejoint qpos
cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
cube_pos = data.xpos[cube_body_id]
print(f"\n方块中心: x={cube_pos[0]:.4f}, y={cube_pos[1]:.4f}, z={cube_pos[2]:.4f}")

# Support position
support_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_support")
support_pos = data.xpos[support_body_id]
print(f"支撑台:   x={support_pos[0]:.4f}, y={support_pos[1]:.4f}, z={support_pos[2]:.4f}")

# Grasp site (palm reference)
grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
grasp_pos = data.site_xpos[grasp_site_id]
print(f"掌心参考: x={grasp_pos[0]:.4f}, y={grasp_pos[1]:.4f}, z={grasp_pos[2]:.4f}")

# Fingertip sites
tips = ["th_tip", "if_tip", "mf_tip", "rf_tip", "pf_tip"]
tip_names = ["拇指尖", "食指尖", "中指尖", "无名指尖", "小指尖"]
print(f"\n{'手指':<10} {'世界 X':>8} {'世界 Y':>8} {'世界 Z':>8}  {'到方块距离':>10}")
print("-" * 55)
for name, label in zip(tips, tip_names):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    pos = data.site_xpos[sid]
    dist = np.linalg.norm(pos - cube_pos)
    print(f"{label:<10} {pos[0]:8.4f} {pos[1]:8.4f} {pos[2]:8.4f}  {dist:10.4f}")

# Palm body
palm_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
palm_pos = data.xpos[palm_body_id]
print(f"\n手掌body: x={palm_pos[0]:.4f}, y={palm_pos[1]:.4f}, z={palm_pos[2]:.4f}")

# V2 mount
mount_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "v2_mount")
mount_pos = data.xpos[mount_body_id]
print(f"V2挂载:   x={mount_pos[0]:.4f}, y={mount_pos[1]:.4f}, z={mount_pos[2]:.4f}")

# Also print relative distances (fingertip to cube)
print("\n" + "=" * 60)
print("指尖到方块的相对位置 (finger - cube)")
print("=" * 60)
for name, label in zip(tips, tip_names):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    pos = data.site_xpos[sid]
    rel = pos - cube_pos
    print(f"{label:<10} dx={rel[0]:+.4f} dy={rel[1]:+.4f} dz={rel[2]:+.4f}")

# Compute ideal position for precision pinch
print("\n" + "=" * 60)
print("理想方块位置分析")
print("=" * 60)
thumb_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")]
index_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")]
middle_pos = data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")]

centroid = (thumb_pos + index_pos + middle_pos) / 3.0
print(f"拇指+食指+中指指尖中心: x={centroid[0]:.4f}, y={centroid[1]:.4f}, z={centroid[2]:.4f}")
print(f"当前方块位置:           x={cube_pos[0]:.4f}, y={cube_pos[1]:.4f}, z={cube_pos[2]:.4f}")
print(f"偏差:                   dx={cube_pos[0]-centroid[0]:+.4f}, dy={cube_pos[1]-centroid[1]:+.4f}, dz={cube_pos[2]-centroid[2]:+.4f}")
