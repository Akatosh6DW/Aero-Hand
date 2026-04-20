"""诊断脚本：模拟不同拇指弯曲角度下指尖位置，找到最佳方块放置点。"""
import mujoco
import numpy as np
import os

XML_PATH = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"
xml_dir = os.path.dirname(XML_PATH)
os.chdir(xml_dir)
model = mujoco.MjModel.from_xml_path(os.path.basename(XML_PATH))
data = mujoco.MjData(model)

# Joint names
THUMB_JOINTS = ["right_thumb_cmc_abd", "right_thumb_cmc_flex", "right_thumb_mcp"]
INDEX_JOINTS = ["right_index_mcp", "right_index_pip"]
MIDDLE_JOINTS = ["right_middle_mcp", "right_middle_pip"]
RING_JOINTS = ["right_ring_mcp", "right_ring_pip"]

# Get joint ids
def get_jid(name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)

thumb_jids = [get_jid(n) for n in THUMB_JOINTS]
index_jids = [get_jid(n) for n in INDEX_JOINTS]
middle_jids = [get_jid(n) for n in MIDDLE_JOINTS]
ring_jids = [get_jid(n) for n in RING_JOINTS]

# qpos addresses
thumb_qadr = [model.jnt_qposadr[j] for j in thumb_jids]
index_qadr = [model.jnt_qposadr[j] for j in index_jids]
middle_qadr = [model.jnt_qposadr[j] for j in middle_jids]
ring_qadr = [model.jnt_qposadr[j] for j in ring_jids]

# Tip site ids
tip_sites = {
    "th_tip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "th_tip"),
    "if_tip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "if_tip"),
    "mf_tip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip"),
    "rf_tip": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "rf_tip"),
}

# Load home keyframe  
key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
home_qpos = model.key_qpos[key_id * model.nq : (key_id + 1) * model.nq].copy()

print("=" * 70)
print("拇指弯曲扫描：找到拇指指尖的运动轨迹")
print("=" * 70)

# 扫描：拇指渐进弯曲（自然对握姿态）
# 人手抓取目标角：abd≈1.1, flex≈0.3, mcp≈0.5
print(f"\n{'abd':>5} {'flex':>5} {'mcp':>5} | {'th_X':>8} {'th_Y':>8} {'th_Z':>8} | {'if_X':>8} {'if_Y':>8} {'if_Z':>8} | {'mf_X':>8} {'mf_Y':>8} {'mf_Z':>8}")
print("-" * 110)

best_pos = None
best_dist = 999

for abd_frac in [0.0, 0.3, 0.5, 0.7, 0.9, 1.0]:
    for flex_frac in [0.0, 0.3, 0.5, 0.7, 1.0]:
        # 设置拇指角度
        abd = abd_frac * 1.2  # max range ~1.2
        flex = flex_frac * 0.5  # max range ~0.5
        mcp = min(abd_frac, flex_frac) * 0.5  # follows roughly
        
        data.qpos[:] = home_qpos.copy()
        data.qpos[thumb_qadr[0]] = abd
        data.qpos[thumb_qadr[1]] = flex
        data.qpos[thumb_qadr[2]] = mcp
        
        # 同时设置食指中指半闭合（目标 MCP=1.0, PIP=0.8）
        for qadr in index_qadr:
            data.qpos[qadr] = 0.7 if qadr == index_qadr[0] else 0.5
        for qadr in middle_qadr:
            data.qpos[qadr] = 0.7 if qadr == middle_qadr[0] else 0.5
        
        mujoco.mj_forward(model, data)
        
        th = data.site_xpos[tip_sites["th_tip"]]
        if_ = data.site_xpos[tip_sites["if_tip"]]
        mf = data.site_xpos[tip_sites["mf_tip"]]
        
        print(f"{abd:5.2f} {flex:5.2f} {mcp:5.2f} | "
              f"{th[0]:8.4f} {th[1]:8.4f} {th[2]:8.4f} | "
              f"{if_[0]:8.4f} {if_[1]:8.4f} {if_[2]:8.4f} | "
              f"{mf[0]:8.4f} {mf[1]:8.4f} {mf[2]:8.4f}")

print("\n" + "=" * 70)
print("关键姿态下的最佳方块位置（拇指与食/中指中点）")
print("=" * 70)

# 几个关键的抓取姿态
poses = [
    ("半闭合", {"thumb": (0.6, 0.15, 0.25), "finger_mcp": 0.7, "finger_pip": 0.5}),
    ("自然抓握", {"thumb": (0.9, 0.25, 0.4), "finger_mcp": 0.9, "finger_pip": 0.7}),
    ("目标姿态", {"thumb": (1.1, 0.3, 0.5), "finger_mcp": 1.0, "finger_pip": 0.8}),
]

for pose_name, angles in poses:
    data.qpos[:] = home_qpos.copy()
    data.qpos[thumb_qadr[0]] = angles["thumb"][0]
    data.qpos[thumb_qadr[1]] = angles["thumb"][1]
    data.qpos[thumb_qadr[2]] = angles["thumb"][2]
    
    for qadr in index_qadr:
        data.qpos[qadr] = angles["finger_mcp"] if qadr == index_qadr[0] else angles["finger_pip"]
    for qadr in middle_qadr:
        data.qpos[qadr] = angles["finger_mcp"] if qadr == middle_qadr[0] else angles["finger_pip"]
    # Ring/pinky stay relaxed
    
    mujoco.mj_forward(model, data)
    
    th = data.site_xpos[tip_sites["th_tip"]].copy()
    if_ = data.site_xpos[tip_sites["if_tip"]].copy()
    mf = data.site_xpos[tip_sites["mf_tip"]].copy()
    rf = data.site_xpos[tip_sites["rf_tip"]].copy()
    
    # 方块理想位置：拇指与食/中指中点
    midpoint_if = (th + if_) / 2
    midpoint_mf = (th + mf) / 2
    centroid = (th + if_ + mf) / 3
    
    print(f"\n{pose_name}:")
    print(f"  拇指尖: ({th[0]:.4f}, {th[1]:.4f}, {th[2]:.4f})")
    print(f"  食指尖: ({if_[0]:.4f}, {if_[1]:.4f}, {if_[2]:.4f})")
    print(f"  中指尖: ({mf[0]:.4f}, {mf[1]:.4f}, {mf[2]:.4f})")
    print(f"  无名指: ({rf[0]:.4f}, {rf[1]:.4f}, {rf[2]:.4f})")
    print(f"  拇指↔食指中点: ({midpoint_if[0]:.4f}, {midpoint_if[1]:.4f}, {midpoint_if[2]:.4f})")
    print(f"  拇指↔中指中点: ({midpoint_mf[0]:.4f}, {midpoint_mf[1]:.4f}, {midpoint_mf[2]:.4f})")
    print(f"  三指重心:      ({centroid[0]:.4f}, {centroid[1]:.4f}, {centroid[2]:.4f})")
    print(f"  拇指↔食指距离: {np.linalg.norm(th - if_):.4f}")
    print(f"  拇指↔中指距离: {np.linalg.norm(th - mf):.4f}")
