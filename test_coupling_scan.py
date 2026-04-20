"""扫描不同耦合系数对拇指可达性的影响"""
import mujoco
import numpy as np

xml = 'sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml'

def test_coupling(coeff):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0.0
    
    jid_abd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_thumb_cmc_abd')
    jid_flex = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_thumb_cmc_flex')
    th_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'th_tip')
    if_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'if_tip')
    
    for i in range(m.neq):
        if m.eq_type[i] == 2:
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[i])
            if n1 and 'thumb_cmc_flex' in n1:
                if coeff == 0:
                    m.eq_active0[i] = 0
                else:
                    m.eq_data[i, 0] = 0
                    m.eq_data[i, 1] = coeff
    
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[0] = m.actuator_ctrlrange[0, 1]
    d.ctrl[2] = m.actuator_ctrlrange[2, 1]
    for _ in range(2000):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    
    abd = d.qpos[m.jnt_qposadr[jid_abd]]
    flx = d.qpos[m.jnt_qposadr[jid_flex]]
    th = d.site_xpos[th_sid].copy()
    fi = d.site_xpos[if_sid].copy()
    dist = np.linalg.norm(th - fi) * 1000
    return abd, flx, th, fi, dist

print("coeff   abd_deg flex_deg th_Y     if_Y     dist_mm  Y_gap_mm")
print("-" * 65)

best_coeff = 0
best_dist = 999

for c_pct in range(0, 41, 2):
    c = c_pct / 100.0
    abd, flx, th, fi, dist = test_coupling(c)
    y_gap = abs(th[1]-fi[1])*1000
    marker = ""
    if dist < best_dist:
        best_dist = dist
        best_coeff = c
        marker = " <-best"
    print(f"{c:6.2f}  {np.degrees(abd):7.1f} {np.degrees(flx):7.1f}  {th[1]:8.4f} {fi[1]:8.4f} {dist:8.1f} {y_gap:8.1f}{marker}")

print()
print(f"best coupling coeff: {best_coeff:.2f}  min dist: {best_dist:.1f}mm")
