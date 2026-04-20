"""测试不同耦合系数+thumb_flex组合对指尖距离的影响"""
import mujoco
import numpy as np

xml = 'sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml'

def test(coupling_coeff, thumb_flex_ctrl):
    m = mujoco.MjModel.from_xml_path(xml)
    d = mujoco.MjData(m)
    m.opt.gravity[:] = 0.0
    for i in range(m.neq):
        if m.eq_type[i] == 2:
            n1 = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.eq_obj1id[i])
            if n1 and 'thumb_cmc_flex' in n1:
                if coupling_coeff == 0:
                    m.eq_active0[i] = 0
                else:
                    m.eq_data[i, 1] = coupling_coeff
    mujoco.mj_resetDataKeyframe(m, d, 0)
    d.ctrl[0] = m.actuator_ctrlrange[0, 1]  # thumb_rot max
    d.ctrl[1] = thumb_flex_ctrl               # thumb_flex (mcp)
    d.ctrl[2] = m.actuator_ctrlrange[2, 1]  # index max
    for _ in range(2000):
        mujoco.mj_step(m, d)
    mujoco.mj_forward(m, d)
    th = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'th_tip')].copy()
    fi = d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'if_tip')].copy()
    dist = np.linalg.norm(th - fi) * 1000
    jid_abd = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, 'right_thumb_cmc_abd')
    abd = np.degrees(d.qpos[m.jnt_qposadr[jid_abd]])
    return abd, th, fi, dist

for coeff_label, coeff in [("0.16", 0.16), ("0.10", 0.10), ("0.00(disabled)", 0.0)]:
    print(f"=== coupling={coeff_label} + varying thumb_flex ===")
    print(f"  flex    abd    th_Y     if_Y     dist_mm")
    best = 999
    for fp in range(0, 80, 5):
        fv = fp / 100.0 * 0.79
        abd, th, fi, dist = test(coeff, fv)
        marker = " <-" if dist < best else ""
        if dist < best:
            best = dist
        print(f"  {fv:.3f}  {abd:5.1f}  {th[1]:.4f}  {fi[1]:.4f}  {dist:6.1f}{marker}")
    print(f"  ** min dist = {best:.1f}mm")
    print()
