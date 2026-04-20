"""V2 灵犀手拇指外展极限测试。
将拇指外展到最大，观察其位置，然后让食指尝试去碰。

运行: python check_thumb_index_pinch.py
"""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

DEFAULT_XML = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"
ZERO_GRAVITY = True
ISOLATE_SCENE_OBJECTS = True
ISOLATION_POS = np.array([1.0, 1.0, 1.0], dtype=np.float64)
PHASE_S = 2.5
HOLD_S = 2.0


def apply_home_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        return
    nq, nu = model.nq, model.nu
    data.qpos[:] = model.key_qpos[key_id * nq : (key_id + 1) * nq]
    if nu > 0:
        data.ctrl[:] = model.key_ctrl[key_id * nu : (key_id + 1) * nu]
    if model.nmocap > 0:
        data.mocap_pos[:] = model.key_mpos[key_id * model.nmocap : (key_id + 1) * model.nmocap]
        data.mocap_quat[:] = model.key_mquat[key_id * model.nmocap : (key_id + 1) * model.nmocap]
    mujoco.mj_forward(model, data)


def isolate_scene_objects(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Move cube/support away so this script measures the hand, not scene contacts."""
    moved = []

    support_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube_support")
    if support_bid >= 0:
        mocap_id = model.body_mocapid[support_bid]
        if mocap_id >= 0:
            data.mocap_pos[mocap_id] = ISOLATION_POS
            data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0])
            moved.append("cube_support")

    cube_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")
    if cube_bid >= 0:
        for jid in range(model.njnt):
            if (model.jnt_bodyid[jid] == cube_bid and
                    model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
                qadr = model.jnt_qposadr[jid]
                data.qpos[qadr:qadr + 7] = np.array([
                    ISOLATION_POS[0], ISOLATION_POS[1], ISOLATION_POS[2],
                    1.0, 0.0, 0.0, 0.0,
                ])
                moved.append("cube")
                break

    if moved:
        mujoco.mj_forward(model, data)
    return moved


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * np.clip(t, 0.0, 1.0)


def get_tip_positions(model, data):
    th_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "th_tip")
    if_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "if_tip")
    mf_sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mf_tip")
    return (data.site_xpos[th_sid].copy(),
            data.site_xpos[if_sid].copy(),
            data.site_xpos[mf_sid].copy())


def ramp_pose(viewer, model, data, ctrl_from, ctrl_to, duration_s):
    t0 = time.time()
    while viewer.is_running():
        t = (time.time() - t0) / max(duration_s, 1e-6)
        if t >= 1.0:
            break
        data.ctrl[:] = lerp(ctrl_from, ctrl_to, t)
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)
    data.ctrl[:] = ctrl_to


def hold_pose(viewer, model, data, ctrl, hold_s):
    t_end = time.time() + hold_s
    while viewer.is_running() and time.time() < t_end:
        data.ctrl[:] = ctrl
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def print_state(model, data, label):
    th_pos, if_pos, mf_pos = get_tip_positions(model, data)
    d_if = np.linalg.norm(th_pos - if_pos) * 1000
    d_mf = np.linalg.norm(th_pos - mf_pos) * 1000

    # 读取关节实际角度
    jid_abd = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_cmc_abd")
    jid_flex = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_cmc_flex")
    jid_mcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_mcp")
    abd_rad = data.qpos[model.jnt_qposadr[jid_abd]]
    flex_rad = data.qpos[model.jnt_qposadr[jid_flex]]
    mcp_rad = data.qpos[model.jnt_qposadr[jid_mcp]]

    print(f"  [{label}]")
    print(f"    拇指关节: cmc_abd={np.degrees(abd_rad):.1f}°({abd_rad:.4f}rad)  "
          f"cmc_flex={np.degrees(flex_rad):.1f}°({flex_rad:.4f}rad)  "
          f"mcp={np.degrees(mcp_rad):.1f}°({mcp_rad:.4f}rad)")
    print(f"    拇指指尖: [{th_pos[0]:.4f}, {th_pos[1]:.4f}, {th_pos[2]:.4f}]")
    print(f"    食指指尖: [{if_pos[0]:.4f}, {if_pos[1]:.4f}, {if_pos[2]:.4f}]  距拇指={d_if:.1f}mm")
    print(f"    中指指尖: [{mf_pos[0]:.4f}, {mf_pos[1]:.4f}, {mf_pos[2]:.4f}]  距拇指={d_mf:.1f}mm")
    if data.ncon:
        print(f"    ⚠ 当前有 {data.ncon} 个接触，会影响关节极限读数")
    return abd_rad, d_if, d_mf


def main() -> None:
    xml_path = os.path.abspath(DEFAULT_XML)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.basename(xml_path)
    os.chdir(xml_dir)

    model = mujoco.MjModel.from_xml_path(xml_name)
    if ZERO_GRAVITY:
        model.opt.gravity[:] = 0.0

    data = mujoco.MjData(model)
    apply_home_keyframe(model, data)
    if ISOLATE_SCENE_OBJECTS:
        moved = isolate_scene_objects(model, data)
        if moved:
            print(f"[info] 已隔离场景物体: {', '.join(moved)} -> {ISOLATION_POS.tolist()}")
            print("[info] 当前测试只检查手本体运动范围，不让方块/支撑台干涉拇指")

    nu = model.nu
    THUMB_ROT_MAX = model.actuator_ctrlrange[0, 1]
    THUMB_FLEX_MAX = model.actuator_ctrlrange[1, 1]
    INDEX_MAX = model.actuator_ctrlrange[2, 1]

    # 关节范围
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_thumb_cmc_abd")
    jnt_lo, jnt_hi = model.jnt_range[jid]
    print(f"[info] thumb_cmc_abd 关节范围: [{np.degrees(jnt_lo):.1f}°, {np.degrees(jnt_hi):.1f}°] = [{jnt_lo:.4f}, {jnt_hi:.4f}] rad")
    print(f"[info] thumb_rot 执行器范围: [0, {THUMB_ROT_MAX:.4f}] rad = [0, {np.degrees(THUMB_ROT_MAX):.1f}°]")
    print(f"[info] thumb_flex 执行器范围: [0, {THUMB_FLEX_MAX:.4f}] rad = [0, {np.degrees(THUMB_FLEX_MAX):.1f}°]")
    print(f"[info] index 执行器范围: [0, {INDEX_MAX:.4f}] rad = [0, {np.degrees(INDEX_MAX):.1f}°]")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        ctrl = np.zeros(nu, dtype=np.float32)

        # ═══ Phase 0: 张开 ═══
        print("\n" + "=" * 70)
        print("Phase 0: 初始张开状态")
        print("=" * 70)
        hold_pose(viewer, model, data, ctrl, 1.0)
        print_state(model, data, "张开")

        # ═══ Phase 1: 只拇指外展到最大 ═══
        print("\n" + "=" * 70)
        print(f"Phase 1: 拇指外展到最大 (ctrl[0]={THUMB_ROT_MAX:.4f})")
        print("=" * 70)
        ctrl_ph1 = np.zeros(nu, dtype=np.float32)
        ctrl_ph1[0] = THUMB_ROT_MAX
        ramp_pose(viewer, model, data, ctrl, ctrl_ph1, PHASE_S)
        hold_pose(viewer, model, data, ctrl_ph1, HOLD_S)
        abd_rad, _, _ = print_state(model, data, "拇指最大外展")
        print(f"    ★ 拇指外展实际到达: {np.degrees(abd_rad):.1f}° (目标: {np.degrees(THUMB_ROT_MAX):.1f}°)")

        # ═══ Phase 2: 拇指外展 + 食指弯曲 (尝试碰触) ═══
        print("\n" + "=" * 70)
        print("Phase 2: 拇指外展最大 + 食指弯曲最大 (尝试碰触)")
        print("=" * 70)
        ctrl_ph2 = ctrl_ph1.copy()
        ctrl_ph2[2] = INDEX_MAX
        ramp_pose(viewer, model, data, ctrl_ph1, ctrl_ph2, PHASE_S)
        hold_pose(viewer, model, data, ctrl_ph2, HOLD_S)
        _, d_if, _ = print_state(model, data, "拇指外展+食指弯曲")

        # ═══ Phase 3: 拇指全部最大 + 食指最大 ═══
        print("\n" + "=" * 70)
        print("Phase 3: 拇指外展+弯曲全最大 + 食指最大")
        print("=" * 70)
        ctrl_ph3 = ctrl_ph2.copy()
        ctrl_ph3[1] = THUMB_FLEX_MAX
        ramp_pose(viewer, model, data, ctrl_ph2, ctrl_ph3, PHASE_S)
        hold_pose(viewer, model, data, ctrl_ph3, HOLD_S)
        print_state(model, data, "全最大")

        # ═══ Phase 4: 保持，让用户观察 ═══
        print("\n" + "=" * 70)
        print("Phase 4: 保持最佳姿态 — 请在viewer中观察")
        print("  → 对比硬件上拇指的实际外展范围")
        print("  → 仿真中拇指外展到最大时的位置是否和硬件一致?")
        print("=" * 70)

        # 回到最佳姿态 (Phase 2, 不加flex)
        ctrl_best = np.zeros(nu, dtype=np.float32)
        ctrl_best[0] = THUMB_ROT_MAX
        ctrl_best[2] = INDEX_MAX
        ramp_pose(viewer, model, data, ctrl_ph3, ctrl_best, PHASE_S)

        print(f"\n  总结:")
        print(f"    拇指外展关节极限: {np.degrees(jnt_hi):.1f}° ({jnt_hi:.4f} rad)")
        print(f"    拇指-食指最小距离: {d_if:.1f}mm")
        print(f"    若硬件能碰到但仿真不能，优先检查:")
        print(f"      1. 拇指/食指碰撞体或支撑台是否干涉")
        print(f"      2. 拇指几何、site位置、旋转轴方向是否和硬件一致")
        print(f"      3. CMC_FLEX被动耦合是否符合实物连杆")
        print(f"\n关闭窗口退出。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
