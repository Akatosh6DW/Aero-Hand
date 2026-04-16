"""V2 灵犀手手指开合测试 - 验证手掌与手指连接是否正常。
直接运行即可: python check_finger_open_v2.py
"""
import os
import time

import mujoco
import mujoco.viewer
import numpy as np

# V2 训练场景 XML（vertical 版本，和训练完全相同）
DEFAULT_XML = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"
ZERO_GRAVITY = True
PHASE_S = 2.0


def apply_home_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        print("[warn] no 'home' keyframe found")
        return
    nq, nu = model.nq, model.nu
    data.qpos[:] = model.key_qpos[key_id * nq : (key_id + 1) * nq]
    if nu > 0:
        data.ctrl[:] = model.key_ctrl[key_id * nu : (key_id + 1) * nu]
    if model.nmocap > 0:
        data.mocap_pos[:] = model.key_mpos[key_id * model.nmocap : (key_id + 1) * model.nmocap]
        data.mocap_quat[:] = model.key_mquat[key_id * model.nmocap : (key_id + 1) * model.nmocap]
    mujoco.mj_forward(model, data)
    print(f"[info] home keyframe applied, qpos={np.round(data.qpos[:13], 3).tolist()}")


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return a + (b - a) * t


def hold_pose(viewer, model, data, ctrl_target: np.ndarray, hold_s: float) -> None:
    t_end = time.time() + hold_s
    while viewer.is_running() and time.time() < t_end:
        data.ctrl[:] = ctrl_target
        mujoco.mj_step(model, data)
        viewer.sync()
        time.sleep(model.opt.timestep)


def ramp_pose(viewer, model, data, ctrl_from: np.ndarray, ctrl_to: np.ndarray, duration_s: float) -> None:
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


def main() -> None:
    xml_path = os.path.abspath(DEFAULT_XML)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    # chdir to xml directory so <include> works
    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.basename(xml_path)
    os.chdir(xml_dir)

    model = mujoco.MjModel.from_xml_path(xml_name)
    if ZERO_GRAVITY:
        model.opt.gravity[:] = 0.0
        print("[check] zero gravity enabled")

    data = mujoco.MjData(model)
    apply_home_keyframe(model, data)

    # V2 执行器: [thumb_rot, thumb_flex, index, middle, ring, pinky] (6通道)
    nu = model.nu
    print(f"[check] nu={nu} actuators")
    for i in range(nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        low, high = model.actuator_ctrlrange[i]
        print(f"  [{i}] {name}: range=[{low:.4f}, {high:.4f}]")

    ctrl_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32)

    # V2: 所有执行器 high=弯曲/收拢, low=伸展/张开
    # thumb_rot(0): high=收拢, low=张开
    # thumb_flex(1): high=弯曲, low=伸展
    # index-pinky(2-5): high=弯曲, low=伸展
    ctrl_fist = ctrl_high.copy()   # 全弯曲 = 握拳
    ctrl_open = ctrl_low.copy()    # 全伸展 = 张开

    # 手指序列：逐个打开
    sequence = [
        ("拇指(thumb)", [0, 1]),
        ("食指(index)", [2]),
        ("中指(middle)", [3]),
        ("无名指(ring)", [4]),
        ("小指(pinky)", [5]),
    ]

    print(f"\n[check] ctrl_fist (握拳): {np.round(ctrl_fist, 4).tolist()}")
    print(f"[check] ctrl_open (张开): {np.round(ctrl_open, 4).tolist()}")

    # Print body hierarchy for verification
    print("\n[check] Body hierarchy (验证手掌-手指连接):")
    for i in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        parent_id = model.body_parentid[i]
        parent_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, parent_id) if parent_id >= 0 else "world"
        if name and ("palm" in name or "right_" in name or "mount" in name or "cube" in name):
            pos = model.body_pos[i]
            print(f"  {name} (parent={parent_name}, pos={np.round(pos, 4).tolist()})")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("\n===== 开始手指开合测试 =====")
        print("[1/7] 握拳（全弯曲）")
        hold_pose(viewer, model, data, ctrl_fist, PHASE_S)

        current = ctrl_fist.copy()
        stage_idx = 2
        for name, idxs in sequence:
            target = current.copy()
            for idx in idxs:
                target[idx] = ctrl_open[idx]
            print(f"[{stage_idx}/7] 打开 {name}")
            ramp_pose(viewer, model, data, current, target, PHASE_S)
            hold_pose(viewer, model, data, target, 0.5)
            current = target
            stage_idx += 1

        print("[7/7] 保持全张开")
        hold_pose(viewer, model, data, current, PHASE_S)

        # 再做一次全握拳 -> 全张开的循环
        print("\n[额外] 从张开到握拳再张开（整体同步）")
        ramp_pose(viewer, model, data, ctrl_open, ctrl_fist, PHASE_S)
        hold_pose(viewer, model, data, ctrl_fist, 1.0)
        ramp_pose(viewer, model, data, ctrl_fist, ctrl_open, PHASE_S)
        hold_pose(viewer, model, data, ctrl_open, 1.0)

        print("\n[完成] 测试结束。请在viewer中检查：")
        print("  1. 手指是否都连在手掌上")
        print("  2. 弯曲方向是否正确")
        print("  3. 五根手指是否都能独立控制")
        print("关闭窗口退出。")

        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
