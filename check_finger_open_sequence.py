import os
import time

import mujoco
import mujoco.viewer
import numpy as np


DEFAULT_XML = "/home/ll/SRTP/Aero-Hand/sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_hw6_palmup.xml"
ZERO_GRAVITY = True
PHASE_S = 1.8


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
    """一键检查脚本：直接运行即可，不需要传参。"""
    xml_path = os.path.abspath(DEFAULT_XML)
    if not os.path.exists(xml_path):
        raise FileNotFoundError(f"XML not found: {xml_path}")

    xml_dir = os.path.dirname(xml_path)
    xml_name = os.path.basename(xml_path)
    os.chdir(xml_dir)

    model = mujoco.MjModel.from_xml_path(xml_name)
    if ZERO_GRAVITY:
        model.opt.gravity[:] = 0.0
        print("[check] zero gravity enabled:", model.opt.gravity)

    data = mujoco.MjData(model)
    apply_home_keyframe(model, data)

    ctrl_low = np.asarray(model.actuator_ctrlrange[:, 0], dtype=np.float32)
    ctrl_high = np.asarray(model.actuator_ctrlrange[:, 1], dtype=np.float32)

    # 当前 HW6 方向定义:
    # - 拇指外展(0): 低值=更张开, 高值=更收拢
    # - 拇指弯曲(1): 高值=更张开, 低值=更收拢
    # - 四指(2..5): 低值=更收拢, 高值=更张开
    ctrl_fist = ctrl_low.copy()
    ctrl_fist[0] = ctrl_high[0]
    ctrl_fist[1] = ctrl_low[1]

    ctrl_open = ctrl_high.copy()
    ctrl_open[0] = ctrl_low[0]
    ctrl_open[1] = ctrl_high[1]

    sequence = [
        ("thumb", [0, 1]),
        ("index", [2]),
        ("middle", [3]),
        ("ring", [4]),
        ("pinky", [5]),
    ]

    print("[check] actuator ctrlrange low :", np.round(ctrl_low, 6).tolist())
    print("[check] actuator ctrlrange high:", np.round(ctrl_high, 6).tolist())
    print("[check] start fist ctrl:", np.round(ctrl_fist, 6).tolist())
    print("[check] final open ctrl:", np.round(ctrl_open, 6).tolist())

    with mujoco.viewer.launch_passive(model, data) as viewer:
        print("[check] stage 1/7: hold fist")
        hold_pose(viewer, model, data, ctrl_fist, PHASE_S)

        current = ctrl_fist.copy()
        stage_idx = 2
        for name, idxs in sequence:
            target = current.copy()
            for idx in idxs:
                target[idx] = ctrl_open[idx]
            print(f"[check] stage {stage_idx}/7: open {name}")
            ramp_pose(viewer, model, data, current, target, PHASE_S)
            hold_pose(viewer, model, data, target, 0.5)
            current = target
            stage_idx += 1

        print("[check] stage 7/7: hold all open")
        hold_pose(viewer, model, data, current, PHASE_S)

        print("[check] done. You can close the viewer.")
        while viewer.is_running():
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
