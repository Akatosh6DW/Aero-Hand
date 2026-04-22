#!/home/ll/miniconda3/envs/aero_rl/bin/python
"""Aero-Hand 抓取初始状态交互编辑器。

功能:
  - 打开和 `view.py` 类似的 MuJoCo 场景查看器。
  - 用 ctrl 滑条调手指关节目标角。
  - 用 `cube_freejoint` 的 qpos 或快捷键调待抓取物体的初始位姿。
  - 编辑模式下自动去除支撑台、关闭重力，方便直接摆放瓶子/物体。
  - 编辑模式下临时弱化待抓物体的惯性响应，避免被手指轻碰就飞走。
  - 把当前状态导出成 JSON，便于整理成抓取初始数据。

常用运行方式:
  cd /home/ll/SRTP/Aero-Hand
  /home/ll/miniconda3/envs/aero_rl/bin/python edit_grasp_initial_state.py \
    --scene can_330ml \
    --mode overlay \
    --show-sites

交互说明:
  - 手指: 用 viewer 右侧的 ctrl 滑条调节。
  - 物体: 用 `cube_freejoint` 相关 qpos，或直接用快捷键微调。
  - F8: 保存当前状态到 JSON。
  - F9: 在终端打印当前状态摘要。
  - F10: 重置回 XML 的 `home` keyframe。
  - F11: 重新打印帮助。
  - Insert/Delete: 物体 x 负/正方向平移。
  - Home/End: 物体 y 正/负方向平移。
  - PageUp/PageDown: 物体 z 正/负方向平移。
  - 主键盘 6/7, 8/9, -/=: 分别绕 x/y/z 轴负/正方向旋转。

注意:
  - 不再使用 `I/J/K/L/U/O/1..6/S/P/R/H` 这组字母数字键，因为它们会和 MuJoCo viewer 自带快捷键冲突。
  - 也不直接使用 `!@#$` 这类符号键；MuJoCo 回调拿到的是基础 keycode，`!` 会和 `1` 视为同一个键。

关闭 viewer 后，脚本会自动把当前状态再保存一次。
"""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

import view as view_utils


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUT_DIR = REPO_ROOT / "temp_initial_state"
OBJECT_TRANSLATION_STEP = 0.005
OBJECT_ROTATION_STEP_DEG = 5.0
SUPPORT_HIDE_POS = np.array([0.0, 0.0, -5.0], dtype=float)
SUPPORT_HIDE_QUAT = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
EDIT_OBJECT_MIN_MASS_KG = 12.0
EDIT_OBJECT_MIN_INERTIA = np.array([0.08, 0.08, 0.08], dtype=float)
EDIT_OBJECT_MIN_DAMPING = 80.0
MJKEY_INSERT = 260
MJKEY_DELETE = 261
MJKEY_PAGE_UP = 266
MJKEY_PAGE_DOWN = 267
MJKEY_HOME = 268
MJKEY_END = 269
MJKEY_F8 = 297
MJKEY_F9 = 298
MJKEY_F10 = 299
MJKEY_F11 = 300


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open a grasp scene editor. Adjust hand joints with ctrl sliders "
            "and object pose with qpos sliders, then export the resulting "
            "initial state."
        )
    )
    parser.add_argument(
        "--scene",
        choices=sorted(view_utils.SCENE_PRESETS.keys()),
        default=view_utils.SCENE_NAME,
        help="Scene preset to load.",
    )
    parser.add_argument(
        "--mode",
        choices=("visual", "collision", "overlay"),
        default="overlay",
        help="visual=STL only, collision=physical collision geoms, overlay=show both.",
    )
    parser.add_argument(
        "--camera",
        default="free",
        help="Initial camera. Use free, or a fixed XML camera name such as side/palm.",
    )
    parser.add_argument(
        "--show-sites",
        action="store_true",
        help="Show MuJoCo sites such as grasp_site and object grasp-band markers.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="JSON output path. Defaults to temp_initial_state/<scene>_initial_state.json",
    )
    return parser.parse_args()


def _joint_qpos_width(joint_type: int) -> int:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return 7
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return 4
    return 1


def _joint_ctrl_width(joint_type: int) -> str:
    if joint_type == int(mujoco.mjtJoint.mjJNT_FREE):
        return "pos xyz + quat wxyz"
    if joint_type == int(mujoco.mjtJoint.mjJNT_BALL):
        return "quat wxyz"
    return "scalar"


def _joint_name(model: mujoco.MjModel, joint_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"


def _body_name(model: mujoco.MjModel, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"


def _object_freejoint_info(model: mujoco.MjModel) -> tuple[int, str, int, str] | None:
    freejoint_type = int(mujoco.mjtJoint.mjJNT_FREE)
    candidates: list[tuple[int, str, int, str]] = []
    for joint_id in range(model.njnt):
        if int(model.jnt_type[joint_id]) != freejoint_type:
            continue
        joint_name = _joint_name(model, joint_id)
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = _body_name(model, body_id)
        candidates.append((joint_id, joint_name, body_id, body_name))

    if not candidates:
        return None

    preferred_tokens = ("cube", "bottle", "object")
    for candidate in candidates:
        _, joint_name, _, body_name = candidate
        if any(token in joint_name for token in preferred_tokens) or any(
            token in body_name for token in preferred_tokens
        ):
            return candidate
    return candidates[0]


def _iter_hand_hinge_joints(model: mujoco.MjModel) -> list[tuple[int, str, int]]:
    joints: list[tuple[int, str, int]] = []
    for joint_id in range(model.njnt):
        joint_type = int(model.jnt_type[joint_id])
        if joint_type != int(mujoco.mjtJoint.mjJNT_HINGE):
            continue
        joint_name = _joint_name(model, joint_id)
        if not joint_name.startswith("right_"):
            continue
        joints.append((joint_id, joint_name, int(model.jnt_qposadr[joint_id])))
    return joints


def _iter_actuators(model: mujoco.MjModel) -> list[tuple[int, str, str]]:
    actuators: list[tuple[int, str, str]] = []
    for actuator_id in range(model.nu):
        actuator_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            or f"actuator_{actuator_id}"
        )
        trn_type = int(model.actuator_trntype[actuator_id])
        trn0 = int(model.actuator_trnid[actuator_id, 0])
        target_name = "?"
        if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn0 >= 0:
            target_name = _joint_name(model, trn0)
        actuators.append((actuator_id, actuator_name, target_name))
    return actuators


def _iter_mocap_bodies(model: mujoco.MjModel) -> list[tuple[str, int]]:
    bodies: list[tuple[str, int]] = []
    for body_id in range(model.nbody):
        mocap_id = int(model.body_mocapid[body_id])
        if mocap_id < 0:
            continue
        bodies.append((_body_name(model, body_id), mocap_id))
    return bodies


def _is_support_entity(name: str | None) -> bool:
    return "support" in (name or "").lower()


def _disable_support_for_editing(model: mujoco.MjModel, data: mujoco.MjData) -> bool:
    changed = False

    for geom_id in range(model.ngeom):
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        body_id = int(model.geom_bodyid[geom_id])
        body_name = _body_name(model, body_id)
        if not (_is_support_entity(geom_name) or _is_support_entity(body_name)):
            continue

        model.geom_contype[geom_id] = 0
        model.geom_conaffinity[geom_id] = 0
        model.geom_rgba[geom_id, 3] = 0.0
        changed = True

    for body_name, mocap_id in _iter_mocap_bodies(model):
        if not _is_support_entity(body_name):
            continue
        data.mocap_pos[mocap_id] = SUPPORT_HIDE_POS
        data.mocap_quat[mocap_id] = SUPPORT_HIDE_QUAT
        changed = True

    if changed:
        mujoco.mj_forward(model, data)
    return changed


def _stabilize_object_for_editing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
) -> bool:
    if object_info is None:
        return False

    joint_id, _, body_id, _ = object_info
    changed = False
    saved_qpos = np.array(data.qpos, dtype=float)
    saved_qvel = np.array(data.qvel, dtype=float)
    saved_ctrl = np.array(data.ctrl, dtype=float)
    saved_mocap_pos = np.array(data.mocap_pos, dtype=float)
    saved_mocap_quat = np.array(data.mocap_quat, dtype=float)

    if float(model.body_mass[body_id]) < EDIT_OBJECT_MIN_MASS_KG:
        model.body_mass[body_id] = EDIT_OBJECT_MIN_MASS_KG
        changed = True

    target_inertia = np.maximum(
        np.asarray(model.body_inertia[body_id], dtype=float),
        EDIT_OBJECT_MIN_INERTIA,
    )
    if not np.allclose(model.body_inertia[body_id], target_inertia):
        model.body_inertia[body_id] = target_inertia
        changed = True

    dof_addr = int(model.jnt_dofadr[joint_id])
    damping_slice = np.asarray(model.dof_damping[dof_addr:dof_addr + 6], dtype=float)
    target_damping = np.maximum(damping_slice, EDIT_OBJECT_MIN_DAMPING)
    if not np.allclose(damping_slice, target_damping):
        model.dof_damping[dof_addr:dof_addr + 6] = target_damping
        changed = True

    if changed:
        mujoco.mj_setConst(model, data)
        data.qpos[:] = saved_qpos
        data.qvel[:] = saved_qvel
        data.ctrl[:] = saved_ctrl
        data.mocap_pos[:] = saved_mocap_pos
        data.mocap_quat[:] = saved_mocap_quat
    data.qvel[dof_addr:dof_addr + 6] = 0.0
    mujoco.mj_forward(model, data)
    return changed


def _freeze_object_motion_for_editing(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
) -> None:
    if object_info is None:
        return
    joint_id, _, _, _ = object_info
    dof_addr = int(model.jnt_dofadr[joint_id])
    data.qvel[dof_addr:dof_addr + 6] = 0.0


def _equilibrate_and_forward(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    view_utils._equilibrate_ctrl(model, data)
    mujoco.mj_forward(model, data)


def _forward_only(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)


def _default_output_path(scene_name: str) -> Path:
    return DEFAULT_OUT_DIR / f"{scene_name}_initial_state.json"


def _normalize_quat(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(quat_wxyz))
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    return quat_wxyz / norm


def _quat_mul(lhs_wxyz: np.ndarray, rhs_wxyz: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = lhs_wxyz
    rw, rx, ry, rz = rhs_wxyz
    return np.array([
        lw * rw - lx * rx - ly * ry - lz * rz,
        lw * rx + lx * rw + ly * rz - lz * ry,
        lw * ry - lx * rz + ly * rw + lz * rx,
        lw * rz + lx * ry - ly * rx + lz * rw,
    ], dtype=float)


def _axis_angle_quat(axis_xyz: np.ndarray, angle_rad: float) -> np.ndarray:
    axis_xyz = np.asarray(axis_xyz, dtype=float)
    axis_xyz = axis_xyz / max(float(np.linalg.norm(axis_xyz)), 1e-12)
    half = angle_rad * 0.5
    sin_half = np.sin(half)
    return np.array(
        [np.cos(half), axis_xyz[0] * sin_half, axis_xyz[1] * sin_half, axis_xyz[2] * sin_half],
        dtype=float,
    )


def _set_object_qpos(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
    position_xyz: np.ndarray,
    quaternion_wxyz: np.ndarray,
) -> None:
    if object_info is None:
        return

    joint_id, _, _, _ = object_info
    qpos_addr = int(model.jnt_qposadr[joint_id])
    dof_addr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_addr:qpos_addr + 3] = np.asarray(position_xyz, dtype=float)
    data.qpos[qpos_addr + 3:qpos_addr + 7] = _normalize_quat(quaternion_wxyz)
    data.qvel[dof_addr:dof_addr + 6] = 0.0
    _forward_only(model, data)


def _apply_object_translation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
    delta_xyz: np.ndarray,
) -> None:
    if object_info is None:
        return
    joint_id, _, _, _ = object_info
    qpos_addr = int(model.jnt_qposadr[joint_id])
    position = np.array(data.qpos[qpos_addr:qpos_addr + 3], dtype=float) + np.asarray(delta_xyz, dtype=float)
    quat = np.array(data.qpos[qpos_addr + 3:qpos_addr + 7], dtype=float)
    _set_object_qpos(model, data, object_info, position, quat)


def _apply_object_rotation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
    axis_xyz: np.ndarray,
    angle_rad: float,
) -> None:
    if object_info is None:
        return
    joint_id, _, _, _ = object_info
    qpos_addr = int(model.jnt_qposadr[joint_id])
    position = np.array(data.qpos[qpos_addr:qpos_addr + 3], dtype=float)
    quat = np.array(data.qpos[qpos_addr + 3:qpos_addr + 7], dtype=float)
    delta_quat = _axis_angle_quat(axis_xyz, angle_rad)
    rotated = _quat_mul(delta_quat, quat)
    _set_object_qpos(model, data, object_info, position, rotated)


def _format_vec(values: np.ndarray | list[float], precision: int = 6) -> str:
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def _collect_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    scene_name: str,
    xml_path: Path,
    object_info: tuple[int, str, int, str] | None,
) -> dict:
    _forward_only(model, data)

    joint_angles = {
        joint_name: float(data.qpos[qpos_addr])
        for _, joint_name, qpos_addr in _iter_hand_hinge_joints(model)
    }

    actuator_targets = {}
    for actuator_id in range(model.nu):
        actuator_name = (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
            or f"actuator_{actuator_id}"
        )
        actuator_targets[actuator_name] = float(data.ctrl[actuator_id])

    object_state = None
    if object_info is not None:
        joint_id, joint_name, body_id, body_name = object_info
        qpos_addr = int(model.jnt_qposadr[joint_id])
        object_qpos = np.array(data.qpos[qpos_addr:qpos_addr + 7], dtype=float)
        object_state = {
            "body_name": body_name,
            "freejoint_name": joint_name,
            "qpos_wxyz": object_qpos.tolist(),
            "position": object_qpos[:3].tolist(),
            "quaternion_wxyz": object_qpos[3:].tolist(),
            "world_position": np.array(data.xpos[body_id], dtype=float).tolist(),
            "world_quaternion_wxyz": np.array(data.xquat[body_id], dtype=float).tolist(),
        }

    mocap_bodies = {}
    for body_name, mocap_id in _iter_mocap_bodies(model):
        mocap_bodies[body_name] = {
            "mocap_id": mocap_id,
            "position": np.array(data.mocap_pos[mocap_id], dtype=float).tolist(),
            "quaternion_wxyz": np.array(data.mocap_quat[mocap_id], dtype=float).tolist(),
        }

    grasp_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    grasp_site = None
    object_minus_grasp = None
    if grasp_site_id >= 0:
        grasp_site = np.array(data.site_xpos[grasp_site_id], dtype=float).tolist()
        if object_state is not None:
            object_minus_grasp = (
                np.array(object_state["world_position"], dtype=float)
                - np.array(grasp_site, dtype=float)
            ).tolist()

    qpos = np.array(data.qpos, dtype=float)
    ctrl = np.array(data.ctrl, dtype=float)
    mpos = np.array(data.mocap_pos, dtype=float).reshape(-1)
    mquat = np.array(data.mocap_quat, dtype=float).reshape(-1)

    return {
        "scene": scene_name,
        "xml_path": str(xml_path),
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "joint_angles": joint_angles,
        "actuator_targets": actuator_targets,
        "object": object_state,
        "grasp_site_position": grasp_site,
        "object_minus_grasp_site": object_minus_grasp,
        "mocap_bodies": mocap_bodies,
        "mjcf_keyframe": {
            "qpos": " ".join(f"{v:.8f}" for v in qpos),
            "ctrl": " ".join(f"{v:.8f}" for v in ctrl),
            "mpos": " ".join(f"{v:.8f}" for v in mpos),
            "mquat": " ".join(f"{v:.8f}" for v in mquat),
        },
        "raw_state": {
            "qpos": qpos.tolist(),
            "ctrl": ctrl.tolist(),
            "mpos": mpos.tolist(),
            "mquat": mquat.tolist(),
        },
    }


def _save_state(
    output_path: Path,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    scene_name: str,
    xml_path: Path,
    object_info: tuple[int, str, int, str] | None,
    reason: str,
) -> dict:
    state = _collect_state(
        model,
        data,
        scene_name=scene_name,
        xml_path=xml_path,
        object_info=object_info,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"[save] {reason}: {output_path}")
    return state


def _print_state_summary(state: dict) -> None:
    print("\nCurrent grasp init state")
    if state.get("object"):
        obj = state["object"]
        print(f"  object body      : {obj['body_name']}")
        print(f"  object position  : {_format_vec(obj['position'])}")
        print(f"  object quat wxyz : {_format_vec(obj['quaternion_wxyz'])}")
    if state.get("grasp_site_position") is not None:
        print(f"  grasp_site       : {_format_vec(state['grasp_site_position'])}")
    if state.get("object_minus_grasp_site") is not None:
        print(f"  object-grasp     : {_format_vec(state['object_minus_grasp_site'])}")
    print("  hand joints:")
    for joint_name, value in state["joint_angles"].items():
        print(f"    {joint_name:24s} {value: .6f}")


def _print_help(output_path: Path) -> None:
    print("\nViewer controls")
    print("  Use ctrl sliders for hand joints; use the cube_freejoint qpos sliders for object pose.")
    print("  Edit mode keeps gravity off and removes support-platform geoms from the scene.")
    print("  Edit mode also makes the object quasi-static, so fingertip bumps do not fling it away.")
    print("  Note: object hotkeys avoid MuJoCo built-in viewer shortcuts on purpose.")
    print(f"  F8: save current state to {output_path}")
    print("  F9: print current state summary")
    print("  F10: reset to XML home keyframe")
    print("  F11: print this help again")
    print(
        f"  Insert/Delete: object x -/+   Home/End: object y + / -   "
        f"PageUp/PageDown: object z + / -   "
        f"(step = {OBJECT_TRANSLATION_STEP:.3f} m)"
    )
    print(
        f"  Top-row 6/7: roll -/+   8/9: pitch -/+   -/=: yaw -/+   "
        f"(step = {OBJECT_ROTATION_STEP_DEG:.1f} deg)"
    )


def _print_qpos_layout(model: mujoco.MjModel, object_info: tuple[int, str, int, str] | None) -> None:
    print("\nEditable qpos layout")
    print("  注意: 这是状态量列表，不是执行器列表。GUI 里看到 right_index_mcp / right_ring_pip 很正常。")
    if object_info is not None:
        joint_id, joint_name, _, body_name = object_info
        qpos_addr = int(model.jnt_qposadr[joint_id])
        print(
            f"  {joint_name:24s} qpos[{qpos_addr}:{qpos_addr + 7}]  "
            f"{_joint_ctrl_width(int(model.jnt_type[joint_id]))}  body={body_name}"
        )
    for joint_id, joint_name, qpos_addr in _iter_hand_hinge_joints(model):
        print(
            f"  {joint_name:24s} qpos[{qpos_addr}]      "
            f"{_joint_ctrl_width(int(model.jnt_type[joint_id]))}"
        )


def _print_ctrl_layout(model: mujoco.MjModel) -> None:
    print("\nEditable ctrl layout")
    print("  这些才是手指真正有的执行器目标；在 GUI 里请优先调这一组。")
    for actuator_id, actuator_name, target_name in _iter_actuators(model):
        print(
            f"  {actuator_name:24s} ctrl[{actuator_id}]  target_joint={target_name}"
        )


def _reset_home(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None = None,
) -> None:
    view_utils._apply_home_keyframe(model, data)
    _equilibrate_and_forward(model, data)
    _disable_support_for_editing(model, data)
    _stabilize_object_for_editing(model, data, object_info)


def _print_object_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    object_info: tuple[int, str, int, str] | None,
) -> None:
    if object_info is None:
        print("[object] no freejoint object found in this scene")
        return
    joint_id, _, _, body_name = object_info
    qpos_addr = int(model.jnt_qposadr[joint_id])
    pos = np.array(data.qpos[qpos_addr:qpos_addr + 3], dtype=float)
    quat = np.array(data.qpos[qpos_addr + 3:qpos_addr + 7], dtype=float)
    print(f"[object] {body_name} pos={_format_vec(pos)} quat_wxyz={_format_vec(quat)}")


def main() -> None:
    args = parse_args()
    xml_path = view_utils._select_xml_path(args.scene)
    output_path = (args.output or _default_output_path(args.scene)).expanduser().resolve()

    if not xml_path.exists():
        print(f"❌ 找不到文件：{xml_path}")
        return

    os.chdir(xml_path.parent)
    model = mujoco.MjModel.from_xml_path(xml_path.name)
    view_utils._apply_display_mode_to_model(model, args.mode)
    model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    object_info = _object_freejoint_info(model)
    _reset_home(model, data, object_info)
    support_removed = _disable_support_for_editing(model, data)
    _stabilize_object_for_editing(model, data, object_info)
    object_stabilized = object_info is not None

    print(f"⏳ 正在加载场景: {args.scene}")
    print(f"📄 XML: {xml_path.name}")
    print(f"🎛️ 显示模式: {args.mode}")
    print(f"💾 导出路径: {output_path}")
    print("🧲 编辑器模式: 已关闭重力，便于拖动对象 freejoint 和调手指目标角。")
    if support_removed:
        print("🪄 已移除支撑台显示并禁用其碰撞；保存出的状态也会保留这一编辑器设置。")
    if object_stabilized:
        print("🧱 已临时提高待抓物体质量/惯量并锁住 freejoint 速度，避免被手指轻碰撞飞。")
    _print_help(output_path)
    _print_ctrl_layout(model)
    _print_qpos_layout(model, object_info)

    if not view_utils._has_interactive_display():
        state = _save_state(
            output_path,
            model,
            data,
            scene_name=args.scene,
            xml_path=xml_path,
            object_info=object_info,
            reason="headless export",
        )
        _print_state_summary(state)
        return

    pending = {
        "save": False,
        "print": False,
        "reset": False,
        "help": False,
        "object_ops": [],
    }

    def _key_callback(keycode: int) -> None:
        if keycode == MJKEY_F8:
            pending["save"] = True
        elif keycode == MJKEY_F9:
            pending["print"] = True
        elif keycode == MJKEY_F10:
            pending["reset"] = True
        elif keycode == MJKEY_F11:
            pending["help"] = True
        elif keycode == MJKEY_INSERT:
            pending["object_ops"].append(("translate", np.array([-OBJECT_TRANSLATION_STEP, 0.0, 0.0])))
        elif keycode == MJKEY_DELETE:
            pending["object_ops"].append(("translate", np.array([OBJECT_TRANSLATION_STEP, 0.0, 0.0])))
        elif keycode == MJKEY_HOME:
            pending["object_ops"].append(("translate", np.array([0.0, OBJECT_TRANSLATION_STEP, 0.0])))
        elif keycode == MJKEY_END:
            pending["object_ops"].append(("translate", np.array([0.0, -OBJECT_TRANSLATION_STEP, 0.0])))
        elif keycode == MJKEY_PAGE_UP:
            pending["object_ops"].append(("translate", np.array([0.0, 0.0, OBJECT_TRANSLATION_STEP])))
        elif keycode == MJKEY_PAGE_DOWN:
            pending["object_ops"].append(("translate", np.array([0.0, 0.0, -OBJECT_TRANSLATION_STEP])))
        elif keycode == ord("6"):
            pending["object_ops"].append(("rotate", np.array([1.0, 0.0, 0.0]), -np.deg2rad(OBJECT_ROTATION_STEP_DEG)))
        elif keycode == ord("7"):
            pending["object_ops"].append(("rotate", np.array([1.0, 0.0, 0.0]), np.deg2rad(OBJECT_ROTATION_STEP_DEG)))
        elif keycode == ord("8"):
            pending["object_ops"].append(("rotate", np.array([0.0, 1.0, 0.0]), -np.deg2rad(OBJECT_ROTATION_STEP_DEG)))
        elif keycode == ord("9"):
            pending["object_ops"].append(("rotate", np.array([0.0, 1.0, 0.0]), np.deg2rad(OBJECT_ROTATION_STEP_DEG)))
        elif keycode == ord("-"):
            pending["object_ops"].append(("rotate", np.array([0.0, 0.0, 1.0]), -np.deg2rad(OBJECT_ROTATION_STEP_DEG)))
        elif keycode == ord("="):
            pending["object_ops"].append(("rotate", np.array([0.0, 0.0, 1.0]), np.deg2rad(OBJECT_ROTATION_STEP_DEG)))

    opt = view_utils._make_scene_option(args.mode, args.show_sites)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=_key_callback,
        show_left_ui=True,
        show_right_ui=True,
    ) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = opt.geomgroup
            viewer.opt.flags[:] = opt.flags
            if args.camera == "free":
                view_utils._configure_free_camera(
                    viewer.cam,
                    model,
                    data,
                    display_mode=args.mode,
                    show_sites=args.show_sites,
                )
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = view_utils._camera_id(model, args.camera)

        while viewer.is_running():
            if pending["help"]:
                _print_help(output_path)
                pending["help"] = False
            if pending["reset"]:
                _reset_home(model, data, object_info)
                print("[reset] restored XML home keyframe")
                pending["reset"] = False
            if pending["print"]:
                state = _collect_state(
                    model,
                    data,
                    scene_name=args.scene,
                    xml_path=xml_path,
                    object_info=object_info,
                )
                _print_state_summary(state)
                pending["print"] = False
            if pending["save"]:
                _save_state(
                    output_path,
                    model,
                    data,
                    scene_name=args.scene,
                    xml_path=xml_path,
                    object_info=object_info,
                    reason="manual save",
                )
                pending["save"] = False

            while pending["object_ops"]:
                op = pending["object_ops"].pop(0)
                if op[0] == "translate":
                    _apply_object_translation(model, data, object_info, op[1])
                else:
                    _, axis_xyz, angle_rad = op
                    _apply_object_rotation(model, data, object_info, axis_xyz, angle_rad)
                _print_object_pose(model, data, object_info)

            _freeze_object_motion_for_editing(model, data, object_info)
            mujoco.mj_step(model, data)
            _freeze_object_motion_for_editing(model, data, object_info)
            viewer.sync()
            time.sleep(1.0 / 60.0)

    final_state = _save_state(
        output_path,
        model,
        data,
        scene_name=args.scene,
        xml_path=xml_path,
        object_info=object_info,
        reason="auto save on close",
    )
    _print_state_summary(final_state)


if __name__ == "__main__":
    main()
