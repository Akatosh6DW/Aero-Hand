#!/home/ll/miniconda3/envs/aero_rl/bin/python
"""Visualize V2 STL meshes and the actual MuJoCo collision geometry together.

The XML keeps visual STL geoms in group 2 and physical collision geoms in
group 3.  This script loads either the training box scene or the generated
CoACD scene, recolors those loaded geoms in memory, saves quick reference
screenshots, and then opens the MuJoCo viewer when a display is available.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_XML = (
    REPO_ROOT
    / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2.xml"
)
COACD_XML = (
    REPO_ROOT
    / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls/scene_mjx_grasp_v2_coacd.xml"
)
DEFAULT_OUT_DIR = REPO_ROOT / "collision_overlay_outputs"

VISUAL_GROUP = 2
COLLISION_GROUP = 3
CUBE_GROUP = 1
SITE_GROUP = 4

STL_RGBA = np.array([0.05, 0.45, 1.00, 0.32], dtype=np.float32)
COLLISION_RGBA = np.array([1.00, 0.12, 0.04, 0.55], dtype=np.float32)
TIP_COLLISION_RGBA = np.array([1.00, 0.90, 0.02, 0.75], dtype=np.float32)
CUBE_RGBA = np.array([0.08, 0.85, 0.25, 0.55], dtype=np.float32)
SUPPORT_RGBA = np.array([1.00, 0.85, 0.10, 0.38], dtype=np.float32)
FLOOR_RGBA = np.array([0.85, 0.85, 0.85, 0.06], dtype=np.float32)

COLLISION_OPAQUE_RGBA = np.array([0.85, 0.25, 0.15, 1.0], dtype=np.float32)
TIP_COLLISION_OPAQUE_RGBA = np.array([0.95, 0.80, 0.05, 1.0], dtype=np.float32)

TIP_COLLISION_NAMES = {
    "if_tip_col",
    "mf_tip_col",
    "rf_tip_col",
    "pf_tip_col",
    "th_tip_col_1",
    "th_tip_col_2",
}

# CoACD-lite tip bodies — any geom whose name starts with these prefixes
# is a fingertip collision mesh and should be highlighted in yellow.
_COACD_TIP_PREFIXES = (
    "v2_index_distal_coacd_",
    "v2_middle_distal_coacd_",
    "v2_ring_distal_coacd_",
    "v2_pinky_distal_coacd_",
    "v2_thumb_tip_coacd_",
    "right_index_distal_coacd_",
    "right_middle_distal_coacd_",
    "right_ring_distal_coacd_",
    "right_pinky_distal_coacd_",
    "right_thumb_tip_coacd_",
)

# Fitted-primitive tip bodies — same logic, different infix.
_FITTED_TIP_PREFIXES = (
    "right_index_distal_fitted_",
    "right_middle_distal_fitted_",
    "right_ring_distal_fitted_",
    "right_pinky_distal_fitted_",
    "right_thumb_tip_fitted_",
)

# Capsule tip bodies — distal/tip capsules.
_CAPSULE_TIP_PREFIXES = (
    "right_index_distal_capsule_",
    "right_middle_distal_capsule_",
    "right_ring_distal_capsule_",
    "right_pinky_distal_capsule_",
    "right_thumb_tip_capsule_",
)

PREGRASP_QPOS = {
    "right_index_mcp": 1.039,
    "right_index_pip": 0.925 * 1.039,
    "right_middle_mcp": 1.046,
    "right_middle_pip": 0.925 * 1.046,
    "right_ring_mcp": 0.0,
    "right_ring_pip": 0.0,
    "right_pinky_mcp": 0.0,
    "right_pinky_pip": 0.0,
    "right_thumb_cmc_abd": 1.260,
    "right_thumb_cmc_flex": 0.16 * 1.260,
    "right_thumb_mcp": 0.253,
}

OPEN_QPOS = {
    "right_index_mcp": 0.0,
    "right_index_pip": 0.0,
    "right_middle_mcp": 0.0,
    "right_middle_pip": 0.0,
    "right_ring_mcp": 0.0,
    "right_ring_pip": 0.0,
    "right_pinky_mcp": 0.0,
    "right_pinky_pip": 0.0,
    "right_thumb_cmc_abd": 0.0,
    "right_thumb_cmc_flex": 0.0,
    "right_thumb_mcp": 0.0,
}

# Thumb at maximum adduction/opposition — CMC abd and flex at their limits.
THUMB_MAX_QPOS = {
    "right_index_mcp": 0.0,
    "right_index_pip": 0.0,
    "right_middle_mcp": 0.0,
    "right_middle_pip": 0.0,
    "right_ring_mcp": 0.0,
    "right_ring_pip": 0.0,
    "right_pinky_mcp": 0.0,
    "right_pinky_pip": 0.0,
    "right_thumb_cmc_abd": 1.3788,
    "right_thumb_cmc_flex": 0.16 * 1.3788,
    "right_thumb_mcp": 0.7854,
}


def _name(model: mujoco.MjModel, obj_type: mujoco.mjtObj, idx: int) -> str:
    return mujoco.mj_id2name(model, obj_type, idx) or f"<unnamed_{idx}>"


def _geom_name(model: mujoco.MjModel, geom_id: int) -> str:
    return _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)


def _camera_id(model: mujoco.MjModel, camera_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id >= 0:
        return cam_id

    available = [
        _name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx) for idx in range(model.ncam)
    ]
    raise ValueError(f"camera {camera_name!r} not found; available: {available}")


def choose_xml(model_kind: str, xml_override: Path | None) -> tuple[Path, str]:
    if xml_override is not None:
        return xml_override.expanduser().resolve(), "custom"
    if model_kind == "auto":
        if COACD_XML.exists():
            return COACD_XML.resolve(), "coacd"
        return DEFAULT_XML.resolve(), "box"
    if model_kind == "coacd":
        return COACD_XML.resolve(), "coacd"
    return DEFAULT_XML.resolve(), "box"


def _keyframe_id(model: mujoco.MjModel, keyframe_name: str) -> int:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, keyframe_name)
    if key_id >= 0:
        return key_id

    available = [_name(model, mujoco.mjtObj.mjOBJ_KEY, idx) for idx in range(model.nkey)]
    raise ValueError(f"keyframe {keyframe_name!r} not found; available: {available}")


def _set_hinge_qpos(model: mujoco.MjModel, data: mujoco.MjData, qpos_by_joint: dict[str, float]) -> None:
    for joint_name, value in qpos_by_joint.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            print(f"[warn] joint {joint_name!r} not found; skipping", file=sys.stderr)
            continue
        qpos_addr = model.jnt_qposadr[joint_id]
        data.qpos[qpos_addr] = value


def _apply_pose(model: mujoco.MjModel, data: mujoco.MjData, pose: str, keyframe: str) -> None:
    mujoco.mj_resetDataKeyframe(model, data, _keyframe_id(model, keyframe))
    if pose == "pregrasp":
        _set_hinge_qpos(model, data, PREGRASP_QPOS)
    elif pose == "open":
        _set_hinge_qpos(model, data, OPEN_QPOS)
    elif pose == "thumb_max":
        _set_hinge_qpos(model, data, THUMB_MAX_QPOS)
    elif pose != "home":
        raise ValueError(f"unknown pose {pose!r}")
    mujoco.mj_forward(model, data)


def _set_material_rgba(model: mujoco.MjModel, mat_id: int, rgba: np.ndarray) -> None:
    if mat_id >= 0:
        model.mat_rgba[mat_id] = rgba


def apply_overlay_colors(model: mujoco.MjModel, *, collision_only: bool = False) -> dict[str, int]:
    """Recolor loaded geoms in memory.  The XML file is not modified.

    If *collision_only* is True, collision geoms are shown opaque and all
    other geoms are hidden (alpha=0).
    """
    counts = {
        "stl_visual_mesh_geoms": 0,
        "collision_geoms": 0,
        "collision_mesh_geoms": 0,
        "collision_box_geoms": 0,
        "tip_collision_geoms": 0,
        "cube_geoms": 0,
        "support_geoms": 0,
    }

    HIDDEN = np.array([0, 0, 0, 0], dtype=np.float32)

    for geom_id in range(model.ngeom):
        group = int(model.geom_group[geom_id])
        name = _geom_name(model, geom_id)

        if group == VISUAL_GROUP:
            rgba = HIDDEN if collision_only else STL_RGBA
            model.geom_rgba[geom_id] = rgba
            _set_material_rgba(model, int(model.geom_matid[geom_id]), rgba)
            counts["stl_visual_mesh_geoms"] += 1
        elif group == COLLISION_GROUP:
            is_tip = name in TIP_COLLISION_NAMES or name.startswith(_COACD_TIP_PREFIXES) or name.startswith(_FITTED_TIP_PREFIXES) or name.startswith(_CAPSULE_TIP_PREFIXES)
            if collision_only:
                rgba = TIP_COLLISION_OPAQUE_RGBA if is_tip else COLLISION_OPAQUE_RGBA
            else:
                rgba = TIP_COLLISION_RGBA if is_tip else COLLISION_RGBA
            model.geom_rgba[geom_id] = rgba
            _set_material_rgba(model, int(model.geom_matid[geom_id]), rgba)
            counts["collision_geoms"] += 1
            geom_type = int(model.geom_type[geom_id])
            if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                counts["collision_mesh_geoms"] += 1
            elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                counts["collision_box_geoms"] += 1
            if is_tip:
                counts["tip_collision_geoms"] += 1
        elif group == CUBE_GROUP or name == "cube":
            model.geom_rgba[geom_id] = CUBE_RGBA
            _set_material_rgba(model, int(model.geom_matid[geom_id]), CUBE_RGBA)
            counts["cube_geoms"] += 1
        elif name == "cube_support_geom":
            model.geom_rgba[geom_id] = SUPPORT_RGBA
            _set_material_rgba(model, int(model.geom_matid[geom_id]), SUPPORT_RGBA)
            counts["support_geoms"] += 1
        elif name == "floor":
            model.geom_rgba[geom_id] = FLOOR_RGBA
            _set_material_rgba(model, int(model.geom_matid[geom_id]), FLOOR_RGBA)

    return counts


def make_scene_option(*, show_sites: bool, show_contacts: bool, show_floor: bool,
                      collision_only: bool = False) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 0
    opt.geomgroup[0] = 1 if show_floor else 0
    opt.geomgroup[CUBE_GROUP] = 0 if collision_only else 1
    opt.geomgroup[VISUAL_GROUP] = 0 if collision_only else 1
    opt.geomgroup[COLLISION_GROUP] = 1
    opt.geomgroup[SITE_GROUP] = 1 if show_sites else 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = 0 if collision_only else 1
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 1 if show_contacts else 0
    opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 1 if show_contacts else 0
    return opt


def save_screenshot(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    out_dir: Path,
    camera_name: str,
    opt: mujoco.MjvOption,
    width: int,
    height: int,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)

    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = _camera_id(model, camera_name)

    renderer.update_scene(data, camera, opt)
    image = renderer.render()
    renderer.close()

    out_path = out_dir / f"v2_collision_overlay_{camera_name}.png"
    Image.fromarray(image).save(out_path)
    return out_path


def print_legend(counts: dict[str, int], xml_path: Path, pose: str, model_kind: str) -> None:
    print("\nV2 碰撞/STL 叠加显示")
    print(f"  Model: {model_kind}")
    print(f"  XML: {xml_path}")
    print(f"  Pose: {pose}")
    print("  颜色图例:")
    print("    蓝色半透明: STL 视觉网格，也就是你在视频里看到的外观")
    print("    红色半透明: MuJoCo 实际参与物理碰撞的几何 (box=盒子, coacd=凸mesh)")
    print("    黄色高亮: 指尖碰撞几何 (box=语义tip盒子, coacd=distal+thumb_tip凸mesh)")
    print("    绿色半透明: 方块")
    print("    黄色半透明: 方块支撑块")
    print("  几何统计:")
    for key, value in counts.items():
        print(f"    {key}: {value}")


def launch_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    camera_name: str,
    opt: mujoco.MjvOption,
    *,
    simulate: bool,
) -> None:
    import mujoco.viewer

    with mujoco.viewer.launch_passive(model, data, show_left_ui=True, show_right_ui=True) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = opt.geomgroup
            viewer.opt.flags[:] = opt.flags
            if camera_name == "free":
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
                viewer.cam.lookat[:] = model.stat.center
                viewer.cam.distance = float(model.stat.extent * 1.25)
                viewer.cam.azimuth = 120
                viewer.cam.elevation = -20
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = _camera_id(model, camera_name)

        print("\nviewer 已打开。关闭 MuJoCo viewer 窗口后脚本结束。")
        print("默认 free 相机可用鼠标旋转/缩放/平移；若指定 --camera side/palm 则是固定相机。")
        print("右侧 UI 里可手动开关 geom group：2=STL, 3=碰撞几何, 4=site。")
        print("拖动 qpos 关节滑条会刷新姿态；拖 ctrl 电机滑条需要加 --simulate 才会驱动模型。")
        while viewer.is_running():
            if simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Show V2 visual STL meshes and actual MuJoCo collision boxes with different colors."
    )
    parser.add_argument(
        "--model",
        choices=("auto", "box", "coacd"),
        default="auto",
        help="auto uses coacd when the generated XML exists, otherwise box. box uses the training primitive XML.",
    )
    parser.add_argument(
        "--xml",
        type=Path,
        help="MuJoCo scene XML to load. Overrides --model when supplied.",
    )
    parser.add_argument("--pose", choices=("home", "pregrasp", "open", "thumb_max"), default="open")
    parser.add_argument("--keyframe", default="home")
    parser.add_argument(
        "--camera",
        default="free",
        help="Initial viewer camera. Use free for rotatable view, or side/palm for fixed XML cameras.",
    )
    parser.add_argument(
        "--save-cameras",
        nargs="*",
        default=("side", "palm"),
        help="Camera names for PNG snapshots. Use an empty value to skip snapshots.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=960)
    parser.add_argument("--show-sites", action="store_true", help="Also show MuJoCo sites/tactile markers.")
    parser.add_argument("--show-contacts", action="store_true", help="Show MuJoCo contact points/forces.")
    parser.add_argument(
        "--collision-only",
        action="store_true",
        default=False,
        help="Only show collision geoms (opaque, no STL overlay, no transparency). "
             "Red=general collision, yellow=fingertip collision.",
    )
    parser.add_argument(
        "--overlay",
        action="store_true",
        help="(Kept for compatibility — overlay is now the default.)",
    )
    parser.add_argument(
        "--hide-floor",
        action="store_true",
        help="Hide group 0. By default it is shown with the floor almost transparent so the support block remains visible.",
    )
    parser.add_argument(
        "--no-viewer",
        action="store_true",
        help="Only save PNG snapshots; do not open the interactive MuJoCo viewer.",
    )
    parser.add_argument(
        "--simulate",
        action="store_true",
        help="Step physics in the viewer so actuator ctrl sliders drive the hand. Without this, qpos sliders are still forwarded for kinematic inspection.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    xml_path, selected_model = choose_xml(args.model, args.xml)
    if not xml_path.exists():
        print(f"[error] XML not found: {xml_path}", file=sys.stderr)
        return 2

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    _apply_pose(model, data, args.pose, args.keyframe)
    collision_only = args.collision_only and not args.overlay
    counts = apply_overlay_colors(model, collision_only=collision_only)
    mujoco.mj_forward(model, data)

    opt = make_scene_option(
        show_sites=args.show_sites,
        show_contacts=args.show_contacts,
        show_floor=not args.hide_floor,
        collision_only=collision_only,
    )
    print_legend(counts, xml_path, args.pose, selected_model)

    saved_paths: list[Path] = []
    for camera_name in args.save_cameras:
        if not camera_name:
            continue
        try:
            saved_paths.append(
                save_screenshot(
                    model,
                    data,
                    args.out_dir,
                    camera_name,
                    opt,
                    args.width,
                    args.height,
                )
            )
        except Exception as exc:  # Offscreen rendering may fail on some display setups.
            print(f"[warn] failed to save camera {camera_name!r}: {exc}", file=sys.stderr)

    if saved_paths:
        print("\n已保存截图:")
        for path in saved_paths:
            print(f"  {path}")

    if args.no_viewer:
        return 0

    if not os.environ.get("DISPLAY") and not os.environ.get("WAYLAND_DISPLAY"):
        print("\n[warn] 未检测到 DISPLAY/WAYLAND_DISPLAY，跳过交互 viewer；已保存 PNG 可直接查看。")
        return 0

    try:
        launch_viewer(model, data, args.camera, opt, simulate=args.simulate)
    except Exception as exc:
        print(f"\n[warn] 交互 viewer 打开失败: {exc}", file=sys.stderr)
        print("       已保存 PNG；若在远程机器上运行，请确认 X11/Wayland 转发或本地桌面可用。")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
