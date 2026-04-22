"""Aero-Hand MuJoCo 场景查看器。

功能:
  - 加载灵巧手抓取场景并打开 MuJoCo viewer。
  - 支持三种显示模式:
      visual: 只看 STL 外观。
      collision: 只看真实物理碰撞体。
      overlay: 同时显示 STL 和碰撞体，便于对照检查。
  - 无图形界面时自动导出离屏预览图。

常用运行方式:
  cd /home/ll/SRTP/Aero-Hand

  # 默认打开 can 场景，只显示 STL 外观
  /home/ll/miniconda3/envs/aero_rl/bin/python view.py

  # 只看真实碰撞体
  /home/ll/miniconda3/envs/aero_rl/bin/python view.py --mode collision

  # STL + 碰撞体叠加显示
  /home/ll/miniconda3/envs/aero_rl/bin/python view.py --mode overlay

  # 指定场景 / 相机 / 显示 site
  /home/ll/miniconda3/envs/aero_rl/bin/python view.py --scene can_330ml --mode overlay --camera side --show-sites

注意:
  - 交互窗口里的键盘行为主要沿用 MuJoCo viewer 自带快捷键，例如 `I` 会切换 inertia 可视化，`L` 会切换 additive 渲染。
"""

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parent
XML_ROOT = (
    REPO_ROOT
    / "sim_rl"
    / "mujoco_playground"
    / "mujoco_playground"
    / "_src"
    / "manipulation"
    / "aero_hand"
    / "xmls"
)
PREVIEW_DIR = REPO_ROOT / "v2_iteration_docs" / "scene_previews"

WORLD_GROUP = 0
OBJECT_GROUP = 1
VISUAL_GROUP = 2
COLLISION_GROUP = 3
SITE_GROUP = 4

# 直接运行 `python view.py` 时默认打开 330ml sleek can 场景。
SCENE_PRESETS = {
    "cube_v2": XML_ROOT / "scene_mjx_grasp_v2_coacd.xml",
    "bottle_550ml": XML_ROOT / "scene_mjx_grasp_bottle_550ml.xml",
    "can_330ml": XML_ROOT / "scene_mjx_grasp_can_330ml.xml",
}
SCENE_NAME = "can_330ml"

# 仅用于可视化排查：将重力临时置零，方便检查初始相对位置。
ZERO_GRAVITY_FOR_VIEW = False
# 将执行器目标自动对齐到当前姿态，避免启动后瞬间弹动。
AUTO_EQUILIBRATE_CTRL = True
PREVIEW_IMAGE_SIZE = (1500, 500)
INTERACTIVE_STEP_PHYSICS = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Open Aero-Hand scenes in MuJoCo. By default this shows the STL "
            "appearance only; use --mode collision to inspect the real "
            "physical collision geoms."
        )
    )
    parser.add_argument(
        "--scene",
        choices=sorted(SCENE_PRESETS.keys()),
        default=SCENE_NAME,
        help="Scene preset to load.",
    )
    parser.add_argument(
        "--mode",
        choices=("visual", "collision", "overlay"),
        default="visual",
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
        help="Also show MuJoCo sites such as grasp references.",
    )
    return parser.parse_args()


def _select_xml_path(scene_name: str) -> Path:
    if scene_name not in SCENE_PRESETS:
        raise KeyError(
            f"未知场景 '{scene_name}'，可选: {sorted(SCENE_PRESETS.keys())}"
        )
    return SCENE_PRESETS[scene_name]


def _apply_home_keyframe(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if key_id < 0:
        return

    nq, nu = model.nq, model.nu
    data.qpos[:] = model.key_qpos[key_id * nq: (key_id + 1) * nq]
    if nu > 0:
        data.ctrl[:] = model.key_ctrl[key_id * nu: (key_id + 1) * nu]
    if model.nmocap > 0:
        start = key_id * model.nmocap
        end = (key_id + 1) * model.nmocap
        data.mocap_pos[:] = model.key_mpos[start:end]
        data.mocap_quat[:] = model.key_mquat[start:end]


def _equilibrate_ctrl(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    if not AUTO_EQUILIBRATE_CTRL or model.nu <= 0:
        return

    for i in range(model.nu):
        trn_type = int(model.actuator_trntype[i])
        trn_id = int(model.actuator_trnid[i, 0])
        if trn_type == 0 and trn_id >= 0:  # joint actuator
            qadr = model.jnt_qposadr[trn_id]
            data.ctrl[i] = data.qpos[qadr]
        elif trn_type == 3 and trn_id >= 0:  # tendon actuator
            data.ctrl[i] = data.ten_length[trn_id]


def _print_object_relative_pose(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "grasp_site")
    if site_id < 0:
        return

    grasp_site = data.site_xpos[site_id].copy()
    print("grasp_site:", grasp_site)

    for body_name in ("bottle", "cube"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        body_pos = data.xpos[body_id].copy()
        print(f"{body_name}_pos  :", body_pos)
        print(f"{body_name}-grasp:", body_pos - grasp_site)


def _has_interactive_display() -> bool:
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def _camera_id(model: mujoco.MjModel, camera_name: str) -> int:
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
    if cam_id >= 0:
        return cam_id

    available = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, idx)
        for idx in range(model.ncam)
    ]
    raise ValueError(f"camera {camera_name!r} not found; available: {available}")


def _mode_shows_group(display_mode: str, geom_group: int, show_sites: bool) -> bool:
    if geom_group == WORLD_GROUP:
        return True
    if geom_group == OBJECT_GROUP:
        return True
    if geom_group == VISUAL_GROUP:
        return display_mode in ("visual", "overlay")
    if geom_group == COLLISION_GROUP:
        return display_mode in ("collision", "overlay")
    if geom_group == SITE_GROUP:
        return show_sites
    return True


def _is_tip_collision(geom_name: str | None) -> bool:
    if not geom_name:
        return False
    return geom_name.startswith(
        (
            "right_index_distal_",
            "right_middle_distal_",
            "right_ring_distal_",
            "right_pinky_distal_",
            "right_thumb_tip_",
            "if_tip_",
            "mf_tip_",
            "rf_tip_",
            "pf_tip_",
            "th_tip_",
        )
    )


def _apply_display_mode_to_model(model: mujoco.MjModel, display_mode: str) -> None:
    for geom_id in range(model.ngeom):
        geom_group = int(model.geom_group[geom_id])
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        if geom_group != COLLISION_GROUP:
            continue

        if display_mode == "collision":
            rgba = np.array(
                [0.98, 0.85, 0.15, 1.0], dtype=np.float32
            ) if _is_tip_collision(geom_name) else np.array(
                [0.86, 0.20, 0.14, 1.0], dtype=np.float32
            )
        elif display_mode == "overlay":
            rgba = np.array(
                [0.98, 0.85, 0.15, 0.75], dtype=np.float32
            ) if _is_tip_collision(geom_name) else np.array(
                [0.86, 0.20, 0.14, 0.45], dtype=np.float32
            )
        else:
            continue

        model.geom_rgba[geom_id] = rgba


def _make_scene_option(display_mode: str, show_sites: bool) -> mujoco.MjvOption:
    opt = mujoco.MjvOption()
    opt.geomgroup[:] = 1
    opt.geomgroup[VISUAL_GROUP] = int(display_mode in ("visual", "overlay"))
    opt.geomgroup[COLLISION_GROUP] = int(display_mode in ("collision", "overlay"))
    opt.geomgroup[SITE_GROUP] = int(show_sites)
    return opt


def _geom_color(
    body_name: str,
    geom_name: str | None,
    geom_type: int,
    geom_group: int,
    display_mode: str,
) -> tuple[str, float]:
    body_name = body_name or ""
    geom_name = geom_name or ""
    if geom_group == COLLISION_GROUP:
        if _is_tip_collision(geom_name):
            return "#facc15", 0.86 if display_mode == "collision" else 0.55
        return "#dc2626", 0.80 if display_mode == "collision" else 0.42
    if body_name == "world" or geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return "#94a3b8", 0.12
    if "bottle_label" in geom_name:
        return "#2563eb", 0.86
    if "bottle_cap" in geom_name:
        return "#f59e0b", 0.92
    if "bottle" in body_name or "bottle" in geom_name:
        return "#8bd7f8", 0.74
    if "can_lid" in geom_name:
        return "#e5e7eb", 0.92
    if "can" in body_name or "can" in geom_name:
        return "#d1d5db", 0.80
    if "support" in body_name or "support" in geom_name:
        return "#8b5a3c", 0.60
    if "thumb" in body_name:
        return "#4f46e5", 0.78
    if "palm" in body_name:
        return "#1d4ed8", 0.78
    return "#3b82f6", 0.76


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm < 1e-9:
        return vec.copy()
    return vec / norm


def _build_view_basis(forward: tuple[float, float, float],
                      up_hint: tuple[float, float, float]) -> np.ndarray:
    forward_vec = _normalize(np.asarray(forward, dtype=float))
    up_vec = _normalize(np.asarray(up_hint, dtype=float))
    right_vec = _normalize(np.cross(up_vec, forward_vec))
    true_up_vec = _normalize(np.cross(forward_vec, right_vec))
    return np.vstack([right_vec, true_up_vec, forward_vec])


def _project_points(points: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = points @ basis[:2].T
    depth = points @ basis[2]
    return projected, depth


def _convex_hull(points: np.ndarray) -> np.ndarray:
    if len(points) <= 2:
        return points

    unique_points = np.unique(np.round(points, decimals=6), axis=0)
    if len(unique_points) <= 2:
        return unique_points

    sorted_points = sorted(map(tuple, unique_points.tolist()))

    def cross(o, a, b) -> float:
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower = []
    for point in sorted_points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0:
            lower.pop()
        lower.append(point)

    upper = []
    for point in reversed(sorted_points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0:
            upper.pop()
        upper.append(point)

    return np.asarray(lower[:-1] + upper[:-1], dtype=float)


def _sample_geom_surface(geom_type: int, size: np.ndarray) -> np.ndarray:
    box_type = int(mujoco.mjtGeom.mjGEOM_BOX)
    mesh_type = int(mujoco.mjtGeom.mjGEOM_MESH)
    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    sphere_type = int(mujoco.mjtGeom.mjGEOM_SPHERE)
    capsule_type = int(mujoco.mjtGeom.mjGEOM_CAPSULE)
    cylinder_type = int(mujoco.mjtGeom.mjGEOM_CYLINDER)

    if geom_type in (box_type, mesh_type):
        extents = np.maximum(np.asarray(size[:3], dtype=float), 1e-4)
        signs = np.array(
            [
                [-1, -1, -1],
                [1, -1, -1],
                [1, 1, -1],
                [-1, 1, -1],
                [-1, -1, 1],
                [1, -1, 1],
                [1, 1, 1],
                [-1, 1, 1],
            ],
            dtype=float,
        )
        return signs * extents

    if geom_type == plane_type:
        return np.array(
            [
                [-0.25, -0.25, 0.0],
                [0.25, -0.25, 0.0],
                [0.25, 0.25, 0.0],
                [-0.25, 0.25, 0.0],
            ],
            dtype=float,
        )

    if geom_type == sphere_type:
        radius = max(float(size[0]), 1e-4)
        theta = np.linspace(0.0, 2.0 * np.pi, 18, endpoint=False)
        phi = np.linspace(0.0, np.pi, 9)
        points = []
        for phi_value in phi:
            sin_phi = np.sin(phi_value)
            cos_phi = np.cos(phi_value)
            for theta_value in theta:
                points.append(
                    [
                        radius * sin_phi * np.cos(theta_value),
                        radius * sin_phi * np.sin(theta_value),
                        radius * cos_phi,
                    ]
                )
        return np.asarray(points, dtype=float)

    if geom_type == cylinder_type:
        radius = max(float(size[0]), 1e-4)
        half_length = max(float(size[1]), 1e-4)
        theta = np.linspace(0.0, 2.0 * np.pi, 28, endpoint=False)
        points = []
        for z_coord in (-half_length, half_length):
            points.append([0.0, 0.0, z_coord])
            for theta_value in theta:
                points.append(
                    [
                        radius * np.cos(theta_value),
                        radius * np.sin(theta_value),
                        z_coord,
                    ]
                )
        return np.asarray(points, dtype=float)

    if geom_type == capsule_type:
        radius = max(float(size[0]), 1e-4)
        half_length = max(float(size[1]), 1e-4)
        theta = np.linspace(0.0, 2.0 * np.pi, 24, endpoint=False)
        arc = np.linspace(0.0, np.pi / 2.0, 7)
        points = []

        for z_coord in (-half_length, half_length):
            for theta_value in theta:
                points.append(
                    [
                        radius * np.cos(theta_value),
                        radius * np.sin(theta_value),
                        z_coord,
                    ]
                )

        for arc_value in arc:
            ring_radius = radius * np.cos(arc_value)
            height = radius * np.sin(arc_value)
            for theta_value in theta:
                cos_theta = np.cos(theta_value)
                sin_theta = np.sin(theta_value)
                points.append([ring_radius * cos_theta, ring_radius * sin_theta, half_length + height])
                points.append([ring_radius * cos_theta, ring_radius * sin_theta, -half_length - height])

        return np.asarray(points, dtype=float)

    extents = np.maximum(np.asarray(size[:3], dtype=float), 1e-4)
    return np.array(
        [
            [-extents[0], -extents[1], -extents[2]],
            [extents[0], -extents[1], -extents[2]],
            [extents[0], extents[1], extents[2]],
            [-extents[0], extents[1], extents[2]],
        ],
        dtype=float,
    )


def _free_camera_focus_bounds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    display_mode: str,
    show_sites: bool,
) -> tuple[np.ndarray, np.ndarray]:
    plane_type = int(mujoco.mjtGeom.mjGEOM_PLANE)
    points = []

    for geom_id in range(model.ngeom):
        geom_group = int(model.geom_group[geom_id])
        if not _mode_shows_group(display_mode, geom_group, show_sites):
            continue
        if float(model.geom_rgba[geom_id, 3]) <= 0.01:
            continue

        geom_type = int(model.geom_type[geom_id])
        if geom_type == plane_type:
            continue

        center = np.array(data.geom_xpos[geom_id], dtype=float)
        rotation = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        local_points = _sample_geom_surface(
            geom_type,
            np.array(model.geom_size[geom_id], dtype=float),
        )
        points.append(center + local_points @ rotation.T)

    if not points:
        fallback_center = np.array(model.stat.center, dtype=float)
        fallback_span = np.full(3, max(float(model.stat.extent), 0.18), dtype=float)
        return fallback_center, fallback_span

    cloud = np.vstack(points)
    mins = np.min(cloud, axis=0)
    maxs = np.max(cloud, axis=0)
    return 0.5 * (mins + maxs), maxs - mins


def _configure_free_camera(
    camera: mujoco.MjvCamera,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    display_mode: str,
    show_sites: bool,
) -> None:
    center, span = _free_camera_focus_bounds(model, data, display_mode, show_sites)
    radius = max(0.5 * float(np.linalg.norm(span)), 0.12)

    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = center
    camera.distance = max(radius * 2.7, 0.38)
    camera.azimuth = 120
    camera.elevation = -20


def _save_headless_preview(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    scene_name: str,
    display_mode: str,
    show_sites: bool,
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREVIEW_DIR / f"{scene_name}_{display_mode}_preview.png"

    fig, axes = plt.subplots(
        1,
        3,
        figsize=(PREVIEW_IMAGE_SIZE[0] / 100, PREVIEW_IMAGE_SIZE[1] / 100),
        facecolor="#f8fafc",
    )
    view_specs = [
        ("Perspective", _build_view_basis((1.0, -1.4, 0.8), (0.0, 0.0, 1.0))),
        ("Top", _build_view_basis((0.0, 0.0, -1.0), (0.0, 1.0, 0.0))),
        ("Side", _build_view_basis((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
    ]

    site_markers = []
    if show_sites:
        for site_name, color in (
            ("grasp_site", "#16a34a"),
            ("bottle_grasp_band", "#dc2626"),
            ("can_grasp_band", "#dc2626"),
        ):
            site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                site_markers.append((site_name, color, np.array(data.site_xpos[site_id], dtype=float)))

    renderables = []
    for geom_id in range(model.ngeom):
        geom_group = int(model.geom_group[geom_id])
        if not _mode_shows_group(display_mode, geom_group, show_sites):
            continue

        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
        geom_type = int(model.geom_type[geom_id])
        center = np.array(data.geom_xpos[geom_id], dtype=float)
        rotation = np.array(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        local_points = _sample_geom_surface(
            geom_type,
            np.array(model.geom_size[geom_id], dtype=float),
        )
        world_points = center + local_points @ rotation.T
        color, alpha = _geom_color(
            body_name, geom_name, geom_type, geom_group, display_mode
        )
        renderables.append(
            {
                "body_name": body_name,
                "geom_name": geom_name,
                "geom_type": geom_type,
                "points": world_points,
                "color": color,
                "alpha": alpha,
            }
        )

    for ax, (title, basis) in zip(axes, view_specs):
        ax.set_facecolor("#eef2ff")
        projected_cloud = []
        projected_items = []

        for item in renderables:
            projected_points, depth = _project_points(item["points"], basis)
            hull = _convex_hull(projected_points)
            if len(hull) == 0:
                continue
            projected_cloud.append(projected_points)
            projected_items.append(
                {
                    "polygon": hull,
                    "depth": float(np.mean(depth)),
                    "color": item["color"],
                    "alpha": item["alpha"],
                }
            )

        for _, _, site_point in site_markers:
            projected_point, _ = _project_points(site_point[None, :], basis)
            projected_cloud.append(projected_point)

        projected_items.sort(key=lambda item: item["depth"])

        for item in projected_items:
            patch = Polygon(
                item["polygon"],
                closed=True,
                facecolor=item["color"],
                edgecolor="#0f172a",
                linewidth=0.8,
                alpha=item["alpha"],
                joinstyle="round",
            )
            ax.add_patch(patch)

        for site_name, color, site_point in site_markers:
            projected_point, _ = _project_points(site_point[None, :], basis)
            x_coord, y_coord = projected_point[0]
            ax.scatter(x_coord, y_coord, s=36, c=color, edgecolors="white", linewidths=0.6, zorder=10)
            ax.text(
                x_coord + 0.006,
                y_coord + 0.006,
                site_name,
                color=color,
                fontsize=8,
                weight="bold",
                zorder=11,
            )

        if projected_cloud:
            cloud = np.vstack(projected_cloud)
            mins = cloud.min(axis=0)
            maxs = cloud.max(axis=0)
            center = 0.5 * (mins + maxs)
            span = max(maxs - mins)
            span = max(span, 0.18)
            pad = 0.12 * span
            ax.set_xlim(center[0] - 0.52 * span - pad, center[0] + 0.52 * span + pad)
            ax.set_ylim(center[1] - 0.52 * span - pad, center[1] + 0.52 * span + pad)

        ax.set_aspect("equal")
        ax.set_title(title, fontsize=12, weight="bold")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    fig.suptitle(
        f"{scene_name} scene preview ({display_mode})",
        fontsize=14,
        weight="bold",
        y=0.98,
    )
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def _launch_viewer(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    display_mode: str,
    camera_name: str,
    show_sites: bool,
) -> None:
    from mujoco import viewer as mujoco_viewer

    opt = _make_scene_option(display_mode, show_sites)
    with mujoco_viewer.launch_passive(
        model, data, show_left_ui=True, show_right_ui=True
    ) as viewer:
        with viewer.lock():
            viewer.opt.geomgroup[:] = opt.geomgroup
            viewer.opt.flags[:] = opt.flags
            if camera_name == "free":
                _configure_free_camera(
                    viewer.cam,
                    model,
                    data,
                    display_mode=display_mode,
                    show_sites=show_sites,
                )
            else:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = _camera_id(model, camera_name)

        while viewer.is_running():
            if INTERACTIVE_STEP_PHYSICS:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1.0 / 60.0)


def main() -> None:
    args = parse_args()
    xml_path = _select_xml_path(args.scene)
    if not xml_path.exists():
        print(f"❌ 找不到文件，请检查路径：\n{xml_path}")
        return

    os.chdir(xml_path.parent)
    print(f"⏳ 正在加载场景: {args.scene}")
    print(f"📄 XML: {xml_path.name}")
    print(f"🎛️ 显示模式: {args.mode}")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path.name)
        _apply_display_mode_to_model(model, args.mode)
        if ZERO_GRAVITY_FOR_VIEW:
            model.opt.gravity[:] = 0.0
            print("🔧 已启用零重力可视化模式: gravity =", model.opt.gravity)

        data = mujoco.MjData(model)
        _apply_home_keyframe(model, data)
        mujoco.mj_forward(model, data)
        _equilibrate_ctrl(model, data)
        mujoco.mj_forward(model, data)

        if AUTO_EQUILIBRATE_CTRL:
            print("🔧 已启用控制目标平衡模式，初始状态应保持静态")

        _print_object_relative_pose(model, data)
        if _has_interactive_display():
            print("✅ 加载成功！检测到图形环境，正在打开 MuJoCo 窗口")
            _launch_viewer(
                model,
                data,
                display_mode=args.mode,
                camera_name=args.camera,
                show_sites=args.show_sites,
            )
            return

        preview_path = _save_headless_preview(
            model,
            data,
            args.scene,
            display_mode=args.mode,
            show_sites=args.show_sites,
        )
        print("✅ 加载成功！当前无 DISPLAY，已自动切换到离屏预览模式")
        print(f"🖼️ 预览图已保存到: {preview_path}")
    except Exception as exc:
        print(f"\n❌ MuJoCo 引擎报错了: {exc}")
        print("💡 提示：如果报 mesh/assets 路径错误，优先检查 XML 中的 include 与 meshdir。")


if __name__ == "__main__":
    main()
