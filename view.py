import os
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

# 直接运行 `python view.py` 时默认打开新的 550ml 空瓶场景。
SCENE_PRESETS = {
    "cube_v2": XML_ROOT / "scene_mjx_grasp_v2_coacd.xml",
    "bottle_550ml": XML_ROOT / "scene_mjx_grasp_bottle_550ml.xml",
}
SCENE_NAME = "bottle_550ml"

# 仅用于可视化排查：将重力临时置零，方便检查初始相对位置。
ZERO_GRAVITY_FOR_VIEW = False
# 将执行器目标自动对齐到当前姿态，避免启动后瞬间弹动。
AUTO_EQUILIBRATE_CTRL = True
PREVIEW_IMAGE_SIZE = (1500, 500)


def _select_xml_path() -> Path:
    if SCENE_NAME not in SCENE_PRESETS:
        raise KeyError(
            f"未知场景 '{SCENE_NAME}'，可选: {sorted(SCENE_PRESETS.keys())}"
        )
    return SCENE_PRESETS[SCENE_NAME]


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


def _geom_color(body_name: str, geom_name: str | None, geom_type: int) -> tuple[str, float]:
    body_name = body_name or ""
    geom_name = geom_name or ""
    if body_name == "world" or geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        return "#94a3b8", 0.12
    if "bottle_label" in geom_name:
        return "#2563eb", 0.86
    if "bottle_cap" in geom_name:
        return "#f59e0b", 0.92
    if "bottle" in body_name or "bottle" in geom_name:
        return "#8bd7f8", 0.74
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


def _save_headless_preview(
    model: mujoco.MjModel, data: mujoco.MjData, scene_name: str
) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon

    PREVIEW_DIR.mkdir(parents=True, exist_ok=True)
    output_path = PREVIEW_DIR / f"{scene_name}_preview.png"

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
    for site_name, color in (("grasp_site", "#16a34a"), ("bottle_grasp_band", "#dc2626")):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
        if site_id >= 0:
            site_markers.append((site_name, color, np.array(data.site_xpos[site_id], dtype=float)))

    renderables = []
    for geom_id in range(model.ngeom):
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
        color, alpha = _geom_color(body_name, geom_name, geom_type)
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

    fig.suptitle(f"{scene_name} scene preview", fontsize=14, weight="bold", y=0.98)
    plt.tight_layout(rect=[0.0, 0.0, 1.0, 0.95])
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return output_path


def main() -> None:
    xml_path = _select_xml_path()
    if not xml_path.exists():
        print(f"❌ 找不到文件，请检查路径：\n{xml_path}")
        return

    os.chdir(xml_path.parent)
    print(f"⏳ 正在加载场景: {SCENE_NAME}")
    print(f"📄 XML: {xml_path.name}")

    try:
        model = mujoco.MjModel.from_xml_path(xml_path.name)
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
            from mujoco import viewer as mujoco_viewer

            print("✅ 加载成功！检测到图形环境，正在打开 MuJoCo 窗口")
            mujoco_viewer.launch(model, data)
            return

        preview_path = _save_headless_preview(model, data, SCENE_NAME)
        print("✅ 加载成功！当前无 DISPLAY，已自动切换到离屏预览模式")
        print(f"🖼️ 预览图已保存到: {preview_path}")
    except Exception as exc:
        print(f"\n❌ MuJoCo 引擎报错了: {exc}")
        print("💡 提示：如果报 mesh/assets 路径错误，优先检查 XML 中的 include 与 meshdir。")


if __name__ == "__main__":
    main()
