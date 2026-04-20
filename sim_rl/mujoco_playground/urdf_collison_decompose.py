"""Convex-decompose STL collision meshes for URDF or the V2 MuJoCo hand.

The original version of this script was URDF-only and required optional
visualization packages at import time.  For the Aero-Hand V2 workflow we also
need a MuJoCo XML variant whose physical collision geoms are high-fidelity
convex mesh pieces generated from the same STL files used for visualization.

Typical usage from the repository root:

  /home/ll/miniconda3/envs/aero_rl/bin/python \
    sim_rl/mujoco_playground/urdf_collison_decompose.py --v2-mjcf

This creates:
  - handinformation/URDF/右/qbr/meshes/coacd_v2/*.stl
  - sim_rl/.../xmls/right_hand_v2_vertical_coacd.xml
  - sim_rl/.../xmls/scene_mjx_grasp_v2_coacd.xml
  - sim_rl/.../xmls/right_hand_v2_vertical_coacd_report.txt
"""

from __future__ import annotations

import argparse
import copy
import os
from pathlib import Path
import sys
import xml.etree.ElementTree as ET

from scipy.spatial import ConvexHull as _ScipyConvexHull

import numpy as np
import trimesh

try:
    import coacd
except ImportError as exc:  # pragma: no cover - depends on local environment.
    coacd = None
    COACD_IMPORT_ERROR = exc
else:
    COACD_IMPORT_ERROR = None

try:
    import fast_simplification as _fast_simpl
except ImportError:
    _fast_simpl = None

try:
    from natsort import natsorted
except ImportError:  # pragma: no cover - fallback is deterministic enough here.
    natsorted = sorted


REPO_ROOT = Path(__file__).resolve().parents[2]
XML_DIR = (
    REPO_ROOT
    / "sim_rl/mujoco_playground/mujoco_playground/_src/manipulation/aero_hand/xmls"
)
V2_HAND_XML = XML_DIR / "right_hand_v2_vertical.xml"
V2_SCENE_XML = XML_DIR / "scene_mjx_grasp_v2.xml"
V2_COACD_HAND_XML = XML_DIR / "right_hand_v2_vertical_coacd.xml"
V2_COACD_SCENE_XML = XML_DIR / "scene_mjx_grasp_v2_coacd.xml"
V2_COACD_REPORT = XML_DIR / "right_hand_v2_vertical_coacd_report.txt"
V2_MESH_DIR = REPO_ROOT / "handinformation/URDF/右/qbr/meshes"
V2_COACD_DIR = V2_MESH_DIR / "coacd_v2"

# Per-body CoACD parameters for the "lite" MJX-friendly model.
# Critical contact surfaces (fingertips, thumb tip) get more hulls;
# structural parts (proximal segments, thumb base/mid) get fewer.
# Target: ~38 total meshes (down from 342).
LITE_BODY_PARAMS: dict[str, dict[str, float | int]] = {
    "v2_palm":             {"threshold": 0.08, "max_convex_hull": 8},
    "v2_index_distal":     {"threshold": 0.04, "max_convex_hull": 4, "max_verts": 64},
    "v2_middle_distal":    {"threshold": 0.06, "max_convex_hull": 4, "max_verts": 64},
    "v2_ring_distal":      {"threshold": 0.08, "max_convex_hull": 3, "max_verts": 64},
    "v2_pinky_distal":     {"threshold": 0.08, "max_convex_hull": 3, "max_verts": 64},
    "v2_thumb_tip":        {"threshold": 0.06, "max_convex_hull": 5, "max_verts": 64},
    "v2_index_proximal":   {"threshold": 0.12, "max_convex_hull": 2},
    "v2_middle_proximal":  {"threshold": 0.12, "max_convex_hull": 2},
    "v2_ring_proximal":    {"threshold": 0.12, "max_convex_hull": 2},
    "v2_pinky_proximal":   {"threshold": 0.12, "max_convex_hull": 2},
    "v2_thumb_base":       {"threshold": 0.10, "max_convex_hull": 2},
    "v2_thumb_mid":        {"threshold": 0.15, "max_convex_hull": 1},
}


# ---------------------------------------------------------------------------
# Fitted-primitive helpers
# ---------------------------------------------------------------------------

def _rotation_matrix_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a MuJoCo quaternion (w, x, y, z)."""
    # Shepperd's method
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def _compute_pca_axes(verts: np.ndarray) -> np.ndarray:
    """Compute PCA eigenvectors (3x3 matrix, columns = axes, descending eigenvalue order).

    Returns a right-handed rotation matrix that maps PCA local frame → world frame.
    """
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    cov = centered.T @ centered
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvecs = eigvecs[:, order]
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] = -eigvecs[:, 2]
    return eigvecs


def fit_primitive_to_convex_hull(
    stl_path: Path,
    *,
    reference_axes: np.ndarray | None = None,
) -> dict:
    """Fit an oriented bounding box to a CoACD convex hull STL.

    If *reference_axes* is provided (3×3 rotation matrix, typically from the
    parent visual mesh's PCA), those axes are used for orientation instead of
    computing PCA on the (often too-few) hull vertices.  This dramatically
    improves alignment because the original visual STL has thousands of
    vertices while a simplified CoACD hull may have only 12.

    Returns a dict with keys:
      type: always "box" (capsules are disabled — they overshoot the mesh)
      size: MuJoCo size string (half-extents)
      pos: "x y z" position string relative to parent body
      quat: "w x y z" orientation quaternion string
    """
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    verts = mesh.vertices
    if verts.shape[0] < 3:
        center = verts.mean(axis=0)
        return {
            "type": "sphere",
            "size": "0.001",
            "pos": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
            "quat": "1 0 0 0",
        }

    # Use reference axes from the parent visual mesh when available;
    # fall back to hull-local PCA only as a last resort.
    if reference_axes is not None:
        eigvecs = reference_axes
    else:
        eigvecs = _compute_pca_axes(verts)

    # Project hull vertices onto the reference axes to get the OBB.
    centroid = verts.mean(axis=0)
    centered = verts - centroid
    projected = centered @ eigvecs  # (N, 3)
    lo = projected.min(axis=0)
    hi = projected.max(axis=0)
    half_extents = (hi - lo) / 2.0
    obb_center_pca = (hi + lo) / 2.0
    obb_center_world = centroid + eigvecs @ obb_center_pca

    quat = _rotation_matrix_to_quat(eigvecs)
    return {
        "type": "box",
        "size": f"{half_extents[0]:.6f} {half_extents[1]:.6f} {half_extents[2]:.6f}",
        "pos": f"{obb_center_world[0]:.6f} {obb_center_world[1]:.6f} {obb_center_world[2]:.6f}",
        "quat": f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
    }


# Per-body capsule overrides.  Keys are visual mesh names (v2_xxx).
# "axis_hint" forces the capsule axis: 0=PCA-axis-0 (longest), 1=axis-1, 2=axis-2.
# "radius_scale" multiplies the fitted radius (>1 enlarges, <1 shrinks).
# Bodies not listed use default axis=0 (longest PCA axis) and radius_scale=1.0.
CAPSULE_BODY_PARAMS: dict[str, dict[str, float | int]] = {
    "v2_palm": {"use_boxes": True},  # palm is flat, keep boxes
    "v2_thumb_base": {"use_boxes": True},  # thumb base is boxy
    "v2_index_distal": {"n_capsules": 2},
    "v2_middle_distal": {"n_capsules": 2},
    "v2_ring_distal": {"n_capsules": 2},
    "v2_pinky_distal": {"n_capsules": 2},
    "v2_thumb_tip": {"n_capsules": 2},
}


def fit_capsule_to_visual_mesh(
    stl_path: Path,
    *,
    axis_hint: int = 0,
    radius_scale: float = 1.0,
) -> dict:
    """Fit a single capsule to a visual mesh STL.

    Computes PCA on the full visual mesh (thousands of vertices), uses the
    longest axis as the capsule axis, and fits the capsule radius from radial
    distances of all vertices to that axis.

    Returns a dict with keys: type, fromto, size (radius).
    """
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    verts = mesh.vertices
    if verts.shape[0] < 3:
        center = verts.mean(axis=0)
        return {
            "type": "sphere",
            "size": "0.002",
            "pos": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
            "quat": "1 0 0 0",
        }

    eigvecs = _compute_pca_axes(verts)
    # Select capsule axis (default: longest PCA axis = column 0)
    axis = eigvecs[:, axis_hint]

    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # Project onto capsule axis → endpoints
    projections = centered @ axis  # (N,)
    t_min = projections.min()
    t_max = projections.max()

    # Compute radial distances (perpendicular to capsule axis)
    along_axis = np.outer(projections, axis)  # (N, 3)
    radial_vecs = centered - along_axis
    radial_dists = np.linalg.norm(radial_vecs, axis=1)

    # Use 90th percentile of radial distances as capsule radius
    # (avoids outlier vertices inflating the capsule)
    radius = float(np.percentile(radial_dists, 90)) * radius_scale

    # Shrink endpoints inward by radius (capsule hemispherical caps extend beyond)
    t_min_shrunk = t_min + radius
    t_max_shrunk = t_max - radius
    if t_min_shrunk >= t_max_shrunk:
        # Very short segment: use a sphere-like capsule
        t_min_shrunk = (t_min + t_max) / 2 - 0.001
        t_max_shrunk = (t_min + t_max) / 2 + 0.001

    p0 = centroid + t_min_shrunk * axis
    p1 = centroid + t_max_shrunk * axis

    fromto_str = (
        f"{p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f} "
        f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}"
    )
    return {
        "type": "capsule",
        "fromto": fromto_str,
        "size": f"{radius:.6f}",
    }


def fit_ellipsoids_to_visual_mesh(
    stl_path: Path,
    *,
    n_ellipsoids: int = 2,
) -> list[dict]:
    """Split a visual mesh along its longest PCA axis and fit an ellipsoid to each half.

    Returns a list of n_ellipsoids dicts, each with keys:
      type: "ellipsoid"
      size: "rx ry rz" (semi-axes)
      pos: "x y z"
      quat: "w x y z"
    """
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    verts = mesh.vertices
    if verts.shape[0] < 4:
        center = verts.mean(axis=0)
        return [{
            "type": "sphere",
            "size": "0.002",
            "pos": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
            "quat": "1 0 0 0",
        }]

    eigvecs = _compute_pca_axes(verts)
    centroid = verts.mean(axis=0)
    centered = verts - centroid

    # Project onto PCA axes
    projected = centered @ eigvecs  # (N, 3)

    # Split along longest axis (axis 0) into n_ellipsoids segments
    t_min = projected[:, 0].min()
    t_max = projected[:, 0].max()
    boundaries = np.linspace(t_min, t_max, n_ellipsoids + 1)

    results = []
    for seg_idx in range(n_ellipsoids):
        lo = boundaries[seg_idx]
        hi = boundaries[seg_idx + 1]
        mask = (projected[:, 0] >= lo) & (projected[:, 0] <= hi)
        if mask.sum() < 4:
            # Extend to include nearest vertices
            dists = np.minimum(np.abs(projected[:, 0] - lo), np.abs(projected[:, 0] - hi))
            mask = dists <= np.sort(dists)[min(10, len(dists)-1)]

        seg_verts_pca = projected[mask]
        seg_verts_world = verts[mask]

        # Fit ellipsoid to this segment: semi-axes from extent along each PCA axis
        seg_lo = seg_verts_pca.min(axis=0)
        seg_hi = seg_verts_pca.max(axis=0)
        semi_axes = (seg_hi - seg_lo) / 2.0

        # Ensure minimum size
        semi_axes = np.maximum(semi_axes, 0.001)

        # Center in PCA space → world
        center_pca = (seg_hi + seg_lo) / 2.0
        center_world = centroid + eigvecs @ center_pca

        quat = _rotation_matrix_to_quat(eigvecs)
        results.append({
            "type": "ellipsoid",
            "size": f"{semi_axes[0]:.6f} {semi_axes[1]:.6f} {semi_axes[2]:.6f}",
            "pos": f"{center_world[0]:.6f} {center_world[1]:.6f} {center_world[2]:.6f}",
            "quat": f"{quat[0]:.6f} {quat[1]:.6f} {quat[2]:.6f} {quat[3]:.6f}",
        })

    return results


def fit_multi_capsules_to_visual_mesh(
    stl_path: Path,
    *,
    n_capsules: int = 2,
) -> list[dict]:
    """Split a visual mesh along its longest PCA axis and fit a capsule to each segment.

    Each segment gets a capsule oriented along the original bone axis, with
    radius fitted from the radial distribution of that segment's vertices.
    This better approximates curved finger shapes than a single capsule.

    Returns a list of n_capsules dicts, each with keys:
      type: "capsule"
      fromto: "x0 y0 z0 x1 y1 z1"
      size: "radius"
    """
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    verts = mesh.vertices
    if verts.shape[0] < 4:
        center = verts.mean(axis=0)
        return [{
            "type": "sphere",
            "size": "0.002",
            "pos": f"{center[0]:.6f} {center[1]:.6f} {center[2]:.6f}",
            "quat": "1 0 0 0",
        }]

    eigvecs = _compute_pca_axes(verts)
    axis = eigvecs[:, 0]  # longest PCA axis = bone direction

    centroid = verts.mean(axis=0)
    centered = verts - centroid
    projections = centered @ axis  # (N,)

    t_min = projections.min()
    t_max = projections.max()
    boundaries = np.linspace(t_min, t_max, n_capsules + 1)

    results = []
    for seg_idx in range(n_capsules):
        lo = boundaries[seg_idx]
        hi = boundaries[seg_idx + 1]
        mask = (projections >= lo) & (projections <= hi)
        if mask.sum() < 4:
            dists = np.minimum(np.abs(projections - lo), np.abs(projections - hi))
            mask = dists <= np.sort(dists)[min(10, len(dists) - 1)]

        seg_projs = projections[mask]
        seg_centered = centered[mask]

        # Radial distances for this segment
        along_axis = np.outer(seg_projs, axis)
        radial_vecs = seg_centered - along_axis
        radial_dists = np.linalg.norm(radial_vecs, axis=1)
        radius = float(np.percentile(radial_dists, 90))
        radius = max(radius, 0.001)

        # Capsule endpoints for this segment
        seg_t_min = float(seg_projs.min())
        seg_t_max = float(seg_projs.max())

        # Shrink by radius for hemispherical caps
        t0 = seg_t_min + radius
        t1 = seg_t_max - radius
        if t0 >= t1:
            mid = (seg_t_min + seg_t_max) / 2
            t0 = mid - 0.001
            t1 = mid + 0.001

        p0 = centroid + t0 * axis
        p1 = centroid + t1 * axis

        fromto_str = (
            f"{p0[0]:.6f} {p0[1]:.6f} {p0[2]:.6f} "
            f"{p1[0]:.6f} {p1[1]:.6f} {p1[2]:.6f}"
        )
        results.append({
            "type": "capsule",
            "fromto": fromto_str,
            "size": f"{radius:.6f}",
        })

    return results


def compute_capsule_fit_quality(
    stl_path: Path,
    capsule_params: list[dict],
) -> dict:
    """Compute how well capsule(s) approximate the original STL mesh surface.

    For each mesh vertex, compute the minimum distance to any capsule surface.
    Returns statistics for ALL vertices and for the FRONT/PAD side only
    (vertices with negative PCA-axis-1 component, i.e., the concave/gripping side).

    Returns dict with keys:
      all_mean, all_max, all_p95: distance stats for all vertices (mm)
      pad_mean, pad_max, pad_p95: distance stats for pad-side vertices (mm)
      coverage_1mm: fraction of pad vertices within 1mm of a capsule surface
    """
    mesh = trimesh.load(stl_path, force="mesh", process=False)
    verts = mesh.vertices
    if len(verts) < 4 or not capsule_params:
        return {}

    eigvecs = _compute_pca_axes(verts)
    centroid = verts.mean(axis=0)
    projected = (verts - centroid) @ eigvecs  # (N, 3) in PCA space

    # Compute distance from each vertex to nearest capsule surface
    min_dists = np.full(len(verts), np.inf)
    for cap in capsule_params:
        if cap["type"] != "capsule" or "fromto" not in cap:
            continue
        ft = list(map(float, cap["fromto"].split()))
        p0 = np.array(ft[:3])
        p1 = np.array(ft[3:])
        radius = float(cap["size"])

        # Distance from each vertex to the capsule axis segment
        axis_vec = p1 - p0
        axis_len = np.linalg.norm(axis_vec)
        if axis_len < 1e-8:
            dists = np.linalg.norm(verts - p0, axis=1) - radius
        else:
            axis_unit = axis_vec / axis_len
            v_p0 = verts - p0
            t = np.clip(v_p0 @ axis_unit, 0.0, axis_len)
            closest = p0 + np.outer(t, axis_unit)
            dists = np.linalg.norm(verts - closest, axis=1) - radius

        min_dists = np.minimum(min_dists, np.abs(dists))

    min_dists_mm = min_dists * 1000.0

    # Pad side: vertices with negative PCA-axis-1 projection (concave/inner side)
    pad_mask = projected[:, 1] < 0
    if pad_mask.sum() < 10:
        pad_mask = np.ones(len(verts), dtype=bool)

    pad_dists_mm = min_dists_mm[pad_mask]

    return {
        "all_mean": float(np.mean(min_dists_mm)),
        "all_max": float(np.max(min_dists_mm)),
        "all_p95": float(np.percentile(min_dists_mm, 95)),
        "pad_mean": float(np.mean(pad_dists_mm)),
        "pad_max": float(np.max(pad_dists_mm)),
        "pad_p95": float(np.percentile(pad_dists_mm, 95)),
        "coverage_1mm": float(np.mean(pad_dists_mm < 1.0)),
    }


def create_capsule_mjcf(
    *,
    hand_xml_in: Path,
    scene_xml_in: Path,
    hand_xml_out: Path,
    scene_xml_out: Path,
    report_out: Path,
) -> None:
    """Generate V2 MuJoCo XML with capsule collision bodies.

    Each finger segment gets a single capsule fitted from the visual STL mesh.
    The palm and thumb base keep their existing box collision geoms (multiple
    boxes for the complex palm shape).  All collision is primitive-only, making
    MJX JIT compilation instant.
    """
    tree = ET.parse(hand_xml_in)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError(f"{hand_xml_in} has no <asset> section")

    mesh_dir = find_compiler_meshdir(root)
    mesh_name_to_file = mesh_assets(root)

    # For bodies with use_boxes=True, we need the CoACD STLs for fitted boxes
    coacd_mesh_dir = Path(V2_COACD_DIR)

    report_lines = [
        "V2 Capsule collision model report",
        f"source hand xml: {hand_xml_in}",
        f"output hand xml: {hand_xml_out}",
        "",
    ]

    total_prims = 0
    visual_mesh_count = 0
    old_collision_count = 0
    type_counts: dict[str, int] = {}

    for parent in root.iter():
        children = list(parent)
        visual_insertions: list[tuple[int, ET.Element, str]] = []
        for idx, child in enumerate(children):
            if is_old_collision_geom(child):
                parent.remove(child)
                old_collision_count += 1
            elif is_visual_mesh_geom(child):
                visual_insertions.append((idx, child, child.get("mesh", "")))

        insert_after_offset = 0
        parent_name = parent.get("name", "body")
        for original_idx, visual_geom, visual_mesh_name in visual_insertions:
            mesh_file = mesh_name_to_file.get(visual_mesh_name, "?")
            params = CAPSULE_BODY_PARAMS.get(visual_mesh_name, {})

            visual_mesh_count += 1
            insert_idx = original_idx + 1 + insert_after_offset

            if params.get("use_boxes"):
                # Use fitted boxes from CoACD hulls (for palm, thumb base)
                prefix = f"{visual_mesh_name}_coacd"
                stl_files = natsorted(coacd_mesh_dir.glob(f"{prefix}_*.stl"))
                if not stl_files:
                    report_lines.append(
                        f"{visual_mesh_name:22s} body={parent_name:28s} "
                        f"NO CoACD STLs — skipped"
                    )
                    continue
                orig_mesh_path = resolve_mesh_path(mesh_file, mesh_dir)
                reference_axes = None
                if orig_mesh_path.exists():
                    orig_mesh = trimesh.load(orig_mesh_path, force="mesh", process=False)
                    reference_axes = _compute_pca_axes(orig_mesh.vertices)

                part_types: list[str] = []
                for part_idx, stl_path in enumerate(stl_files):
                    prim = fit_primitive_to_convex_hull(stl_path, reference_axes=reference_axes)
                    prim_type = prim["type"]
                    part_types.append(prim_type)
                    type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
                    geom_attrs = {
                        "name": f"{parent_name}_capsule_{part_idx:03d}",
                        "type": prim_type,
                        "size": prim["size"],
                        "pos": prim["pos"],
                        "quat": prim["quat"],
                        "group": "3",
                        "friction": "1.5",
                        "solref": "0.01 1.2",
                        "solimp": "0.95 0.995 0.0005",
                    }
                    total_prims += 1
                    geom = ET.Element("geom", geom_attrs)
                    parent.insert(insert_idx, geom)
                    insert_idx += 1
                    insert_after_offset += 1

                types_str = ", ".join(part_types)
                report_lines.append(
                    f"{visual_mesh_name:22s} body={parent_name:28s} "
                    f"parts={len(stl_files):3d} [BOX] types=[{types_str}] file={mesh_file}"
                )
            else:
                # Fit collision primitives from the original visual mesh
                orig_mesh_path = resolve_mesh_path(mesh_file, mesh_dir)
                if not orig_mesh_path.exists():
                    report_lines.append(
                        f"{visual_mesh_name:22s} body={parent_name:28s} "
                        f"STL not found: {mesh_file}"
                    )
                    continue

                if params.get("n_capsules", 1) > 1:
                    # Fit N capsules for distal/tip bodies (better arc approximation)
                    n_caps = params.get("n_capsules", 2)
                    capsules = fit_multi_capsules_to_visual_mesh(
                        orig_mesh_path, n_capsules=n_caps,
                    )
                    part_types_cap: list[str] = []
                    for cap_idx, cap in enumerate(capsules):
                        cap_type = cap["type"]
                        type_counts[cap_type] = type_counts.get(cap_type, 0) + 1
                        part_types_cap.append(cap_type)
                        geom_attrs = {
                            "name": f"{parent_name}_capsule_{cap_idx:03d}",
                            "type": cap_type,
                            "group": "3",
                            "friction": "1.5",
                            "solref": "0.01 1.2",
                            "solimp": "0.95 0.995 0.0005",
                        }
                        if "fromto" in cap:
                            geom_attrs["fromto"] = cap["fromto"]
                            geom_attrs["size"] = cap["size"]
                        else:
                            geom_attrs["size"] = cap["size"]
                            geom_attrs["pos"] = cap["pos"]
                            geom_attrs["quat"] = cap["quat"]
                        total_prims += 1
                        geom = ET.Element("geom", geom_attrs)
                        parent.insert(insert_idx, geom)
                        insert_idx += 1
                        insert_after_offset += 1

                    types_str = ", ".join(part_types_cap)
                    # Compute fit quality between capsules and original STL
                    fit_quality = compute_capsule_fit_quality(orig_mesh_path, capsules)
                    fq_str = ""
                    if fit_quality:
                        fq_str = (
                            f" pad_mean={fit_quality['pad_mean']:.2f}mm"
                            f" pad_p95={fit_quality['pad_p95']:.2f}mm"
                            f" pad_max={fit_quality['pad_max']:.2f}mm"
                            f" cover_1mm={fit_quality['coverage_1mm']:.1%}"
                        )
                    report_lines.append(
                        f"{visual_mesh_name:22s} body={parent_name:28s} "
                        f"parts={len(capsules):3d} [MULTI-CAP] types=[{types_str}] file={mesh_file}{fq_str}"
                    )
                else:
                    # Fit a single capsule for proximal/mid bodies
                    axis_hint = params.get("axis_hint", 0)
                    radius_scale = params.get("radius_scale", 1.0)
                    cap = fit_capsule_to_visual_mesh(
                        orig_mesh_path,
                        axis_hint=axis_hint,
                        radius_scale=radius_scale,
                    )

                    cap_type = cap["type"]
                    type_counts[cap_type] = type_counts.get(cap_type, 0) + 1
                    geom_attrs = {
                        "name": f"{parent_name}_capsule_000",
                        "type": cap_type,
                        "group": "3",
                        "friction": "1.5",
                        "solref": "0.01 1.2",
                        "solimp": "0.95 0.995 0.0005",
                    }
                    if "fromto" in cap:
                        geom_attrs["fromto"] = cap["fromto"]
                        geom_attrs["size"] = cap["size"]
                    else:
                        geom_attrs["size"] = cap["size"]
                        geom_attrs["pos"] = cap["pos"]
                        geom_attrs["quat"] = cap["quat"]

                    total_prims += 1
                    geom = ET.Element("geom", geom_attrs)
                    parent.insert(insert_idx, geom)
                    insert_after_offset += 1

                    # Compute fit quality between capsule and original STL
                    fit_quality = compute_capsule_fit_quality(orig_mesh_path, [cap])
                    fq_str = ""
                    if fit_quality:
                        fq_str = (
                            f" pad_mean={fit_quality['pad_mean']:.2f}mm"
                            f" pad_p95={fit_quality['pad_p95']:.2f}mm"
                            f" pad_max={fit_quality['pad_max']:.2f}mm"
                            f" cover_1mm={fit_quality['coverage_1mm']:.1%}"
                        )
                    report_lines.append(
                        f"{visual_mesh_name:22s} body={parent_name:28s} "
                        f"parts=  1 [CAPSULE] type={cap_type} file={mesh_file}{fq_str}"
                    )

    report_lines.extend([
        "",
        f"visual mesh geoms processed: {visual_mesh_count}",
        f"old collision geoms removed: {old_collision_count}",
        f"new collision geoms: {total_prims}",
        f"type breakdown: {type_counts}",
    ])

    indent(root)
    hand_xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(hand_xml_out, encoding="utf-8", xml_declaration=False)

    scene_tree = ET.parse(scene_xml_in)
    scene_root = scene_tree.getroot()
    for include in scene_root.findall("include"):
        if include.get("file") == hand_xml_in.name:
            include.set("file", hand_xml_out.name)

    indent(scene_root)
    scene_tree.write(scene_xml_out, encoding="utf-8", xml_declaration=False)
    report_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] Capsule hand XML: {hand_xml_out}")
    print(f"[OK] Capsule scene XML: {scene_xml_out}")
    print(f"[OK] report: {report_out}")
    print(f"[OK] collision geoms: {total_prims} ({type_counts})")


def create_fitted_primitive_mjcf(
    *,
    hand_xml_in: Path,
    scene_xml_in: Path,
    hand_xml_out: Path,
    scene_xml_out: Path,
    report_out: Path,
    coacd_mesh_dir: Path,
    force: bool,
    mesh_body_prefixes: set[str] | None = None,
) -> None:
    """Generate V2 MuJoCo XML with fitted primitives derived from CoACD hulls.

    Reads the existing CoACD convex hull STLs from *coacd_mesh_dir* and fits
    an optimally-oriented box to each hull.  The resulting XML uses only
    primitive collision geoms by default, which MJX can JIT-compile instantly.

    If *mesh_body_prefixes* is given, visual meshes whose name starts with
    any of those prefixes will keep their CoACD convex mesh geoms (type="mesh")
    instead of fitted boxes.  This gives high-fidelity collision at fingertips
    while keeping the rest lightweight.
    """
    tree = ET.parse(hand_xml_in)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError(f"{hand_xml_in} has no <asset> section")

    mesh_dir = find_compiler_meshdir(root)
    mesh_name_to_file = mesh_assets(root)
    existing_asset_names = {m.get("name") for m in asset.findall("mesh") if m.get("name")}

    if mesh_body_prefixes is None:
        mesh_body_prefixes = set()

    report_lines = [
        "V2 Hybrid collision model report",
        f"source hand xml: {hand_xml_in}",
        f"output hand xml: {hand_xml_out}",
        f"coacd mesh dir (input): {coacd_mesh_dir}",
        "",
    ]

    total_prims = 0
    visual_mesh_count = 0
    old_collision_count = 0
    type_counts: dict[str, int] = {}

    for parent in root.iter():
        children = list(parent)
        visual_insertions: list[tuple[int, ET.Element, str]] = []
        for idx, child in enumerate(children):
            if is_old_collision_geom(child):
                parent.remove(child)
                old_collision_count += 1
            elif is_visual_mesh_geom(child):
                visual_insertions.append((idx, child, child.get("mesh", "")))

        insert_after_offset = 0
        parent_name = parent.get("name", "body")
        for original_idx, visual_geom, visual_mesh_name in visual_insertions:
            # Find existing CoACD STLs for this body
            prefix = f"{visual_mesh_name}_coacd"
            stl_files = natsorted(coacd_mesh_dir.glob(f"{prefix}_*.stl"))
            if not stl_files:
                mesh_file = mesh_name_to_file.get(visual_mesh_name, "?")
                report_lines.append(
                    f"{visual_mesh_name:22s} body={parent_name:28s} "
                    f"NO CoACD STLs found — skipped"
                )
                continue

            # Decide: CoACD mesh geoms (high-fidelity) or fitted box?
            use_mesh = any(visual_mesh_name.startswith(p) for p in mesh_body_prefixes)

            # Compute PCA reference axes from the ORIGINAL visual STL mesh.
            # The original mesh has thousands of vertices → stable, accurate PCA.
            # Individual CoACD hulls have only ~12 unique vertices, making their
            # PCA unreliable (up to 90° deviation measured on this hand model).
            reference_axes = None
            mesh_file = mesh_name_to_file.get(visual_mesh_name, "?")
            if not use_mesh and mesh_file != "?":
                orig_mesh_path = resolve_mesh_path(mesh_file, mesh_dir)
                if orig_mesh_path.exists():
                    orig_mesh = trimesh.load(orig_mesh_path, force="mesh", process=False)
                    reference_axes = _compute_pca_axes(orig_mesh.vertices)

            visual_mesh_count += 1
            insert_idx = original_idx + 1 + insert_after_offset

            part_types: list[str] = []
            for part_idx, stl_path in enumerate(stl_files):
                if use_mesh:
                    # High-fidelity: use the CoACD convex mesh directly
                    rel_file = Path(os.path.relpath(stl_path, mesh_dir)).as_posix()
                    asset_name = make_unique_asset_name(
                        existing_asset_names,
                        f"{visual_mesh_name}_coacd_{part_idx:03d}",
                    )
                    ET.SubElement(asset, "mesh", {"name": asset_name, "file": rel_file})
                    geom_attrs = {
                        "name": f"{parent_name}_coacd_{part_idx:03d}",
                        "type": "mesh",
                        "mesh": asset_name,
                        "group": "3",
                        "friction": "1.5",
                        "solref": "0.01 1.2",
                        "solimp": "0.95 0.995 0.0005",
                    }
                    part_types.append("mesh")
                    type_counts["mesh"] = type_counts.get("mesh", 0) + 1
                else:
                    # Lightweight: fitted OBB box
                    prim = fit_primitive_to_convex_hull(stl_path, reference_axes=reference_axes)
                    prim_type = prim["type"]
                    part_types.append(prim_type)
                    type_counts[prim_type] = type_counts.get(prim_type, 0) + 1
                    geom_attrs = {
                        "name": f"{parent_name}_fitted_{part_idx:03d}",
                        "type": prim_type,
                        "size": prim["size"],
                        "pos": prim["pos"],
                        "quat": prim["quat"],
                        "group": "3",
                        "friction": "1.5",
                        "solref": "0.01 1.2",
                        "solimp": "0.95 0.995 0.0005",
                    }

                total_prims += 1
                geom = ET.Element("geom", geom_attrs)
                parent.insert(insert_idx, geom)
                insert_idx += 1
                insert_after_offset += 1

            types_str = ", ".join(part_types)
            mode_tag = "MESH" if use_mesh else "BOX"
            report_lines.append(
                f"{visual_mesh_name:22s} body={parent_name:28s} "
                f"parts={len(stl_files):3d} [{mode_tag}] types=[{types_str}] file={mesh_file}"
            )

    report_lines.extend([
        "",
        f"visual mesh geoms processed: {visual_mesh_count}",
        f"old collision geoms removed: {old_collision_count}",
        f"new fitted primitive geoms: {total_prims}",
        f"type breakdown: {type_counts}",
    ])

    indent(root)
    hand_xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(hand_xml_out, encoding="utf-8", xml_declaration=False)

    scene_tree = ET.parse(scene_xml_in)
    scene_root = scene_tree.getroot()
    include_hits = 0
    for include in scene_root.findall("include"):
        if include.get("file") == hand_xml_in.name:
            include.set("file", hand_xml_out.name)
            include_hits += 1
    if include_hits != 1:
        raise RuntimeError(f"expected one include of {hand_xml_in.name}, found {include_hits}")

    indent(scene_root)
    scene_tree.write(scene_xml_out, encoding="utf-8", xml_declaration=False)
    report_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] Fitted-primitive hand XML: {hand_xml_out}")
    print(f"[OK] Fitted-primitive scene XML: {scene_xml_out}")
    print(f"[OK] report: {report_out}")
    print(f"[OK] fitted primitives: {total_prims} ({type_counts})")


def indent(elem: ET.Element, level: int = 0) -> None:
    """Pretty-print XML while staying compatible with older Python versions."""
    i = "\n" + level * "  "
    if len(elem):
        if not elem.text or not elem.text.strip():
            elem.text = i + "  "
        for child in elem:
            indent(child, level + 1)
        if not child.tail or not child.tail.strip():
            child.tail = i
    if level and (not elem.tail or not elem.tail.strip()):
        elem.tail = i


def require_coacd() -> None:
    if coacd is None:
        raise RuntimeError(
            "coacd is not installed. Install it in the active env with: "
            "python -m pip install coacd"
        ) from COACD_IMPORT_ERROR


def _farthest_point_sample(points: np.ndarray, n: int) -> np.ndarray:
    """Select *n* well-spread vertices via farthest-point sampling."""
    idx = [0]
    dists = np.full(len(points), np.inf)
    for _ in range(n - 1):
        d = np.linalg.norm(points - points[idx[-1]], axis=1)
        dists = np.minimum(dists, d)
        idx.append(int(np.argmax(dists)))
    return np.array(idx)


def _simplify_convex_mesh(mesh: trimesh.Trimesh, max_verts: int = 12) -> trimesh.Trimesh:
    """Reduce a convex mesh to *max_verts* via FPS + ConvexHull."""
    hull = mesh.convex_hull
    verts = hull.vertices
    if verts.shape[0] <= max_verts:
        return hull
    sampled = verts[_farthest_point_sample(verts, max_verts)]
    ch = _ScipyConvexHull(sampled)
    return trimesh.Trimesh(vertices=sampled, faces=ch.simplices, process=False)


def resolve_mesh_path(mesh_file: str, mesh_dir: Path) -> Path:
    path = Path(mesh_file)
    if path.is_absolute():
        return path
    return (mesh_dir / path).resolve()


def load_mesh_for_coacd(mesh_path: Path) -> trimesh.Trimesh:
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    if not isinstance(mesh, trimesh.Trimesh):
        raise TypeError(f"{mesh_path} did not load as a triangle mesh")
    if mesh.vertices.size == 0 or mesh.faces.size == 0:
        raise ValueError(f"{mesh_path} has no vertices/faces")
    return mesh


def decompose_mesh_to_stl_parts(
    mesh_path: Path,
    out_dir: Path,
    prefix: str,
    *,
    threshold: float,
    max_convex_hull: int,
    preprocess_resolution: int,
    resolution: int,
    mcts_nodes: int,
    mcts_iterations: int,
    mcts_max_depth: int,
    max_ch_vertex: int,
    seed: int,
    force: bool,
    simplify_max_verts: int = 12,
) -> list[Path]:
    """Run CoACD on a mesh and export each convex part as STL."""
    require_coacd()
    out_dir.mkdir(parents=True, exist_ok=True)
    existing = natsorted(out_dir.glob(f"{prefix}_*.stl"))
    if existing and not force:
        return list(existing)

    for path in existing:
        path.unlink()

    mesh = load_mesh_for_coacd(mesh_path)
    coacd_mesh = coacd.Mesh(mesh.vertices, mesh.faces)
    parts = coacd.run_coacd(
        coacd_mesh,
        threshold=threshold,
        max_convex_hull=max_convex_hull,
        preprocess_mode="auto",
        preprocess_resolution=preprocess_resolution,
        resolution=resolution,
        mcts_nodes=mcts_nodes,
        mcts_iterations=mcts_iterations,
        mcts_max_depth=mcts_max_depth,
        pca=False,
        merge=True,
        decimate=False,
        max_ch_vertex=max_ch_vertex,
        extrude=False,
        seed=seed,
    )

    saved_paths: list[Path] = []
    for part_idx, (vertices, faces) in enumerate(parts):
        part_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=True)
        # Post-process: reduce to a low-poly convex hull via farthest-point
        # sampling.  MJX pads ALL meshes to the max face count, so every
        # mesh must be tiny (~12 verts / ~20 faces).
        try:
            part_mesh = _simplify_convex_mesh(part_mesh, max_verts=simplify_max_verts)
        except Exception:
            pass  # Keep the original if processing fails.
        part_path = out_dir / f"{prefix}_{part_idx:03d}.stl"
        part_mesh.export(part_path)
        saved_paths.append(part_path)
    return saved_paths


def replace_mesh_collision_with_multi_convex(
    urdf_in: str,
    urdf_out: str,
    target_mesh_relpath: str,
    convex_dir: str,
    convex_glob: str,
) -> None:
    """URDF helper retained for compatibility with the original script."""
    urdf_in_path = Path(urdf_in).resolve()
    urdf_out_path = Path(urdf_out).resolve()
    urdf_dir = urdf_in_path.parent

    tree = ET.parse(str(urdf_in_path))
    root = tree.getroot()

    convex_dir_path = Path(convex_dir).resolve()
    convex_files = natsorted(convex_dir_path.glob(convex_glob))
    if not convex_files:
        raise FileNotFoundError(f"在 {convex_dir_path} 下没有找到 {convex_glob}")

    target_hits = 0
    for link in root.findall("link"):
        collisions = list(link.findall("collision"))
        for col in collisions:
            geom = col.find("geometry")
            mesh = geom.find("mesh") if geom is not None else None
            if mesh is None:
                continue

            filename = mesh.get("filename", "")
            if filename != target_mesh_relpath:
                continue

            origin = col.find("origin")
            insert_idx = list(link).index(col)
            link.remove(col)

            for fpath in convex_files:
                new_col = ET.Element("collision")
                if origin is not None:
                    new_col.append(copy.deepcopy(origin))
                else:
                    new_ori = ET.SubElement(new_col, "origin")
                    new_ori.set("xyz", "0 0 0")

                new_geom = ET.SubElement(new_col, "geometry")
                new_mesh = ET.SubElement(new_geom, "mesh")
                rel = Path(os.path.relpath(fpath, urdf_dir)).as_posix()
                new_mesh.set("filename", rel)

                link.insert(insert_idx, new_col)
                insert_idx += 1

            target_hits += 1

    if target_hits == 0:
        raise RuntimeError(
            f"未在 URDF 找到 filename='{target_mesh_relpath}' 的 <collision><mesh/>。"
        )

    indent(root)
    urdf_out_path.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(urdf_out_path), encoding="utf-8", xml_declaration=True)
    print(f"[OK] 已替换 {target_hits} 处碰撞体。输出：{urdf_out_path}")


def urdf_decompose(
    urdf_file: str,
    mesh_name_list: list[str],
    urdf_out: str | None = None,
    output_mesh_path: str | None = None,
    visualize: bool = False,
) -> None:
    """Original URDF workflow, now with optional visualization import."""
    require_coacd()
    if visualize:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise RuntimeError(
                "visualize=True requires open3d. Install with: python -m pip install open3d"
            ) from exc
    else:
        o3d = None

    def iter_collision_mesh_filenames(urdf_path: str):
        tree = ET.parse(urdf_path)
        root = tree.getroot()
        for link in root.findall("link"):
            for col in link.findall("collision"):
                geom = col.find("geometry")
                if geom is None:
                    continue
                mesh = geom.find("mesh")
                if mesh is None:
                    continue
                fn = mesh.get("filename")
                if fn:
                    yield link.get("name", ""), fn

    matched: list[tuple[str, str, str]] = []
    for link_name, fn in iter_collision_mesh_filenames(urdf_file):
        for mesh_name in mesh_name_list:
            if os.path.basename(fn) == mesh_name:
                clean_fn = fn.replace("package://", "", 1) if fn.startswith("package://") else fn
                path = Path(clean_fn)
                mesh_path = path if path.is_absolute() else Path(urdf_file).resolve().parent / path
                matched.append((link_name, fn, str(mesh_path.resolve())))

    stem, ext = os.path.splitext(os.path.basename(urdf_file))
    urdf_out = (
        os.path.join(os.path.dirname(os.path.abspath(urdf_file)), f"{stem}_decompose{ext}")
        if urdf_out is None
        else urdf_out
    )
    if not matched:
        raise ValueError(f'未在 <collision> 中找到 filename 以 "{mesh_name_list}" 结尾的 mesh。')

    for link_name, fn, mesh_path_str in matched:
        mesh_path = Path(mesh_path_str)
        print(f'  link={link_name:15s}  filename="{fn}"')
        print(f"    -> 绝对路径: {mesh_path}")

        mesh_name = mesh_path.name
        out_dir = Path(output_mesh_path) if output_mesh_path else mesh_path.parent
        parts = decompose_mesh_to_stl_parts(
            mesh_path,
            out_dir,
            mesh_path.stem,
            threshold=0.03,
            max_convex_hull=-1,
            preprocess_resolution=50,
            resolution=2000,
            mcts_nodes=20,
            mcts_iterations=150,
            mcts_max_depth=3,
            max_ch_vertex=256,
            seed=0,
            force=True,
        )

        if visualize and o3d is not None:
            original = load_mesh_for_coacd(mesh_path)
            original_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(original.vertices),
                triangles=o3d.utility.Vector3iVector(original.faces),
            )
            original_mesh.paint_uniform_color([0.5, 0.5, 0.5])
            o3d.visualization.draw_geometries([original_mesh], window_name="Original Mesh")

        print(f"已保存 {len(parts)} 个凸网格到：{out_dir}")
        replace_mesh_collision_with_multi_convex(
            urdf_file,
            urdf_out,
            fn,
            str(out_dir),
            convex_glob=f"{Path(mesh_name).stem}_*.stl",
        )
        urdf_file = urdf_out


def mesh_assets(root: ET.Element) -> dict[str, str]:
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError("MJCF has no <asset> section")
    assets: dict[str, str] = {}
    for mesh in asset.findall("mesh"):
        name = mesh.get("name")
        file = mesh.get("file")
        if name and file:
            assets[name] = file
    return assets


def find_compiler_meshdir(root: ET.Element) -> Path:
    compiler = root.find("compiler")
    if compiler is None:
        raise RuntimeError("MJCF has no <compiler> element")
    meshdir = compiler.get("meshdir")
    if not meshdir:
        raise RuntimeError("MJCF compiler has no meshdir")
    return Path(meshdir).expanduser().resolve()


def make_unique_asset_name(existing_names: set[str], base: str) -> str:
    name = base
    idx = 1
    while name in existing_names:
        name = f"{base}_{idx}"
        idx += 1
    existing_names.add(name)
    return name


def is_visual_mesh_geom(elem: ET.Element) -> bool:
    return elem.tag == "geom" and elem.get("type") == "mesh" and elem.get("class") == "visual"


def is_old_collision_geom(elem: ET.Element) -> bool:
    if elem.tag != "geom":
        return False
    if elem.get("class") == "visual":
        return False
    if elem.get("group") == "3":
        return True
    name = elem.get("name", "")
    return name.endswith("_col") or "_col_" in name or name == "palm_rubber"


def create_coacd_mjcf(
    *,
    hand_xml_in: Path,
    scene_xml_in: Path,
    hand_xml_out: Path,
    scene_xml_out: Path,
    report_out: Path,
    out_mesh_dir: Path,
    threshold: float,
    max_convex_hull: int,
    preprocess_resolution: int,
    resolution: int,
    mcts_nodes: int,
    mcts_iterations: int,
    mcts_max_depth: int,
    max_ch_vertex: int,
    seed: int,
    force: bool,
    lite_body_params: dict[str, dict[str, float | int]] | None = None,
) -> None:
    """Generate a V2 MuJoCo XML whose collision geoms are CoACD mesh pieces."""
    require_coacd()

    tree = ET.parse(hand_xml_in)
    root = tree.getroot()
    asset = root.find("asset")
    if asset is None:
        raise RuntimeError(f"{hand_xml_in} has no <asset> section")

    mesh_dir = find_compiler_meshdir(root)
    mesh_name_to_file = mesh_assets(root)
    existing_asset_names = {mesh.get("name") for mesh in asset.findall("mesh") if mesh.get("name")}

    report_lines = [
        "V2 CoACD collision model report",
        f"source hand xml: {hand_xml_in}",
        f"output hand xml: {hand_xml_out}",
        f"meshdir: {mesh_dir}",
        f"coacd output dir: {out_mesh_dir}",
        (
            "coacd params: "
            f"threshold={threshold}, max_convex_hull={max_convex_hull}, "
            f"preprocess_resolution={preprocess_resolution}, resolution={resolution}, "
            f"mcts_nodes={mcts_nodes}, mcts_iterations={mcts_iterations}, "
            f"mcts_max_depth={mcts_max_depth}, max_ch_vertex={max_ch_vertex}, seed={seed}"
        ),
        "",
    ]

    total_parts = 0
    visual_mesh_count = 0
    old_collision_count = 0

    for parent in root.iter():
        children = list(parent)
        visual_insertions: list[tuple[int, ET.Element, str]] = []
        for idx, child in enumerate(children):
            if is_old_collision_geom(child):
                parent.remove(child)
                old_collision_count += 1
            elif is_visual_mesh_geom(child):
                visual_insertions.append((idx, child, child.get("mesh", "")))

        insert_after_offset = 0
        parent_name = parent.get("name", "body")
        for original_idx, visual_geom, visual_mesh_name in visual_insertions:
            mesh_file = mesh_name_to_file.get(visual_mesh_name)
            if not mesh_file:
                raise RuntimeError(f"visual mesh asset {visual_mesh_name!r} has no file")

            mesh_path = resolve_mesh_path(mesh_file, mesh_dir)
            prefix = f"{visual_mesh_name}_coacd"

            # Resolve per-body overrides for lite mode.
            effective_threshold = threshold
            effective_max_ch = max_convex_hull
            effective_max_verts = 12  # default: aggressive simplification
            if lite_body_params and visual_mesh_name in lite_body_params:
                overrides = lite_body_params[visual_mesh_name]
                effective_threshold = overrides.get("threshold", threshold)
                effective_max_ch = overrides.get("max_convex_hull", max_convex_hull)
                effective_max_verts = overrides.get("max_verts", 12)

            part_paths = decompose_mesh_to_stl_parts(
                mesh_path,
                out_mesh_dir,
                prefix,
                threshold=effective_threshold,
                max_convex_hull=effective_max_ch,
                preprocess_resolution=preprocess_resolution,
                resolution=resolution,
                mcts_nodes=mcts_nodes,
                mcts_iterations=mcts_iterations,
                mcts_max_depth=mcts_max_depth,
                max_ch_vertex=max_ch_vertex,
                seed=seed,
                force=force,
                simplify_max_verts=effective_max_verts,
            )

            visual_mesh_count += 1
            total_parts += len(part_paths)
            original_mesh = load_mesh_for_coacd(mesh_path)
            report_lines.append(
                f"{visual_mesh_name:22s} body={parent_name:28s} "
                f"tri={len(original_mesh.faces):6d} parts={len(part_paths):3d} "
                f"th={effective_threshold:.2f} maxch={effective_max_ch} file={mesh_file}"
            )

            insert_idx = original_idx + 1 + insert_after_offset
            for part_idx, part_path in enumerate(part_paths):
                rel_file = Path(os.path.relpath(part_path, mesh_dir)).as_posix()
                asset_name = make_unique_asset_name(
                    existing_asset_names, f"{visual_mesh_name}_coacd_{part_idx:03d}"
                )
                ET.SubElement(asset, "mesh", {"name": asset_name, "file": rel_file})

                geom = ET.Element(
                    "geom",
                    {
                        "name": f"{parent_name}_coacd_{part_idx:03d}",
                        "type": "mesh",
                        "mesh": asset_name,
                        "group": "3",
                        "friction": "1.5",
                        "solref": "0.01 1.2",
                        "solimp": "0.95 0.995 0.0005",
                    },
                )
                parent.insert(insert_idx, geom)
                insert_idx += 1
                insert_after_offset += 1

    report_lines.extend(
        [
            "",
            f"visual mesh geoms decomposed: {visual_mesh_count}",
            f"old primitive collision geoms removed: {old_collision_count}",
            f"new convex mesh collision geoms: {total_parts}",
        ]
    )

    indent(root)
    hand_xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(hand_xml_out, encoding="utf-8", xml_declaration=False)

    scene_tree = ET.parse(scene_xml_in)
    scene_root = scene_tree.getroot()
    include_hits = 0
    for include in scene_root.findall("include"):
        if include.get("file") == hand_xml_in.name:
            include.set("file", hand_xml_out.name)
            include_hits += 1
    if include_hits != 1:
        raise RuntimeError(f"expected one include of {hand_xml_in.name}, found {include_hits}")

    indent(scene_root)
    scene_tree.write(scene_xml_out, encoding="utf-8", xml_declaration=False)
    report_out.write_text("\n".join(report_lines) + "\n", encoding="utf-8")

    print(f"[OK] CoACD hand XML: {hand_xml_out}")
    print(f"[OK] CoACD scene XML: {scene_xml_out}")
    print(f"[OK] report: {report_out}")
    print(f"[OK] convex parts: {total_parts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convex-decompose URDF collisions or generate Aero-Hand V2 CoACD MJCF."
    )
    parser.add_argument(
        "--v2-mjcf",
        action="store_true",
        help="Generate V2 MuJoCo CoACD collision XML from right_hand_v2_vertical.xml.",
    )
    parser.add_argument("--hand-xml-in", type=Path, default=V2_HAND_XML)
    parser.add_argument("--scene-xml-in", type=Path, default=V2_SCENE_XML)
    parser.add_argument("--hand-xml-out", type=Path, default=V2_COACD_HAND_XML)
    parser.add_argument("--scene-xml-out", type=Path, default=V2_COACD_SCENE_XML)
    parser.add_argument("--report-out", type=Path, default=V2_COACD_REPORT)
    parser.add_argument("--out-mesh-dir", type=Path, default=V2_COACD_DIR)
    parser.add_argument("--threshold", type=float, default=0.01)
    parser.add_argument("--max-convex-hull", type=int, default=48)
    parser.add_argument("--preprocess-resolution", type=int, default=80)
    parser.add_argument("--resolution", type=int, default=3000)
    parser.add_argument("--mcts-nodes", type=int, default=24)
    parser.add_argument("--mcts-iterations", type=int, default=180)
    parser.add_argument("--mcts-max-depth", type=int, default=4)
    parser.add_argument("--max-ch-vertex", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--force", action="store_true", help="Recompute existing convex part STL files.")
    parser.add_argument(
        "--lite",
        action="store_true",
        help="Use per-body simplified parameters for MJX-friendly CoACD-lite model.",
    )
    parser.add_argument(
        "--fitted",
        action="store_true",
        help="Generate fitted primitives (box/capsule) from existing CoACD convex hulls.",
    )
    parser.add_argument(
        "--hybrid",
        action="store_true",
        help="Hybrid mode (use with --fitted): fingertip/distal bodies keep CoACD "
             "convex mesh geoms for high fidelity; other bodies use fitted boxes.",
    )
    parser.add_argument(
        "--capsule",
        action="store_true",
        help="Generate capsule collision model: one capsule per finger segment, "
             "boxes for palm/thumb-base.  MJX-friendly (all analytical collision).",
    )

    parser.add_argument("--urdf", type=Path, help="URDF file for the legacy workflow.")
    parser.add_argument("--mesh", nargs="*", default=[], help="Mesh basenames to decompose in the URDF.")
    parser.add_argument("--urdf-out", type=Path)
    parser.add_argument("--visualize", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.v2_mjcf and args.capsule:
        # Capsule mode: one capsule per finger segment, boxes for palm/thumb-base.
        create_capsule_mjcf(
            hand_xml_in=args.hand_xml_in.resolve(),
            scene_xml_in=args.scene_xml_in.resolve(),
            hand_xml_out=args.hand_xml_out.resolve(),
            scene_xml_out=args.scene_xml_out.resolve(),
            report_out=args.report_out.resolve(),
        )
        return 0

    if args.v2_mjcf and args.fitted:
        # Fitted-primitive mode: read existing CoACD STLs and generate primitives.
        # In hybrid mode, distal segments + thumb tip keep CoACD mesh geoms.
        mesh_prefixes: set[str] | None = None
        if args.hybrid:
            mesh_prefixes = {
                "v2_index_distal",
                "v2_middle_distal",
                "v2_ring_distal",
                "v2_pinky_distal",
                "v2_thumb_tip",
            }
        create_fitted_primitive_mjcf(
            hand_xml_in=args.hand_xml_in.resolve(),
            scene_xml_in=args.scene_xml_in.resolve(),
            hand_xml_out=args.hand_xml_out.resolve(),
            scene_xml_out=args.scene_xml_out.resolve(),
            report_out=args.report_out.resolve(),
            coacd_mesh_dir=args.out_mesh_dir.resolve(),
            force=args.force,
            mesh_body_prefixes=mesh_prefixes,
        )
        return 0

    if args.v2_mjcf:
        create_coacd_mjcf(
            hand_xml_in=args.hand_xml_in.resolve(),
            scene_xml_in=args.scene_xml_in.resolve(),
            hand_xml_out=args.hand_xml_out.resolve(),
            scene_xml_out=args.scene_xml_out.resolve(),
            report_out=args.report_out.resolve(),
            out_mesh_dir=args.out_mesh_dir.resolve(),
            threshold=args.threshold,
            max_convex_hull=args.max_convex_hull,
            preprocess_resolution=args.preprocess_resolution,
            resolution=args.resolution,
            mcts_nodes=args.mcts_nodes,
            mcts_iterations=args.mcts_iterations,
            mcts_max_depth=args.mcts_max_depth,
            max_ch_vertex=args.max_ch_vertex,
            seed=args.seed,
            force=args.force,
            lite_body_params=LITE_BODY_PARAMS if args.lite else None,
        )
        return 0

    if args.urdf:
        if not args.mesh:
            print("[error] --urdf workflow requires at least one --mesh basename", file=sys.stderr)
            return 2
        urdf_decompose(
            str(args.urdf),
            args.mesh,
            urdf_out=str(args.urdf_out) if args.urdf_out else None,
            visualize=args.visualize,
        )
        return 0

    print("Nothing to do. Use --v2-mjcf or --urdf ... --mesh ...", file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
