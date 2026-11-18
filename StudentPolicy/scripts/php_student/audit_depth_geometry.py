"""Compare recorded normalized depth against the paired static terrain mesh."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import trimesh


def rotate_xyzw(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    qv = quat[:3]
    return vec + 2.0 * np.cross(qv, np.cross(qv, vec) + quat[3] * vec)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    parser.add_argument("terrain_mesh", type=Path, help="terrain mesh NPZ emitted by apply_terrain_preprocess")
    parser.add_argument("--frame-stride", type=int, default=2)
    args = parser.parse_args()
    if args.frame_stride < 1:
        parser.error("--frame-stride must be positive")
    mesh_data = np.load(args.terrain_mesh)
    mesh = trimesh.Trimesh(vertices=mesh_data["vertices"], faces=mesh_data["faces"], process=False)

    width, height, hfov = 106, 60, np.deg2rad(89.5)
    fx = (width / 2.0) / np.tan(hfov / 2.0)
    fy = fx
    cx, cy = width / 2.0, height / 2.0
    # The term crops [top=2,bottom=0,left=4,right=4], then resizes 58x98 -> 58x87.
    rows, cols = np.indices((58, 87), dtype=np.float64)
    y_full = rows + 2.0
    x_full = (cols + 0.5) * (98.0 / 87.0) - 0.5 + 4.0
    rays_camera = np.stack(((x_full - cx) / fx, (y_full - cy) / fy, np.ones_like(x_full)), axis=-1)
    rays_camera /= np.linalg.norm(rays_camera, axis=-1, keepdims=True)
    rays_camera = rays_camera.reshape(-1, 3)

    print("motion images terrain_hit_fraction within_0.10m within_0.20m median_abs_m p90_abs_m")
    all_errors: list[np.ndarray] = []
    for path in sorted(args.trace_dir.glob("motion_*.npz")):
        with np.load(path) as data:
            images = data["depth"][:: args.frame_stride]
            poses = data["depth_camera_pos_w"][:: args.frame_stride]
            quats = data["depth_camera_quat_xyzw"][:: args.frame_stride]
            measured = np.clip((images + 0.5) * 2.7 + 0.3, 0.3, 3.0)
        origins = np.repeat(poses, rays_camera.shape[0], axis=0)
        directions = np.concatenate(
            [np.stack([rotate_xyzw(q, ray) for ray in rays_camera]) for q in quats], axis=0
        )
        hit_locs, ray_ids, _ = mesh.ray.intersects_location(origins, directions, multiple_hits=False)
        expected = np.full(len(origins), np.nan, dtype=np.float64)
        if len(ray_ids):
            ray_dir = directions[ray_ids]
            principal = np.stack([rotate_xyzw(q, np.array([0.0, 0.0, 1.0])) for q in quats])
            principal = np.repeat(principal, rays_camera.shape[0], axis=0)[ray_ids]
            distance = np.einsum("ij,ij->i", hit_locs - origins[ray_ids], ray_dir)
            z_depth = distance * np.einsum("ij,ij->i", ray_dir, principal)
            valid = (z_depth >= 0.3) & (z_depth <= 3.0)
            expected[ray_ids[valid]] = z_depth[valid]
        measured_flat = measured.reshape(-1)
        valid = np.isfinite(expected)
        errors = np.abs(measured_flat[valid] - expected[valid])
        all_errors.append(errors)
        print(
            f"{path.stem} {len(images)} {valid.mean():.3f} "
            f"{(errors < 0.10).mean():.3f} {(errors < 0.20).mean():.3f} "
            f"{np.median(errors):.3f} {np.quantile(errors, 0.90):.3f}"
        )
    combined = np.concatenate(all_errors) if all_errors else np.array([], dtype=np.float64)
    if combined.size:
        print(
            f"OVERALL hit_pixels={combined.size} within_0.10m={(combined < 0.10).mean():.3f} "
            f"within_0.20m={(combined < 0.20).mean():.3f} median_abs_m={np.median(combined):.3f} "
            f"p90_abs_m={np.quantile(combined, 0.90):.3f}"
        )


if __name__ == "__main__":
    main()
