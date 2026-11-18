"""Independent path-projection proxy for whether a rollout reaches the first obstacle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--student-traces", type=Path, required=True)
    parser.add_argument("--teacher-traces", type=Path, required=True)
    parser.add_argument("--terrain-mesh", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    with np.load(args.terrain_mesh) as mesh:
        vertices = mesh["vertices"]
    if len(vertices) % 8:
        raise ValueError("Expected 8 vertices per obstacle/ground cuboid")
    boxes = vertices.reshape(-1, 8, 3)[:-1]  # final cuboid is the broad ground plane
    names = [line.strip() for line in (args.student_traces.parent / "motion_names.txt").read_text().splitlines() if line.strip()]
    rows = []
    for student_file in sorted(args.student_traces.glob("motion_*.npz")):
        motion_id = int(student_file.stem.split("_")[-1])
        with np.load(student_file) as student, np.load(args.teacher_traces / student_file.name) as teacher:
            start = teacher["ref_pos_w"][0, :2]
            end = teacher["ref_pos_w"][-1, :2]
            path = end - start
            length = np.linalg.norm(path)
            direction = path / length
            centers = boxes.mean(axis=1)[:, :2]
            center_projection = (centers - start) @ direction
            lateral_distance = np.linalg.norm(
                centers - start - center_projection[:, None] * direction, axis=1
            )
            candidates = np.flatnonzero(
                (center_projection > 0) & (center_projection < length) & (lateral_distance < 1.5)
            )
            if not len(candidates):
                raise RuntimeError(f"No obstacle in reference corridor for {student_file.name}")
            first = int(candidates[np.argmin(center_projection[candidates])])
            corners = (boxes[first, :, :2] - start) @ direction
            student_progress = (student["robot_ref_pos_w"][:, :2] - start) @ direction
            teacher_progress = (teacher["robot_ref_pos_w"][:, :2] - start) @ direction
            rows.append({
                "motion_id": motion_id,
                "motion": names[motion_id],
                "reference_path_length_m": float(length),
                "first_obstacle_index": first,
                "obstacle_front_projection_m": float(corners.min()),
                "obstacle_center_projection_m": float(center_projection[first]),
                "obstacle_back_projection_m": float(corners.max()),
                "student_max_anchor_projection_m": float(student_progress.max()),
                "teacher_max_anchor_projection_m": float(teacher_progress.max()),
                "student_passed_front_proxy": bool(student_progress.max() > corners.min()),
                "student_passed_center_proxy": bool(student_progress.max() > center_projection[first]),
                "student_passed_back_proxy": bool(student_progress.max() > corners.max()),
            })
    result = {
        "contract": "xy_reference_path_projection_proxy_not_contact_or_clearance_success",
        "corridor_half_width_m": 1.5,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print("motion student_max front center back passed_center")
    for row in rows:
        print(
            f"{row['motion']} {row['student_max_anchor_projection_m']:.3f} "
            f"{row['obstacle_front_projection_m']:.3f} {row['obstacle_center_projection_m']:.3f} "
            f"{row['obstacle_back_projection_m']:.3f} {row['student_passed_center_proxy']}"
        )
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
