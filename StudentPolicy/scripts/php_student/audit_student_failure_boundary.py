#!/usr/bin/env python3
"""Build a per-clip causal timeline from paired Student/Teacher traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def first_sustained(mask: np.ndarray, width: int) -> int | None:
    if len(mask) < width:
        return None
    hits = np.convolve(mask.astype(np.int32), np.ones(width, dtype=np.int32), mode="valid")
    found = np.flatnonzero(hits == width)
    return int(found[0]) if found.size else None


def frame_at(frames: np.ndarray, index: int | None) -> int | None:
    return int(frames[index]) if index is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--start-map", type=Path, required=True)
    parser.add_argument("--action-threshold", type=float, default=1.0)
    parser.add_argument("--sustain", type=int, default=10)
    parser.add_argument("--lead-frames", type=int, default=20)
    args = parser.parse_args()

    names_path = args.trace_dir.parent / "motion_names.txt"
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    rows: list[dict] = []
    start_map: dict[str, int] = {}
    for path in sorted(args.trace_dir.glob("motion_*.npz")):
        motion_id = int(path.stem.rsplit("_", 1)[1])
        with np.load(path) as data:
            frames = data["frame"].astype(np.int64)
            valid = data["teacher_valid"].astype(bool)
            action_l2 = np.linalg.norm(data["student_action"] - data["teacher_action"], axis=1)
            anchor = np.linalg.norm(data["ref_pos_w"] - data["robot_ref_pos_w"], axis=1)
            contact = np.linalg.norm(data["foot_contact_force"], axis=-1) > 20.0
            action_index = first_sustained(action_l2 > args.action_threshold, args.sustain)
            anchor10_index = first_sustained(anchor > 0.10, args.sustain)
            invalid_indices = np.flatnonzero(~valid)
            invalid_index = int(invalid_indices[0]) if invalid_indices.size else None
            # A robust contact event is the first sustained pattern change relative
            # to the initial contact state. It is descriptive, not used to choose
            # the intervention point.
            initial_contact = contact[: min(10, len(contact))].mean(axis=0) >= 0.5
            changed = np.any(contact != initial_contact[None, :], axis=1)
            contact_index = first_sustained(changed, 3)
            action_frame = frame_at(frames, action_index)
            invalid_frame = frame_at(frames, invalid_index)
            if action_frame is not None:
                intervention = max(int(frames[0]), action_frame - args.lead_frames)
                source = "action_divergence"
            elif anchor10_index is not None:
                intervention = max(int(frames[0]), int(frames[anchor10_index]) - args.lead_frames)
                source = "anchor_error"
            else:
                intervention = int(frames[max(0, len(frames) // 2 - args.lead_frames)])
                source = "midpoint_fallback"
            start_map[str(motion_id)] = intervention
            if action_frame is None or invalid_frame is None:
                order = "incomplete_timeline"
            elif action_frame < invalid_frame:
                order = "action_before_label_invalid"
            elif invalid_frame < action_frame:
                order = "label_invalid_before_action"
            else:
                order = "same_frame"
            rows.append({
                "motion_id": motion_id,
                "motion": names[motion_id],
                "recorded_steps": len(frames),
                "end_frame": int(frames[-1]),
                "action_divergence_frame": action_frame,
                "anchor_gt_10cm_frame": frame_at(frames, anchor10_index),
                "teacher_invalid_frame": invalid_frame,
                "contact_pattern_change_frame": frame_at(frames, contact_index),
                "timeline_order": order,
                "recovery_start_frame": intervention,
                "recovery_start_source": source,
                "valid_fraction": float(valid.mean()),
                "action_l2_valid_mean": float(action_l2[valid].mean()) if valid.any() else None,
            })

    counts: dict[str, int] = {}
    for row in rows:
        counts[row["timeline_order"]] = counts.get(row["timeline_order"], 0) + 1
    payload = {
        "contract": "paired actions on Student-controlled states; sustained thresholds",
        "action_l2_threshold": args.action_threshold,
        "sustain_frames": args.sustain,
        "teacher_recovery_lead_frames": args.lead_frames,
        "motion_count": len(rows),
        "timeline_counts": counts,
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    args.start_map.write_text(json.dumps(start_map, indent=2) + "\n")
    print("id | action_div | anchor10 | invalid | contact | order | recovery_start")
    for row in rows:
        print(
            f"{row['motion_id']:2d} | {str(row['action_divergence_frame']):>10} | "
            f"{str(row['anchor_gt_10cm_frame']):>8} | {str(row['teacher_invalid_frame']):>7} | "
            f"{str(row['contact_pattern_change_frame']):>7} | {row['timeline_order']} | "
            f"{row['recovery_start_frame']}"
        )
    print(f"[SUMMARY] {counts}")
    print(f"[PASS] {args.output}")
    print(f"[PASS] {args.start_map}")


if __name__ == "__main__":
    main()
