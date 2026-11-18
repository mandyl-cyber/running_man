"""Locate action divergence and tracking-error onset in paired Student traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def sustained(values: np.ndarray, threshold: float, width: int = 10) -> int | None:
    mask = values > threshold
    if len(mask) < width:
        return None
    hits = np.convolve(mask.astype(np.int32), np.ones(width, dtype=np.int32), mode="valid")
    indices = np.flatnonzero(hits == width)
    return int(indices[0]) if indices.size else None


def frame_at(frames: np.ndarray, index: int | None) -> int | None:
    return int(frames[index]) if index is not None else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dirs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = []
    for trace_dir in args.trace_dirs:
        arm = trace_dir.parent.name
        names = [x.strip() for x in (trace_dir.parent / "motion_names.txt").read_text().splitlines() if x.strip()]
        for path in sorted(trace_dir.glob("motion_*.npz")):
            motion_id = int(path.stem.rsplit("_", 1)[1])
            with np.load(path) as data:
                frames = data["frame"].astype(np.int32)
                valid = data["teacher_valid"].astype(bool)
                action = np.linalg.norm(data["student_action"] - data["teacher_action"], axis=1)
                anchor = np.linalg.norm(data["ref_pos_w"] - data["robot_ref_pos_w"], axis=1)
                lost_indices = np.flatnonzero(~valid)
                lost = int(lost_indices[0]) if lost_indices.size else None
                action1 = sustained(action, 1.0)
                pos10 = sustained(anchor, 0.10)
                pos20 = sustained(anchor, 0.20)
                row = {
                    "arm": arm, "motion_id": motion_id, "motion": names[motion_id],
                    "steps": len(frames), "end_frame": int(frames[-1]),
                    "action_l2_first20": float(action[:20].mean()),
                    "action_l2_valid_mean": float(action[valid].mean()) if valid.any() else None,
                    "action_l2_sustained_gt1_frame": frame_at(frames, action1),
                    "anchor_gt_10cm_frame": frame_at(frames, pos10),
                    "anchor_gt_20cm_frame": frame_at(frames, pos20),
                    "teacher_invalid_frame": frame_at(frames, lost),
                    "anchor_end_m": float(anchor[-1]),
                    "depth_finite_fraction": float(np.isfinite(data["depth"]).mean()),
                    "depth_std_mean": float(np.nanstd(data["depth"], axis=(1, 2)).mean()),
                    "depth_latency_mean_frames": float(data["depth_latency_frames"].mean()),
                    "depth_latency_max_frames": int(data["depth_latency_frames"].max()),
                    "depth_buffer_match_max": float(data["depth_buffer_match_error"].max()),
                }
                records.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({"records": records}, indent=2) + "\n")
    header = ("arm id motion end act_first act_mean act>1 pos>10cm pos>20cm invalid "
              "end_err depth_std latency")
    print(header)
    for r in records:
        print(r["arm"], r["motion_id"], r["motion"], r["end_frame"],
              f'{r["action_l2_first20"]:.3f}', f'{r["action_l2_valid_mean"]:.3f}',
              r["action_l2_sustained_gt1_frame"], r["anchor_gt_10cm_frame"],
              r["anchor_gt_20cm_frame"], r["teacher_invalid_frame"],
              f'{r["anchor_end_m"]:.3f}', f'{r["depth_std_mean"]:.3f}',
              f'{r["depth_latency_mean_frames"]:.2f}/{r["depth_latency_max_frames"]}')
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
