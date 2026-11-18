#!/usr/bin/env python3
"""Audit failures after a finite Teacher intervention using paired-action traces."""

import argparse
import json
from pathlib import Path

import numpy as np


def first_run(mask: np.ndarray, width: int) -> int | None:
    if len(mask) < width:
        return None
    hits = np.convolve(mask.astype(np.int32), np.ones(width, dtype=np.int32), mode="valid")
    found = np.flatnonzero(hits == width)
    return int(found[0]) if found.size else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.report.read_text())
    intervention = report["intervention"]
    if not report["complete"] or intervention["mode"] != "mapped_window":
        raise ValueError("Expected complete mapped_window evaluation")
    starts = intervention["start_map"]
    steps = int(intervention["steps"])
    rows = []
    for motion in report["motions"]:
        idx = motion["motion_id"]
        with np.load(args.trace_dir / f"motion_{idx:02d}.npz") as data:
            frames = data["frame"].astype(int)
            gap = np.linalg.norm(data["student_action"] - data["teacher_action"], axis=1)
            valid = data["teacher_valid"].astype(bool)
            anchor = np.linalg.norm(data["ref_pos_w"] - data["robot_ref_pos_w"], axis=1)
            release = int(starts[str(idx)]) + steps
            after = frames >= release
            post_frames = frames[after]
            post_gap = gap[after]
            post_valid = valid[after]
            onset = first_run(post_gap > 1.0, 10)
            invalid = np.flatnonzero(~post_valid)
            failure = motion["diagnostic"]["failure_frame_mean"]
            rows.append({
                "motion_id": idx,
                "motion": motion["motion"],
                "success": bool(motion["successes"]),
                "release_frame_planned": release,
                "observed_release": bool(after.any()),
                "terminal_frame": int(frames[-1]),
                "failure_frame": failure,
                "failure_frames_after_release": round(float(failure - release), 1) if failure is not None else None,
                "first_sustained_post_release_gap_frame": int(post_frames[onset]) if onset is not None else None,
                "first_post_release_invalid_frame": int(post_frames[invalid[0]]) if invalid.size else None,
                "post_release_anchor_error_start_m": float(anchor[after][0]) if after.any() else None,
                "post_release_valid_fraction": float(post_valid.mean()) if after.any() else None,
            })
    failures = [row for row in rows if not row["success"]]
    summary = {
        "count": len(rows),
        "successes": len(rows) - len(failures),
        "failures_after_planned_release": sum(
            row["failure_frames_after_release"] is not None and row["failure_frames_after_release"] >= 0
            for row in failures
        ),
        "failed_with_post_release_action_gap": sum(
            row["first_sustained_post_release_gap_frame"] is not None for row in failures
        ),
        "failed_with_post_release_label_invalid": sum(
            row["first_post_release_invalid_frame"] is not None for row in failures
        ),
        "failure_delay_median_frames": float(np.median([
            row["failure_frames_after_release"] for row in failures
            if row["failure_frames_after_release"] is not None
        ])) if failures else None,
    }
    payload = {"summary": summary, "motions": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
