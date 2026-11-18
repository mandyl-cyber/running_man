"""Compare action disagreement on Teacher-controlled versus Student-controlled states."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


WINDOWS = ((0, 100), (100, 200), (200, 300), (300, 500), (500, 800))


def summarize(trace_dir: Path) -> dict:
    names = [line.strip() for line in (trace_dir.parent / "motion_names.txt").read_text().splitlines() if line.strip()]
    motions = []
    for path in sorted(trace_dir.glob("motion_*.npz")):
        motion_id = int(path.stem.split("_")[-1])
        with np.load(path) as data:
            frames = data["frame"].astype(int)
            valid = data["teacher_valid"].astype(bool)
            difference = np.abs(data["student_action"] - data["teacher_action"])
            l2 = np.linalg.norm(difference, axis=1)
            windows = {}
            for lo, hi in WINDOWS:
                mask = (frames >= lo) & (frames < hi) & valid
                mean_abs = difference[mask].mean(axis=0) if mask.any() else None
                windows[f"{lo}-{hi}"] = {
                    "valid_frames": int(mask.sum()),
                    "action_l2_mean": float(l2[mask].mean()) if mask.any() else None,
                    "action_l2_p90": float(np.quantile(l2[mask], 0.9)) if mask.any() else None,
                    "top_action_columns_by_mean_abs": (
                        [{"column": int(i), "mean_abs": float(mean_abs[i])}
                         for i in np.argsort(-mean_abs)[:5]] if mean_abs is not None else []
                    ),
                }
            motions.append({
                "motion_id": motion_id,
                "motion": names[motion_id],
                "recorded_frames": len(frames),
                "first_expert_lost_frame": int(frames[np.flatnonzero(~valid)[0]]) if (~valid).any() else None,
                "windows": windows,
            })
    return {"trace_dir": str(trace_dir), "motions": motions}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-traces", type=Path, required=True)
    parser.add_argument("--student-traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error(f"Refusing to overwrite {args.output}")
    result = {
        "contract": "same_state_action_outputs_within_each_rollout; controls_differ_between_rollouts",
        "teacher_control": summarize(args.teacher_traces),
        "student_control": summarize(args.student_traces),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[PASS] {args.output}")
    for mode in ("teacher_control", "student_control"):
        print(mode)
        for row in result[mode]["motions"]:
            print(row["motion"], "200-300 L2=", row["windows"]["200-300"]["action_l2_mean"],
                  "first lost=", row["first_expert_lost_frame"])


if __name__ == "__main__":
    main()
