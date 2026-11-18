"""Summarize first-episode Student/Teacher traces from eval_student_full.py."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("trace_dir", type=Path)
    args = parser.parse_args()
    files = sorted(args.trace_dir.glob("motion_*.npz"))
    if not files:
        raise SystemExit(f"No traces in {args.trace_dir}")
    print("motion frames first_l2_gt1 pos_at_l2 first_lost lost_reason pos_at_lost pos_at_end depth_finite")
    for file in files:
        with np.load(file) as data:
            valid = data["teacher_valid"].astype(bool)
            if not valid[0] or valid.all():
                raise ValueError(f"Expected an initially valid teacher and later loss: {file}")
            lost = int(np.flatnonzero(~valid)[0])
            action_l2 = np.linalg.norm(data["student_action"] - data["teacher_action"], axis=1)
            anchor_error = np.linalg.norm(data["ref_pos_w"] - data["robot_ref_pos_w"], axis=1)
            rolling = np.convolve(action_l2[:lost], np.ones(20) / 20, mode="valid")
            candidates = np.flatnonzero(rolling > 1.0)
            onset = int(candidates[0]) if candidates.size else None
            reasons = []
            for key, label in (("expert_lost_ref_pos", "ref_pos"), ("expert_lost_ref_ori", "ref_ori"),
                               ("expert_lost_body_pos", "body_pos")):
                if bool(data[key][lost]):
                    reasons.append(label)
            pos_at_onset = f"{anchor_error[onset]:.3f}" if onset is not None else "na"
            print(
                f"{file.stem} {len(action_l2)} {onset if onset is not None else 'none'} "
                f"{pos_at_onset} {lost} {','.join(reasons) or 'unresolved'} "
                f"{anchor_error[lost]:.3f} {anchor_error[-1]:.3f} "
                f"{np.isfinite(data['depth']).mean():.4f}"
            )


if __name__ == "__main__":
    main()
