#!/usr/bin/env python3
"""Measure checkpoint action drift on identical, depth-aligned Student states.

This is an offline mechanism check, not a closed-loop success estimate.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from analyze_student_checkpoint_drift import aligned_trace, predict
from probe_student_temporal_observability import load_policy


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--ordinary", type=Path, required=True)
    parser.add_argument("--anchor", type=Path, required=True)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    for path in (args.source, args.ordinary, args.anchor):
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
    paths = sorted(args.traces.glob("motion_??.npz"))
    if len(paths) != 40:
        parser.error(f"Expected 40 trace files, found {len(paths)} in {args.traces}")
    if args.output.exists():
        parser.error(f"Refusing to overwrite report: {args.output}")

    torch.set_num_threads(4)
    policies = {name: load_policy(path, torch.device("cpu")) for name, path in
                (("source", args.source), ("ordinary", args.ordinary), ("anchor", args.anchor))}
    distances = {window: {name: [] for name in ("ordinary", "anchor")}
                 for window in ("prefix_0_99", "after_100_199", "after_200")}
    per_clip = []
    max_replay_error = 0.0
    for path in paths:
        trace = aligned_trace(path)
        predictions = {name: predict(policy, trace) for name, policy in policies.items()}
        replay_error = float(np.max(np.abs(predictions["source"] - trace["recorded_student_action"])))
        if replay_error > 1e-3:
            raise RuntimeError(f"Source replay mismatch for {path}: {replay_error}")
        max_replay_error = max(max_replay_error, replay_error)
        frame = trace["frame"]
        clip = {"motion": path.stem, "aligned_frames": len(frame), "source_replay_max_abs": replay_error}
        for window, mask in (("prefix_0_99", frame < 100),
                             ("after_100_199", (frame >= 100) & (frame < 200)),
                             ("after_200", frame >= 200)):
            clip[window] = {"frames": int(mask.sum())}
            for name in ("ordinary", "anchor"):
                values = np.linalg.norm(predictions[name][mask] - predictions["source"][mask], axis=1)
                clip[window][name + "_mean_action_l2"] = float(values.mean()) if len(values) else None
                distances[window][name].extend(values.tolist())
        per_clip.append(clip)

    summary = {}
    for window, group in distances.items():
        ordinary = float(np.mean(group["ordinary"])) if group["ordinary"] else None
        anchor = float(np.mean(group["anchor"])) if group["anchor"] else None
        summary[window] = {"frames": len(group["ordinary"]), "ordinary_mean_action_l2": ordinary,
                           "anchor_mean_action_l2": anchor,
                           "anchor_over_ordinary": anchor / ordinary if ordinary and anchor is not None else None}
    report = {"contract": "same_state_depth_aligned_early_action_preservation",
              "warning": "Offline action comparison on source Student states; not a success-rate estimate.",
              "checkpoints": {name: str(path.resolve()) for name, path in
                              (("source", args.source), ("ordinary", args.ordinary), ("anchor", args.anchor))},
              "traces": str(args.traces.resolve()), "source_replay_max_abs": max_replay_error,
              "summary": summary, "per_clip": per_clip}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
