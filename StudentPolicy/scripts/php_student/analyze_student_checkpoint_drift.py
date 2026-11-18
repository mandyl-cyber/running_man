#!/usr/bin/env python3
"""Compare Student checkpoints on identical recorded states around first divergence.

This is an offline action diagnostic, not a closed-loop success evaluation.
Only frames with an exactly aligned recorded depth image are used.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch

from probe_student_temporal_observability import load_policy


def first_divergence(data: np.lib.npyio.NpzFile, *, threshold: float = 1.0, sustain: int = 10) -> int:
    difference = np.linalg.norm(data["student_action"] - data["teacher_action"], axis=1)
    above = (difference > threshold) & data["teacher_valid"].astype(bool)
    for index in range(len(above) - sustain + 1):
        if bool(above[index:index + sustain].all()):
            return int(data["frame"][index])
    raise RuntimeError("No sustained, teacher-valid action divergence found")


def aligned_trace(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as data:
        frame = np.asarray(data["frame"], dtype=np.int64)
        depth_frame = np.asarray(data["depth_frame"], dtype=np.int64)
        if len(np.unique(frame)) != len(frame) or np.any(np.diff(frame) <= 0):
            raise RuntimeError(f"Invalid trace frame order: {path}")
        lookup = {int(value): index for index, value in enumerate(frame)}
        if len(np.unique(depth_frame)) != len(depth_frame) or any(int(value) not in lookup for value in depth_frame):
            raise RuntimeError(f"Depth frames are not exactly aligned: {path}")
        rows = np.asarray([lookup[int(value)] for value in depth_frame])
        return {
            "frame": depth_frame,
            "actor": np.asarray(data["actor_obs"][rows], dtype=np.float32),
            "depth": np.asarray(data["depth"], dtype=np.float32),
            "teacher_action": np.asarray(data["teacher_action"][rows], dtype=np.float32),
            "teacher_valid": np.asarray(data["teacher_valid"][rows], dtype=bool),
            "recorded_student_action": np.asarray(data["student_action"][rows], dtype=np.float32),
        }


def predict(policy, trace: dict[str, np.ndarray]) -> np.ndarray:
    output = []
    with torch.inference_mode():
        for start in range(0, len(trace["frame"]), 128):
            output.append(policy.act_inference({
                "actor_obs": torch.from_numpy(trace["actor"][start:start + 128]),
                "depth_obs": torch.from_numpy(trace["depth"][start:start + 128]),
            }).cpu().numpy())
    return np.concatenate(output)


def window_name(frame: int, divergence: int) -> str:
    offset = frame - divergence
    if -80 <= offset < -20:
        return "preparation_-80_-20"
    if -20 <= offset < 0:
        return "pre_divergence_-20_0"
    if 0 <= offset < 20:
        return "first_divergence_0_20"
    if 20 <= offset < 80:
        return "after_divergence_20_80"
    return "outside"


def summarize(rows: list[dict], cohort: str, clip: int, window: str) -> dict:
    selected = [row for row in rows if row["cohort"] == cohort and row["clip_id"] == clip
                and row["window"] == window and row["teacher_valid"]]
    result = {"cohort": cohort, "clip_id": clip, "window": window, "valid_depth_frames": len(selected)}
    if selected:
        for key in ("source_l2", "step25_l2", "step50_l2", "source_to_step25_action_l2",
                    "step25_to_step50_action_l2"):
            result[key] = float(np.mean([row[key] for row in selected]))
        result["step50_minus_source_l2"] = result["step50_l2"] - result["source_l2"]
        result["step50_minus_step25_l2"] = result["step50_l2"] - result["step25_l2"]
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--step25", type=Path, required=True)
    parser.add_argument("--step50", type=Path, required=True)
    parser.add_argument("--student-traces", type=Path, required=True)
    parser.add_argument("--teacher-traces", type=Path, required=True)
    parser.add_argument("--clips", type=int, nargs="+", default=[4, 12, 24])
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error(f"Refusing to reuse output directory: {args.output_dir}")
    for path in (args.source, args.step25, args.step50):
        if not path.is_file():
            parser.error(f"Missing checkpoint: {path}")
    torch.set_num_threads(4)
    policies = {name: load_policy(path, torch.device("cpu")) for name, path in
                (("source", args.source), ("step25", args.step25), ("step50", args.step50))}
    rows: list[dict] = []
    divergences = {}
    fidelity = {}
    for clip in args.clips:
        student_path = args.student_traces / f"motion_{clip:02d}.npz"
        with np.load(student_path) as data:
            divergence = first_divergence(data)
        divergences[str(clip)] = divergence
        for cohort, trace_dir in (("student", args.student_traces), ("teacher", args.teacher_traces)):
            trace = aligned_trace(trace_dir / f"motion_{clip:02d}.npz")
            predictions = {name: predict(policy, trace) for name, policy in policies.items()}
            max_error = float(np.max(np.abs(predictions["source"] - trace["recorded_student_action"])))
            if max_error > 1e-3:
                raise RuntimeError(f"Source action does not reproduce recorded {cohort} clip {clip}: {max_error}")
            fidelity[f"{cohort}_{clip}"] = max_error
            errors = {name: np.linalg.norm(pred - trace["teacher_action"], axis=1)
                      for name, pred in predictions.items()}
            shift_25 = np.linalg.norm(predictions["step25"] - predictions["source"], axis=1)
            shift_50 = np.linalg.norm(predictions["step50"] - predictions["step25"], axis=1)
            for index, frame in enumerate(trace["frame"]):
                rows.append({
                    "cohort": cohort, "clip_id": clip, "frame": int(frame),
                    "divergence_frame": divergence, "window": window_name(int(frame), divergence),
                    "teacher_valid": bool(trace["teacher_valid"][index]),
                    "source_l2": float(errors["source"][index]),
                    "step25_l2": float(errors["step25"][index]),
                    "step50_l2": float(errors["step50"][index]),
                    "source_to_step25_action_l2": float(shift_25[index]),
                    "step25_to_step50_action_l2": float(shift_50[index]),
                })
    windows = ("preparation_-80_-20", "pre_divergence_-20_0",
               "first_divergence_0_20", "after_divergence_20_80")
    groups = [summarize(rows, cohort, clip, window) for cohort in ("student", "teacher")
              for clip in args.clips for window in windows]
    report = {
        "contract": "fixed_observation_student_checkpoint_action_comparison",
        "warning": "Offline same-state actions; Teacher-driven and Student-driven cohorts are different state distributions. Not a closed-loop success estimate.",
        "source": str(args.source.resolve()), "step25": str(args.step25.resolve()),
        "step50": str(args.step50.resolve()), "student_traces": str(args.student_traces.resolve()),
        "teacher_traces": str(args.teacher_traces.resolve()), "divergence_frames": divergences,
        "source_replay_max_abs_error": fidelity, "groups": groups,
    }
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output_dir / "per_frame.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for group in groups:
        if group["valid_depth_frames"] and group["window"] in windows[:2]:
            print(f"{group['cohort']} clip={group['clip_id']} {group['window']} n={group['valid_depth_frames']} "
                  f"L2 source/25/50={group['source_l2']:.3f}/{group['step25_l2']:.3f}/{group['step50_l2']:.3f}")
    print(f"[PASS] {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
