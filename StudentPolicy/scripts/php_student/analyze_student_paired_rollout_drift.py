#!/usr/bin/env python3
"""Analyze where paired Student rollouts first separate in state and contact.

Different policies generate different states after their first action. This
script reports temporal associations, not a same-state causal action effect.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np


def first_sustained(frame: np.ndarray, condition: np.ndarray, count: int = 3) -> int | None:
    for start in range(len(condition) - count + 1):
        if bool(condition[start:start + count].all()):
            return int(frame[start])
    return None


def first_crossing(frame: np.ndarray, value: np.ndarray, threshold: float) -> int | None:
    indices = np.flatnonzero(value >= threshold)
    return int(frame[indices[0]]) if len(indices) else None


def load_arm(root: Path, arm: str, seed: int) -> tuple[list[dict], dict[int, dict[str, np.ndarray]]]:
    base = root / arm
    report = json.loads((base / f"full_clip_seed{seed}.json").read_text())
    rows = report["motions"]
    if (not report.get("complete") or report.get("profile") != "clean"
            or report.get("obstacle_seed") != seed
            or report.get("intervention", {}).get("mode") != "none"
            or len(rows) != 40 or any(row["episodes"] != 1 for row in rows)):
        raise RuntimeError(f"Invalid paired eval report: {base}")
    traces = {}
    for clip in range(40):
        path = base / "traces" / f"motion_{clip:02d}.npz"
        with np.load(path) as data:
            traces[clip] = {key: np.asarray(data[key]).copy() for key in (
                "frame", "robot_ref_pos_w", "ref_pos_w", "root_state", "root_state_post_step",
                "foot_contact_force", "student_action", "teacher_action", "teacher_valid",
                "actor_obs", "teacher_obs",
            )}
        frame = traces[clip]["frame"]
        # The evaluator records the command clock after reset; this run starts
        # at reference frame 1. Require contiguous frames, not a hard-coded 0.
        if len(frame) == 0 or frame[0] not in (0, 1) or not np.array_equal(frame, np.arange(frame[0], frame[0] + len(frame))):
            raise RuntimeError(f"Trace has an unexpected start frame or gaps: {path}")
        if int(rows[clip]["motion_id"]) != clip:
            raise RuntimeError(f"Report motion order mismatch: {base} clip {clip}")
    return rows, traces


def analyze_pair(source: dict[str, np.ndarray], candidate: dict[str, np.ndarray],
                 source_success: bool, candidate_success: bool, clip: int, arm: str,
                 reference_arm: str) -> tuple[dict, list[dict]]:
    if int(source["frame"][0]) != int(candidate["frame"][0]):
        raise RuntimeError(f"First recorded frame differs for clip {clip} arm {arm}")
    n = min(len(source["frame"]), len(candidate["frame"]))
    frame = source["frame"][:n]
    root_delta = np.linalg.norm(source["root_state_post_step"][:n, :3]
                                - candidate["root_state_post_step"][:n, :3], axis=1)
    anchor_delta = np.linalg.norm(source["robot_ref_pos_w"][:n]
                                  - candidate["robot_ref_pos_w"][:n], axis=1)
    action_delta = np.linalg.norm(source["student_action"][:n]
                                  - candidate["student_action"][:n], axis=1)
    source_contact = np.linalg.norm(source["foot_contact_force"][:n], axis=-1) >= 20.0
    candidate_contact = np.linalg.norm(candidate["foot_contact_force"][:n], axis=-1) >= 20.0
    contact_mismatch = np.any(source_contact != candidate_contact, axis=1)
    source_error = np.linalg.norm(source["robot_ref_pos_w"][:n] - source["ref_pos_w"][:n], axis=1)
    candidate_error = np.linalg.norm(candidate["robot_ref_pos_w"][:n] - candidate["ref_pos_w"][:n], axis=1)
    initial_root_delta = float(np.linalg.norm(source["root_state"][0, :3] - candidate["root_state"][0, :3]))
    initial_ref_delta = float(np.linalg.norm(source["ref_pos_w"][0] - candidate["ref_pos_w"][0]))
    initial_actor_max_abs = float(np.max(np.abs(source["actor_obs"][0] - candidate["actor_obs"][0])))
    initial_teacher_obs_max_abs = float(np.max(np.abs(source["teacher_obs"][0] - candidate["teacher_obs"][0])))
    row = {
        "clip_id": clip, "arm": arm, "reference_arm": reference_arm,
        "source_success": source_success,
        "candidate_success": candidate_success, "source_last_frame": int(source["frame"][-1]),
        "candidate_last_frame": int(candidate["frame"][-1]), "common_frames": n,
        "initial_root_delta_m": initial_root_delta, "initial_reference_delta_m": initial_ref_delta,
        "initial_actor_obs_max_abs": initial_actor_max_abs,
        "initial_teacher_obs_max_abs": initial_teacher_obs_max_abs,
        "initial_action_l2": float(action_delta[0]),
        "first_root_delta_2cm": first_sustained(frame, root_delta >= 0.02),
        "first_root_delta_5cm": first_sustained(frame, root_delta >= 0.05),
        "first_anchor_delta_2cm": first_sustained(frame, anchor_delta >= 0.02),
        "first_contact_mismatch": first_sustained(frame, contact_mismatch),
        "source_first_anchor_error_25cm": first_crossing(frame, source_error, 0.25),
        "candidate_first_anchor_error_25cm": first_crossing(frame, candidate_error, 0.25),
        "source_first_anchor_error_50cm": first_crossing(frame, source_error, 0.50),
        "candidate_first_anchor_error_50cm": first_crossing(frame, candidate_error, 0.50),
        "max_root_delta_m": float(root_delta.max()),
        "max_anchor_delta_m": float(anchor_delta.max()),
    }
    timeline = []
    for i, value in enumerate(frame):
        timeline.append({
            "clip_id": clip, "arm": arm, "frame": int(value),
            "root_delta_m": float(root_delta[i]), "anchor_delta_m": float(anchor_delta[i]),
            "action_delta_l2": float(action_delta[i]),
            "foot_contact_mismatch": bool(contact_mismatch[i]),
            "source_anchor_error_m": float(source_error[i]),
            "candidate_anchor_error_m": float(candidate_error[i]),
            "source_teacher_valid": bool(source["teacher_valid"][i]),
            "candidate_teacher_valid": bool(candidate["teacher_valid"][i]),
        })
    return row, timeline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    parser.add_argument("--output-stem", default="paired_drift")
    args = parser.parse_args()
    if (args.root / f"{args.output_stem}_summary.json").exists():
        parser.error("Refusing to overwrite existing paired drift summary")
    runs = {arm: load_arm(args.root, arm, args.seed) for arm in ("source", "step025", "step050")}
    names = [row["motion"] for row in runs["source"][0]]
    if any([row["motion"] for row in runs[arm][0]] != names for arm in ("step025", "step050")):
        raise RuntimeError("Motion order differs across paired runs")
    pairs, timeline = [], []
    for reference_arm, candidate_arm, arm in (
        ("source", "step025", "step025"),
        ("source", "step050", "step050"),
        ("step025", "step050", "step050_vs_step025"),
    ):
        for clip in range(40):
            source_rows, source_traces = runs[reference_arm]
            candidate_rows, candidate_traces = runs[candidate_arm]
            row, points = analyze_pair(
                source_traces[clip], candidate_traces[clip],
                bool(source_rows[clip]["successes"]), bool(candidate_rows[clip]["successes"]),
                clip, arm, reference_arm,
            )
            row["motion"] = names[clip]
            pairs.append(row)
            timeline.extend(points)
    summary = {
        "contract": "student_clean_frame0_seed43_paired_rollout_drift",
        "warning": "Same seed and initial-state checks, but post-action states differ; frame-aligned deltas are descriptive, not same-state causal effects.",
        "seed": args.seed,
        "successes": {arm: sum(row["successes"] for row in runs[arm][0]) for arm in runs},
        "initial_state_match_count_by_arm": {
            arm: sum(row["initial_root_delta_m"] < 1e-4 and row["initial_reference_delta_m"] < 1e-4
                     and row["initial_actor_obs_max_abs"] < 1e-4
                     and row["initial_teacher_obs_max_abs"] < 1e-4
                     for row in pairs if row["arm"] == arm)
            for arm in ("step025", "step050", "step050_vs_step025")
        },
        "pairs": pairs,
    }
    args.root.mkdir(parents=True, exist_ok=True)
    (args.root / f"{args.output_stem}_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    with (args.root / f"{args.output_stem}_timeline.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(timeline[0]))
        writer.writeheader()
        writer.writerows(timeline)
    print("[PAIRED] successes", summary["successes"],
          "initial matches", summary["initial_state_match_count_by_arm"])
    for row in pairs:
        if row["source_success"] != row["candidate_success"] or row["clip_id"] in (4, 12, 24):
            print(f"{row['arm']} clip={row['clip_id']:02d} success={int(row['source_success'])}→{int(row['candidate_success'])} "
                  f"last={row['source_last_frame']}/{row['candidate_last_frame']} "
                  f"root2cm={row['first_root_delta_2cm']} contact={row['first_contact_mismatch']} "
                  f"anchor2cm={row['first_anchor_delta_2cm']}")
    print(f"[PASS] {args.root / f'{args.output_stem}_summary.json'}")


if __name__ == "__main__":
    main()
