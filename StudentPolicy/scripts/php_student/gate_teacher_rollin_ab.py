"""Conservative mechanism gate for paired Student/Teacher roll-in probes."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def fraction(rows: list[dict]) -> float:
    total = sum(row["samples"] for row in rows)
    if total == 0:
        raise ValueError("No samples in the target phase")
    return sum(row["valid_labels"] for row in rows) / total


def analyze(student: dict, teacher: dict, min_delta: float, min_clip_delta: float,
            min_improved_clips: int, min_samples_per_clip: int) -> dict:
    if student["window"] != teacher["window"]:
        raise ValueError("Intervention windows differ")
    if len(student["clip_phase_counts"]) != len(teacher["clip_phase_counts"]):
        raise ValueError("Motion bundles differ")
    if sum(row["teacher_control"] for row in student["counts"]):
        raise ValueError("Student control arm unexpectedly contains Teacher takeover")
    teacher_control = sum(row["teacher_control"] for row in teacher["counts"])
    if teacher_control == 0:
        raise ValueError("No Teacher takeover occurred")
    if teacher["counts"][2]["teacher_control"] == 0:
        raise ValueError("No Teacher takeover occurred in the 40-60% failure phase")
    phases = []
    for i, name in enumerate(student["phase_bins"]):
        a, b = student["counts"][i], teacher["counts"][i]
        phases.append({"phase": name, "student_valid": fraction([a]),
                       "teacher_valid": fraction([b]),
                       "delta": fraction([b]) - fraction([a]),
                       "teacher_control": b["teacher_control"]})
    focus_indices = (2,)  # 40-60% contains 113/120 source Student failures.
    a_focus = [student["counts"][i] for i in focus_indices]
    b_focus = [teacher["counts"][i] for i in focus_indices]
    overall_delta = fraction(b_focus) - fraction(a_focus)
    per_clip = []
    for a, b in zip(student["clip_phase_counts"], teacher["clip_phase_counts"], strict=True):
        if a["clip_id"] != b["clip_id"]:
            raise ValueError("Clip ordering differs")
        a_rows = [a["phase_bins"][i] for i in focus_indices]
        b_rows = [b["phase_bins"][i] for i in focus_indices]
        n_a = sum(row["samples"] for row in a_rows)
        n_b = sum(row["samples"] for row in b_rows)
        eligible = min(n_a, n_b) >= min_samples_per_clip
        fa = fraction(a_rows) if n_a else None
        fb = fraction(b_rows) if n_b else None
        per_clip.append({"clip_id": a["clip_id"], "student_samples": n_a,
                         "teacher_samples": n_b, "student_valid": fa,
                         "teacher_valid": fb,
                         "delta": fb - fa if fa is not None and fb is not None else None,
                         "eligible": eligible})
    improved = sum(row["eligible"] and row["delta"] >= min_clip_delta for row in per_clip)
    eligible = sum(row["eligible"] for row in per_clip)
    reasons = []
    if overall_delta < min_delta:
        reasons.append(f"pooled 40-60% gain {overall_delta:.3%} < {min_delta:.1%}")
    required_clips = max(min_improved_clips, math.ceil(eligible * 0.25))
    if improved < required_clips:
        reasons.append(f"improved clips {improved} < {required_clips}")
    if eligible < min_improved_clips:
        reasons.append(f"eligible clips {eligible} < {min_improved_clips}")
    return {"student_valid_40_60": fraction(a_focus),
            "teacher_valid_40_60": fraction(b_focus), "delta": overall_delta,
            "teacher_control_steps": teacher_control, "eligible_clips": eligible,
            "improved_clips": improved, "thresholds": {
                "min_pooled_delta": min_delta, "min_clip_delta": min_clip_delta,
                "min_improved_clips": required_clips, "min_improved_clip_fraction": 0.25,
                "min_samples_per_clip": min_samples_per_clip},
            "phases": phases, "per_clip": per_clip,
            "gate_pass": not reasons, "reasons": reasons}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--student", type=Path, required=True)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--min-delta", type=float, default=0.05)
    p.add_argument("--min-clip-delta", type=float, default=0.05)
    p.add_argument("--min-improved-clips", type=int, default=5)
    p.add_argument("--min-samples-per-clip", type=int, default=100)
    args = p.parse_args()
    result = analyze(json.loads(args.student.read_text()), json.loads(args.teacher.read_text()),
                     args.min_delta, args.min_clip_delta,
                     args.min_improved_clips, args.min_samples_per_clip)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[PROBE] 40–60% valid labels: Student={result['student_valid_40_60']:.1%}, "
          f"roll-in={result['teacher_valid_40_60']:.1%}, delta={result['delta']:+.1%}")
    print(f"[PROBE] improved clips={result['improved_clips']}/{result['eligible_clips']}, "
          f"Teacher-controlled steps={result['teacher_control_steps']}")
    for phase in result["phases"]:
        print(f"[PHASE {phase['phase']}] delta={phase['delta']:+.1%}, "
              f"Teacher control={phase['teacher_control']}")
    if result["reasons"]:
        raise SystemExit("[STOP] " + "; ".join(result["reasons"]))
    print("[PASS] Mechanism gate; short continuation is justified.")


if __name__ == "__main__":
    main()
