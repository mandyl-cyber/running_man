#!/usr/bin/env python3
"""Join checkpoint curriculum state with deterministic full-clip results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--eval-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for report_path in sorted(args.eval_root.glob("model_*/full_clip_seed*.json")):
        checkpoint = args.train_dir / f"{report_path.parent.name}.pt"
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        report = json.loads(report_path.read_text())
        if not report.get("complete"):
            raise RuntimeError(f"Incomplete report: {report_path}")
        state = torch.load(checkpoint, map_location="cpu", weights_only=False)
        motions = report["motions"]
        episodes = sum(row["episodes"] for row in motions)
        successes = sum(row["successes"] for row in motions)
        failures = [row for row in motions if row["successes"] < row["episodes"]]
        failure_progress = [
            row.get("diagnostic", {}).get("failure_progress_mean")
            for row in failures
        ]
        failure_progress = [value for value in failure_progress if value is not None]
        rows.append({
            "checkpoint": str(checkpoint.resolve()),
            "iteration": int(state.get("iter", state.get("iteration", -1))),
            "ppo_coef": float(state.get("ppo_coef", float("nan"))),
            "learning_rate": float(state.get("learning_rate", float("nan"))),
            "successes": successes,
            "episodes": episodes,
            "success_rate": successes / episodes,
            "passing_clips": sum(row["successes"] == row["episodes"] for row in motions),
            "motion_count": len(motions),
            "mean_failure_progress": (
                sum(failure_progress) / len(failure_progress) if failure_progress else None
            ),
            "report": str(report_path.resolve()),
        })

    if not rows:
        raise RuntimeError(f"No evaluation reports under {args.eval_root}")
    ranked = sorted(
        rows,
        key=lambda row: (
            row["success_rate"],
            row["mean_failure_progress"] if row["mean_failure_progress"] is not None else 1.0,
        ),
        reverse=True,
    )
    payload = {
        "contract": "same-run checkpoint sweep; deterministic frame-0 clean full clips",
        "best_checkpoint": ranked[0]["checkpoint"],
        "best_iteration": ranked[0]["iteration"],
        "ranking_basis": (
            "success_rate, then mean_failure_progress; best does not imply any "
            "successful full clip"
        ),
        "any_full_clip_success": any(row["successes"] > 0 for row in rows),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    print("iteration | ppo_coef | lr | success | clips | failure_progress")
    for row in rows:
        progress = row["mean_failure_progress"]
        progress_text = f"{progress:.3f}" if progress is not None else "n/a"
        print(
            f"{row['iteration']:9d} | {row['ppo_coef']:.3f} | {row['learning_rate']:.1e} | "
            f"{row['successes']:3d}/{row['episodes']:<3d} | "
            f"{row['passing_clips']:2d}/{row['motion_count']:<2d} | {progress_text}"
        )
    label = "BEST_SUCCESS" if payload["any_full_clip_success"] else "FARTHEST_FAILURE"
    print(f"[{label}] iteration={ranked[0]['iteration']} checkpoint={ranked[0]['checkpoint']}")
    print(f"[PASS] {args.output}")


if __name__ == "__main__":
    main()
