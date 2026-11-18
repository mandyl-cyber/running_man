#!/usr/bin/env python3
"""Summarize paired, frame-0 intervention evaluations without claiming Student success."""

import argparse
import json
from pathlib import Path


def read_report(path: Path, seed: int) -> dict:
    report = json.loads(path.read_text())
    if not report.get("complete") or report.get("obstacle_seed") != seed:
        raise ValueError(f"Incomplete or mismatched report: {path}")
    if not report.get("diagnostics_enabled"):
        raise ValueError(f"Diagnostics missing: {path}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    root = args.root
    baseline = read_report(root / "baseline" / f"full_clip_seed{args.seed}.json", args.seed)
    baseline_rows = {row["motion"]: row for row in baseline["motions"]}
    if baseline["intervention"]["mode"] != "none":
        raise ValueError("Baseline has Teacher intervention")
    if any(row["episodes"] != 1 for row in baseline_rows.values()):
        raise ValueError("Expected one paired episode per motion")
    print("condition | wins | losses | ties | assisted successes | takeover steps mean")
    for path in sorted((root / "eval").glob("*/full_clip_seed*.json")):
        if path.stat().st_size == 0:
            print(f"[RUNNING] {path.parent.name}: report not written yet")
            continue
        report = read_report(path, args.seed)
        intervention = report["intervention"]
        if intervention["mode"] not in {"mapped_window", "mapped_tail"}:
            raise ValueError(f"Unexpected intervention: {path}")
        rows = {row["motion"]: row for row in report["motions"]}
        if set(rows) != set(baseline_rows) or any(row["episodes"] != 1 for row in rows.values()):
            raise ValueError(f"Motion/episode mismatch: {path}")
        wins = sum(rows[name]["successes"] > baseline_rows[name]["successes"] for name in rows)
        losses = sum(rows[name]["successes"] < baseline_rows[name]["successes"] for name in rows)
        assisted = sum(row["successes"] for row in rows.values())
        counts = [row["diagnostic"]["teacher_takeover_steps_mean"] for row in rows.values()]
        if any(value is None for value in counts):
            raise ValueError(f"Missing takeover counts: {path}")
        print(f"{path.parent.name} | {wins} | {losses} | {len(rows)-wins-losses} | "
              f"{assisted}/{len(rows)} | {sum(counts)/len(counts):.1f}")
    print("All successes above are Teacher-assisted; none are deployable Student success.")


if __name__ == "__main__":
    main()
