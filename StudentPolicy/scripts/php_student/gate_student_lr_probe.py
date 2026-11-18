#!/usr/bin/env python3
"""Gate a learning-rate curriculum probe against the original same-seed run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def summarize(path: Path) -> dict:
    data = json.loads(path.read_text())
    if not data.get("complete"):
        raise RuntimeError(f"Incomplete report: {path}")
    rows = data["motions"]
    episodes = sum(row["episodes"] for row in rows)
    successes = sum(row["successes"] for row in rows)
    progress = [
        row.get("diagnostic", {}).get("failure_progress_mean")
        for row in rows if row["successes"] < row["episodes"]
    ]
    progress = [value for value in progress if value is not None]
    return {
        "successes": successes,
        "episodes": episodes,
        "success_rate": successes / episodes,
        "failure_progress": sum(progress) / len(progress) if progress else 1.0,
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-root", type=Path, required=True)
    p.add_argument("--candidate-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    checkpoints = ("00500", "01000", "02500")
    rows = []
    for checkpoint in checkpoints:
        base = summarize(args.baseline_root / f"model_{checkpoint}" / "full_clip_seed43.json")
        cand = summarize(args.candidate_root / f"model_{checkpoint}" / "full_clip_seed43.json")
        rows.append({
            "iteration": int(checkpoint), "baseline": base, "candidate": cand,
            "success_delta": cand["success_rate"] - base["success_rate"],
            "failure_progress_delta": cand["failure_progress"] - base["failure_progress"],
        })
    final = rows[-1]
    passed = (
        final["candidate"]["successes"] > final["baseline"]["successes"]
        or final["failure_progress_delta"] >= 0.03
    ) and rows[1]["failure_progress_delta"] >= -0.02
    payload = {
        "contract": "same Teacher/data/seed/config; desired_kl 0.01 vs 0.03",
        "decision": "PASS" if passed else "STOP",
        "rule": "at 2500: any success gain or failure progress +0.03; no >0.02 regression at 1000",
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    for row in rows:
        print(
            f"iter={row['iteration']:4d} base={row['baseline']['successes']}/{row['baseline']['episodes']} "
            f"p={row['baseline']['failure_progress']:.3f} candidate="
            f"{row['candidate']['successes']}/{row['candidate']['episodes']} "
            f"p={row['candidate']['failure_progress']:.3f} "
            f"delta={row['failure_progress_delta']:+.3f}"
        )
    print(f"[{payload['decision']}] {args.output}")


if __name__ == "__main__":
    main()
