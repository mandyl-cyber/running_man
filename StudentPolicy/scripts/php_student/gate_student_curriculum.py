"""Gate a Student curriculum stage using paired quick-evaluation reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(root: Path, seeds: list[int]) -> dict:
    rows = {}
    for seed in seeds:
        path = root / f"seed{seed}" / f"full_clip_seed{seed}.json"
        data = json.loads(path.read_text())
        if not data.get("complete") or data.get("intervention", {}).get("mode") != "none":
            raise RuntimeError(f"Incomplete or intervention-contaminated report: {path}")
        for row in data["motions"]:
            key = (seed, row["motion"])
            rows[key] = row
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--incumbent", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--seeds", type=int, nargs="+", default=[43, 44])
    p.add_argument("--min-success-gain", type=int, default=2)
    args = p.parse_args()
    if args.output.exists():
        p.error(f"Refusing to overwrite {args.output}")
    incumbent = load(args.incumbent, args.seeds)
    candidate = load(args.candidate, args.seeds)
    if incumbent.keys() != candidate.keys():
        raise RuntimeError("Paired evaluation motion/seed sets differ")
    old_success = sum(r["successes"] for r in incumbent.values())
    new_success = sum(r["successes"] for r in candidate.values())
    wins = sum(candidate[k]["successes"] > incumbent[k]["successes"] for k in incumbent)
    losses = sum(candidate[k]["successes"] < incumbent[k]["successes"] for k in incumbent)
    old_progress = [r["diagnostic"]["failure_progress_mean"] for r in incumbent.values()
                    if r.get("diagnostic", {}).get("failure_progress_mean") is not None]
    new_progress = [candidate[k]["diagnostic"]["failure_progress_mean"] for k in incumbent
                    if candidate[k].get("diagnostic", {}).get("failure_progress_mean") is not None]
    old_mean = sum(old_progress) / len(old_progress)
    new_mean = sum(new_progress) / len(new_progress)
    delta = new_success - old_success
    progress_delta = new_mean - old_mean
    if delta >= args.min_success_gain and wins > losses and progress_delta >= -0.02:
        decision = "PASS"
    elif delta <= -args.min_success_gain or (losses >= wins + 3 and progress_delta < -0.02):
        decision = "REGRESS"
    else:
        decision = "HOLD"
    result = {
        "decision": decision, "seeds": args.seeds,
        "incumbent_successes": old_success, "candidate_successes": new_success,
        "success_delta": delta, "clip_seed_wins": wins, "clip_seed_losses": losses,
        "clip_seed_ties": len(incumbent) - wins - losses,
        "incumbent_failure_progress_mean": old_mean,
        "candidate_failure_progress_mean": new_mean,
        "failure_progress_delta": progress_delta,
        "pass_contract": {
            "min_success_gain": args.min_success_gain,
            "wins_must_exceed_losses": True,
            "minimum_failure_progress_delta": -0.02,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
