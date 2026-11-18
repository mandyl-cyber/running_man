"""Validate and summarize the paired no-takeover pre-contact DAgger A/B."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--episodes", type=int, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    args = parser.parse_args()

    summary = {"contract": "precontact_dagger_ab_no_takeover_clean", "seeds": args.seeds,
               "episodes_per_clip_per_seed": args.episodes, "arms": {}}
    reference_clips = None
    for arm in ("source", "baseline", "focus"):
        total = successes = 0
        clips = None
        per_seed = {}
        for seed in args.seeds:
            path = args.root / "full_eval" / arm / f"seed{seed}" / f"full_clip_seed{seed}.json"
            data = json.loads(path.read_text())
            rows = data["motions"]
            if (not data.get("complete") or data.get("profile") != "clean"
                    or data.get("intervention", {}).get("mode") != "none"
                    or data.get("obstacle_seed") != seed
                    or any(row["episodes"] != args.episodes for row in rows)):
                raise RuntimeError(f"Incomplete or mismatched report: {path}")
            names = [row["motion"] for row in rows]
            if clips is None:
                clips = names
            elif names != clips:
                raise RuntimeError(f"Motion order mismatch: {path}")
            n = sum(row["episodes"] for row in rows)
            s = sum(row["successes"] for row in rows)
            total += n
            successes += s
            per_seed[str(seed)] = {"successes": s, "episodes": n, "success_rate": s / n}
        if reference_clips is None:
            reference_clips = clips
        elif clips != reference_clips:
            raise RuntimeError(f"Motion order mismatch across arms: {arm}")
        summary["arms"][arm] = {"successes": successes, "episodes": total,
                                "success_rate": successes / total, "per_seed": per_seed}
        print(f"{arm}: {successes}/{total} ({successes / total:.1%})")

    baseline = summary["arms"]["baseline"]["success_rate"]
    focus = summary["arms"]["focus"]["success_rate"]
    source = summary["arms"]["source"]["success_rate"]
    summary["decision"] = "PASS" if focus > baseline and focus >= source else "STOP"
    path = args.root / "full_eval_summary.json"
    path.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"[DECISION] {summary['decision']} (focus must beat baseline and not regress vs source)")
    print(f"[REPORT] {path}")


if __name__ == "__main__":
    main()
