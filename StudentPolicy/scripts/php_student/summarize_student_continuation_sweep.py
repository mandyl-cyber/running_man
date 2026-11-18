"""Validate and summarize short deterministic continuation checkpoint evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def result(path: Path, seed: int) -> dict:
    data = json.loads(path.read_text())
    rows = data["motions"]
    if (not data.get("complete") or len(rows) != 40
            or data.get("profile") != "clean"
            or data.get("intervention", {}).get("mode") != "none"
            or data.get("obstacle_seed") != seed
            or any(row["episodes"] != 1 for row in rows)):
        raise RuntimeError(f"Incomplete or mismatched report: {path}")
    return {"successes": sum(row["successes"] for row in rows),
            "episodes": 40, "successful_motions": [row["motion"] for row in rows if row["successes"]]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[2] / "reports/php_student/precontact_dagger_ab_v1/eval/source_seed43/full_clip_seed43.json"
    if args.seed != 43:
        raise RuntimeError("Only seed 43 has a paired source report; create that source report first")
    rows = {"source": result(source, args.seed)}
    rows["loaded_only"] = result(args.root / "eval" / f"load_only_seed{args.seed}" / f"full_clip_seed{args.seed}.json", args.seed)
    for step in (1, 5, 10, 25, 50):
        key = f"step_{step:03d}"
        rows[key] = result(args.root / "eval" / f"step{step:03d}_seed{args.seed}" / f"full_clip_seed{args.seed}.json", args.seed)
    for key, value in rows.items():
        print(f"{key}: {value['successes']}/40 {value['successful_motions']}")
    initial = rows["source"]["successes"]
    rows["loaded_only"]["matches_source"] = rows["loaded_only"]["successful_motions"] == rows["source"]["successful_motions"]
    first_drop = next((step for step in (1, 5, 10, 25, 50)
                       if rows[f"step_{step:03d}"]["successes"] < initial), None)
    output = {"contract": "student_continuation_clean_seed43_one_episode_per_clip",
              "seed": args.seed, "results": rows, "first_observed_drop_step": first_drop}
    path = args.root / "sweep_summary.json"
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(f"[FIRST OBSERVED DROP] {first_drop}; report={path}")


if __name__ == "__main__":
    main()
