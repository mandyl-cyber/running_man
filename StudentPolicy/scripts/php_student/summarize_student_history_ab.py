"""Compare paired full-clip Student evaluations, refusing incomplete reports."""

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()
    results = {}
    for arm in ("single", "history3"):
        path = args.root / "eval" / arm / f"full_clip_seed{args.seed}.json"
        if not path.is_file():
            raise SystemExit(f"[PENDING] {arm}: {path}")
        data = json.loads(path.read_text())
        if not data.get("complete") or data.get("intervention", {}).get("mode") != "none":
            raise SystemExit(f"[ERROR] Incomplete or takeover-contaminated report: {path}")
        rows = data["motions"]
        episodes = sum(row["episodes"] for row in rows)
        successes = sum(row["successes"] for row in rows)
        results[arm] = {row["motion"]: row for row in rows}
        print(f"[{arm}] {successes}/{episodes} ({successes / episodes:.1%}) across {len(rows)} clips")
    a, b = results["single"], results["history3"]
    if a.keys() != b.keys():
        raise SystemExit("[ERROR] Motion sets differ")
    wins = sum(b[name]["successes"] > a[name]["successes"] for name in a)
    losses = sum(b[name]["successes"] < a[name]["successes"] for name in a)
    print(f"[A/B] history3 wins={wins}, losses={losses}, ties={len(a) - wins - losses}")
    for name in a:
        if b[name]["successes"] != a[name]["successes"]:
            print(f"  {name}: {a[name]['successes']}/{a[name]['episodes']} -> "
                  f"{b[name]['successes']}/{b[name]['episodes']}")


if __name__ == "__main__":
    main()
