"""Summarize the three Student structural probes without conflating their evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def full_clip(path: Path) -> dict:
    report = json.loads(path.read_text())
    if not report.get("complete") or not report.get("motions"):
        raise ValueError(f"Incomplete full-clip report: {path}")
    rows = report["motions"]
    return {"path": str(path.resolve()), "motions": len(rows),
            "episodes": sum(r["episodes"] for r in rows),
            "successes": sum(r["successes"] for r in rows)}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--source-traces", type=Path, required=True)
    a = p.parse_args()
    root = a.root
    result: dict = {"contract": "student_structural_tests_v1", "tests": {}}
    single = root / "single_clip"
    if (single / "baseline/full_clip_seed43.json").is_file():
        row = {"baseline": full_clip(single / "baseline/full_clip_seed43.json")}
        if (single / "teacher/full_clip_seed43.json").is_file():
            row["teacher"] = full_clip(single / "teacher/full_clip_seed43.json")
        if (single / "specialized/full_clip_seed43.json").is_file():
            row["specialized"] = full_clip(single / "specialized/full_clip_seed43.json")
        result["tests"]["single_clip_specialization"] = row
    obs = root / "observability/observation_vs_state.json"
    if obs.is_file():
        data = json.loads(obs.read_text())
        result["tests"]["observation_ceiling"] = {
            "samples": data["samples"], "clips": len(data["clips"]),
            "heldout_clip_mean_l2": data["means"],
            "within_clip_block_l2": {
                k: v["heldout_l2"] for k, v in data["within_clip_contiguous_block"]["results"].items()
            },
            "warning": "Offline label fit is not closed-loop success; privileged observations are an oracle diagnostic.",
        }
        trainable = root / "observability/trainable_depth.json"
        if trainable.is_file():
            fitted = json.loads(trainable.read_text())
            result["tests"]["observation_ceiling"]["trainable_depth_grouped_cv_l2"] = fitted["means"]
    recovery = root / "recovery"
    base = recovery / "baseline/full_clip_seed43.json"
    rescue = recovery / "eval/lead20_steps80/full_clip_seed43.json"
    if base.is_file() and rescue.is_file() and base.stat().st_size > 0 and rescue.stat().st_size > 0:
        result["tests"]["teacher_recovery"] = {
            "baseline": full_clip(base), "finite_teacher_takeover": full_clip(rescue),
            "warning": "Intervention uses privileged Teacher; rescue does not imply the Student learned recovery.",
        }
    bins = [(0, 100), (100, 250), (250, None)]
    traces = sorted(a.source_traces.glob("motion_*.npz"))
    if traces:
        counts = [[0, 0] for _ in bins]
        for path in traces:
            with np.load(path) as d:
                frames = np.asarray(d["frame"])
                valid = np.asarray(d["teacher_valid"]).astype(bool)
                for i, (lo, hi) in enumerate(bins):
                    mask = frames >= lo if hi is None else (frames >= lo) & (frames < hi)
                    counts[i][0] += int(valid[mask].sum())
                    counts[i][1] += int(mask.sum())
        result["tests"]["teacher_label_coverage"] = {
            "trace_count": len(traces),
            "bins": [{"frames": [lo, hi], "valid": n, "total": total,
                      "fraction": n / total if total else None}
                     for (lo, hi), (n, total) in zip(bins, counts)],
            "warning": "Teacher-valid is a routing/label availability flag, not a quality score.",
        }
    root.mkdir(parents=True, exist_ok=True)
    output = root / "summary.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    print(f"[PASS] {output}")


if __name__ == "__main__":
    main()
