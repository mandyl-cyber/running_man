"""Summarize per-link contact forces from a complete diagnostic trace set."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=10.0)
    args = parser.parse_args()
    if args.output.exists() or args.threshold <= 0:
        parser.error("Output must be fresh and threshold positive")
    report_paths = sorted(args.report_dir.glob("full_clip_seed*.json"))
    if len(report_paths) != 1:
        raise RuntimeError(f"Expected exactly one full-clip report in {args.report_dir}")
    report = json.loads(report_paths[0].read_text())
    trace_dir = args.report_dir / "traces"
    names = json.loads((trace_dir / "contact_body_names.json").read_text())
    files = sorted(trace_dir.glob("motion_*.npz"))
    if not report.get("complete") or len(files) != len(report["motions"]):
        raise RuntimeError("Report or trace set is incomplete")
    wrist_ids = [i for i, name in enumerate(names) if "wrist" in name]
    if len(wrist_ids) != 6:
        raise RuntimeError(f"Expected six G1 wrist links, got {[names[i] for i in wrist_ids]}")
    all_norms = []
    rows = []
    for path in files:
        clip = int(path.stem.rsplit("_", 1)[1])
        with np.load(path) as trace:
            force = trace["body_contact_force"]
            if force.ndim != 3 or force.shape[1:] != (len(names), 3) or not np.isfinite(force).all():
                raise RuntimeError(f"Invalid contact forces: {path}")
            norms = np.linalg.norm(force, axis=-1)
        wrist_frames = int(np.any(norms[:, wrist_ids] > args.threshold, axis=1).sum())
        rows.append({"clip": clip, "successes": report["motions"][clip]["successes"],
                     "frames": len(norms), "wrist_contact_frames": wrist_frames})
        all_norms.append(norms)
    combined = np.concatenate(all_norms)
    body_rows = [{"body": name, "contact_fraction": float((combined[:, i] > args.threshold).mean()),
                  "max_force_N": float(combined[:, i].max())} for i, name in enumerate(names)]
    result = {"contract": "teacher_control_full_body_net_contact_force_audit",
              "warning": "Wrist links are a proxy for fixed half-sphere hands. Net force does not identify the other contact body; threshold choice matters.",
              "report_dir": str(args.report_dir.resolve()), "threshold_N": args.threshold,
              "motions": len(rows), "teacher_successes": sum(r["successes"] for r in rows),
              "successful_clips_without_wrist_contact": sum(r["successes"] for r in rows
                                                             if r["wrist_contact_frames"] == 0),
              "clips_with_wrist_contact": sum(r["wrist_contact_frames"] > 0 for r in rows),
              "body_rows": body_rows, "clip_rows": rows}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"[PASS] {args.output}")
    print(f"[RESULT] Teacher={result['teacher_successes']}/{len(rows)}, "
          f"successful clips without wrist force={result['successful_clips_without_wrist_contact']}, "
          f"clips with wrist force={result['clips_with_wrist_contact']}")


if __name__ == "__main__":
    main()
