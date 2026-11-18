#!/usr/bin/env python3
"""Compare two frozen Students on identical, valid Student-induced states."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from probe_observation_vs_state import load_samples, motion_lengths
from probe_student_temporal_observability import load_policy


def predict(checkpoint: Path, samples: dict) -> np.ndarray:
    policy = load_policy(checkpoint, torch.device("cpu"))
    outputs = []
    with torch.inference_mode():
        for start in range(0, len(samples["action"]), 128):
            stop = start + 128
            outputs.append(policy.act_inference({
                "actor_obs": torch.from_numpy(samples["actor"][start:stop]),
                "depth_obs": torch.from_numpy(samples["depth"][start:stop]),
            }).cpu().numpy())
    return np.concatenate(outputs)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--candidate", type=Path, required=True)
    p.add_argument("--trace-dir", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.output.exists():
        p.error(f"Refusing to overwrite {a.output}")
    torch.set_num_threads(4)
    samples = load_samples(a.trace_dir, 0.2, 0.6)
    source = predict(a.source, samples)
    candidate = predict(a.candidate, samples)
    label = samples["action"]
    source_abs = np.abs(source-label)
    candidate_abs = np.abs(candidate-label)
    source_l2 = np.linalg.norm(source-label, axis=1)
    candidate_l2 = np.linalg.norm(candidate-label, axis=1)
    names = (a.trace_dir.parent/"motion_names.txt").read_text().splitlines()
    groups = []
    for clip in np.unique(samples["clip"]):
        mask = samples["clip"] == clip
        groups.append({"clip_id": int(clip), "motion": names[int(clip)], "samples": int(mask.sum()),
                       "source_l2": float(source_l2[mask].mean()),
                       "candidate_l2": float(candidate_l2[mask].mean()),
                       "delta_l2": float(candidate_l2[mask].mean()-source_l2[mask].mean())})
    phase_rows = []
    for lo, hi in ((.2,.4),(.4,.6)):
        mask=(samples["phase"]>=lo)&(samples["phase"]<hi)
        phase_rows.append({"phase":[lo,hi],"samples":int(mask.sum()),
                           "source_l2":float(source_l2[mask].mean()) if mask.any() else None,
                           "candidate_l2":float(candidate_l2[mask].mean()) if mask.any() else None})
    per_dof = [{"index": i,"source_abs":float(source_abs[:,i].mean()),
                "candidate_abs":float(candidate_abs[:,i].mean()),
                "delta_abs":float(candidate_abs[:,i].mean()-source_abs[:,i].mean())}
               for i in range(label.shape[1])]
    result={"contract":"fixed_student_closed_loop_states_valid_teacher_label",
            "warning":"Offline same-state action comparison, not candidate closed-loop performance. Trace driver must be checked from evaluation report.",
            "source":str(a.source.resolve()),"candidate":str(a.candidate.resolve()),
            "trace_dir":str(a.trace_dir.resolve()),"phase_denominator":"bundle_joint_pos_frames_minus_one",
            "full_motion_lengths":motion_lengths(a.trace_dir),"samples":len(label),
            "source_l2":float(source_l2.mean()),"candidate_l2":float(candidate_l2.mean()),
            "clips_improved":sum(r["delta_l2"]<0 for r in groups),
            "clips_regressed":sum(r["delta_l2"]>0 for r in groups),
            "phase":phase_rows,"per_dof":per_dof,"per_motion":groups}
    a.output.parent.mkdir(parents=True,exist_ok=True)
    a.output.write_text(json.dumps(result,indent=2)+"\n")
    print(f"[RESULT] samples={len(label)} source={result['source_l2']:.4f} candidate={result['candidate_l2']:.4f} improved_clips={result['clips_improved']}/{len(groups)}")
    print(f"[PASS] {a.output}")


if __name__ == "__main__":
    main()
