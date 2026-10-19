#!/usr/bin/env python3
"""Fail-closed compatibility audit for a PHP teacher and motion bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint).expanduser().resolve()
    bundle = Path(args.bundle).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if not checkpoint.is_file():
        raise SystemExit(f"[ERROR] Teacher checkpoint does not exist: {checkpoint}")
    if not bundle.is_dir():
        raise SystemExit(f"[ERROR] Motion bundle does not exist: {bundle}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    required = {
        "actor_model_state_dict",
        "critic_model_state_dict",
        "experiment_config",
        "iteration",
    }
    missing = sorted(required - set(payload))
    if missing:
        raise SystemExit(
            "[ERROR] Checkpoint is not an official-compatible Holosoma teacher; "
            f"missing keys: {missing}"
        )
    actor = payload["actor_model_state_dict"]
    first_key = "actor_module.module.0.weight"
    last_key = "actor_module.module.6.weight"
    if first_key not in actor or last_key not in actor:
        raise SystemExit("[ERROR] Teacher actor MLP keys do not match the PHP contract")
    actor_input_dim = int(actor[first_key].shape[1])
    action_dim = int(actor[last_key].shape[0])
    if actor_input_dim != 449 or action_dim != 29:
        raise SystemExit(
            f"[ERROR] Expected PHP teacher dimensions 449->29, got {actor_input_dim}->{action_dim}"
        )

    motions = sorted(bundle.glob("*_motion.npz"))
    terrains = sorted(bundle.glob("*_terrain.npy"))
    motion_stems = {p.name.removesuffix("_motion.npz") for p in motions}
    terrain_stems = {p.name.removesuffix("_terrain.npy") for p in terrains}
    unpaired_motion = sorted(motion_stems - terrain_stems)
    unpaired_terrain = sorted(terrain_stems - motion_stems)
    if not motions or unpaired_motion or unpaired_terrain:
        raise SystemExit(
            "[ERROR] Invalid motion/terrain bundle: "
            f"motions={len(motions)}, terrains={len(terrains)}, "
            f"motion_only={unpaired_motion}, terrain_only={unpaired_terrain}"
        )

    report = {
        "compatible": True,
        "checkpoint": str(checkpoint),
        "teacher_iteration": int(payload["iteration"]),
        "actor_input_dim": actor_input_dim,
        "action_dim": action_dim,
        "actor_obs_normalizer": payload.get("actor_obs_normalizer_state_dict") is not None,
        "bundle": str(bundle),
        "motion_count": len(motions),
        "terrain_count": len(terrains),
        "paired": True,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        f"[PASS] PHP teacher compatible: iteration={report['teacher_iteration']} "
        f"actor={actor_input_dim}->{action_dim}; paired clips={len(motions)}"
    )
    print(f"[PASS] Audit report: {output}")


if __name__ == "__main__":
    main()
