"""Continue an existing PHP Student on one paired motion/terrain clip.

This is a specialization probe, not evidence of from-scratch learnability.
All saved settings except registry, environment count and save interval stay fixed.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import torch
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim
from wbt_training.utils.student_teacher import PhpDistillationPPO


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=256)
    parser.add_argument("--iterations", type=int, default=200)
    args = parser.parse_args()
    if not args.checkpoint.is_file() or not args.bundle.is_dir():
        parser.error("Checkpoint or bundle is missing")
    if len(list(args.bundle.glob("*_motion.npz"))) != 1 or len(list(args.bundle.glob("*_terrain.npy"))) != 1:
        parser.error("Bundle must contain exactly one paired motion and terrain")
    if args.output_dir.exists() or args.num_envs < 96 or args.iterations < 1:
        parser.error("Output must be new; num-envs >=96; iterations >=1")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(
            saved.training, registry_name=f"file://{args.bundle.resolve()}",
            num_envs=args.num_envs, headless=True, export_onnx=False,
        ),
        algo=dataclasses.replace(
            saved.algo,
            config=dataclasses.replace(saved.algo.config, save_interval=args.iterations + 1),
        ),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    args.output_dir.mkdir(parents=True)
    config.save_config(str(args.output_dir / "holosoma_config.yaml"))
    (args.output_dir / "probe_contract.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint.resolve()),
        "source_registry": saved.training.registry_name,
        "probe_registry": config.training.registry_name,
        "num_envs": args.num_envs,
        "additional_iterations": args.iterations,
        "interpretation": "Saved Student continuation specialized to one clip; not scratch training.",
    }, indent=2) + "\n")
    env, device, app = setup_simulation_environment(config)
    try:
        state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
        algo = PhpDistillationPPO(
            device=device, env=env, config=config.algo.config,
            log_dir=str(args.output_dir), multi_gpu_cfg=None,
        )
        algo.num_teachers = int(state["num_teachers"])
        algo.setup()
        algo.attach_checkpoint_metadata(config, None)
        algo.load(str(args.checkpoint))
        algo.learn(num_learning_iterations=args.iterations)
        final = args.output_dir / "model_final.pt"
        algo.save(str(final))
        print(f"[PASS] Single-clip continuation checkpoint: {final}", flush=True)
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
