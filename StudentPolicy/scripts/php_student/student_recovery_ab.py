"""Matched short Student A/B: Student-driven vs valid-label event rescue.

The intervention trigger uses Teacher actions and is a training-only oracle.
Both arms use the same trigger and zero PPO weight on intervention windows;
only the B arm executes Teacher actions. Evaluation must have no intervention.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path

import torch
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim

from teacher_rollin_ab import RollinPPO


class RecoveryPPO(RollinPPO):
    def configure_recovery(self, arm: str, steps: int, max_events: int, cooldown: int,
                           sustain: int, threshold: float) -> None:
        # An empty fixed window leaves the official per-frame PPO gate intact.
        self.configure_rollin("student", 0.0, 0.0)
        self.recovery_arm = arm
        self.recovery_steps = steps
        self.recovery_max_events = max_events
        self.recovery_cooldown_steps = cooldown
        self.recovery_sustain = sustain
        self.recovery_threshold = threshold
        n = self.env.num_envs
        self._remaining = torch.zeros(n, dtype=torch.long, device=self.device)
        self._cooldown = torch.zeros_like(self._remaining)
        self._streak = torch.zeros_like(self._remaining)
        self._events = torch.zeros_like(self._remaining)
        self._event_count = 0
        self._active_count = 0
        self._teacher_control_count = 0
        self._invalid_abort_count = 0
        self._trigger_phase_counts = torch.zeros(5, dtype=torch.long, device=self.device)
        self._per_clip_events = torch.zeros(len(self._clip_ends), dtype=torch.long, device=self.device)

    def _update_ppo_gate_stats(self, student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> None:
        # Preserve official residual EMA and the existing per-phase coverage audit.
        super()._update_ppo_gate_stats(student_mean, teacher_actions)
        valid = torch.any(teacher_actions != 0, dim=1)
        gap = torch.linalg.vector_norm(student_mean - teacher_actions, dim=1)
        self._cooldown = torch.where(self._remaining == 0, (self._cooldown - 1).clamp_min(0), self._cooldown)
        aborted = (self._remaining > 0) & ~valid
        self._invalid_abort_count += int(aborted.sum().item())
        self._remaining[aborted] = 0
        self._streak = torch.where(valid & (gap > self.recovery_threshold) & (self._remaining == 0),
                                   self._streak + 1, 0)
        trigger = ((self._streak >= self.recovery_sustain) & (self._remaining == 0)
                   & (self._cooldown == 0) & (self._events < self.recovery_max_events))
        self._remaining[trigger] = self.recovery_steps
        self._events += trigger.long()
        self._event_count += int(trigger.sum().item())
        self._trigger_phase_counts.scatter_add_(0, self._phase_bins[trigger],
                                                torch.ones_like(self._phase_bins[trigger], dtype=torch.long))
        self._per_clip_events.scatter_add_(0, self._clip_ids[trigger],
                                           torch.ones_like(self._clip_ids[trigger], dtype=torch.long))
        active = valid & (self._remaining > 0)
        self._remaining[active] -= 1
        self._cooldown[active & (self._remaining == 0)] = self.recovery_cooldown_steps
        self._streak[active] = 0
        self._active_count += int(active.sum().item())
        # _rollout_step keeps a detached view of this same gate before calling
        # this hook; in-place masking therefore excludes current Teacher-driven
        # transitions from on-policy PPO as well as future window steps.
        self._gate[active, 0] = 0.0
        if torch.any(self._gate[active, 0] != 0):
            raise RuntimeError("Teacher-driven transition would enter PPO")
        drive = active if self.recovery_arm == "teacher" else torch.zeros_like(active)
        self._teacher_control_count += int(drive.sum().item())
        self._pending = (drive, teacher_actions)

    def install_action_swap(self) -> None:
        original_step = self.env.step

        def step_with_recovery(actor_state):
            pending, self._pending = self._pending, None
            if pending is not None:
                drive, teacher_actions = pending
                if torch.any(drive):
                    if self._warmup_active:
                        raise RuntimeError("Recovery A/B requires resumed Student, not warmup")
                    actor_state = dict(actor_state)
                    actor_state["actions"] = torch.where(drive[:, None], teacher_actions, actor_state["actions"])
            result = original_step(actor_state)
            done = result[2].bool()
            self._remaining[done] = 0
            self._cooldown[done] = 0
            self._streak[done] = 0
            self._events[done] = 0
            return result

        self.env.step = step_with_recovery

    def summary(self) -> dict:
        base = super().summary()
        base["recovery"] = {
            "arm": self.recovery_arm,
            "steps_per_event": self.recovery_steps,
            "max_events": self.recovery_max_events,
            "cooldown_steps": self.recovery_cooldown_steps,
            "sustain": self.recovery_sustain,
            "action_l2_threshold": self.recovery_threshold,
            "event_count": self._event_count,
            "active_valid_step_count": self._active_count,
            "teacher_control_step_count": self._teacher_control_count,
            "invalid_abort_count": self._invalid_abort_count,
            "event_phase_bins": self._trigger_phase_counts.cpu().tolist(),
            "per_clip_event_counts": self._per_clip_events.cpu().tolist(),
        }
        return base


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--mode", choices=("probe", "train"), required=True)
    p.add_argument("--arm", choices=("student", "teacher"), required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--num-envs", type=int, default=256)
    p.add_argument("--iterations", type=int, default=100)
    p.add_argument("--rollouts", type=int, default=21)
    p.add_argument("--steps", type=int, default=80)
    p.add_argument("--max-events", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=40)
    p.add_argument("--sustain", type=int, default=3)
    p.add_argument("--threshold", type=float, default=1.0)
    args = p.parse_args()
    if not args.checkpoint.is_file() or args.output_dir.exists():
        p.error("Source checkpoint must exist and output dir must be fresh")
    if min(args.num_envs, args.iterations, args.rollouts, args.steps, args.max_events, args.sustain) <= 0 or args.cooldown < 0 or args.threshold <= 0:
        p.error("Invalid experiment settings")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(saved.training, num_envs=args.num_envs, headless=True, export_onnx=False),
        algo=dataclasses.replace(saved.algo, config=dataclasses.replace(
            saved.algo.config, save_interval=args.iterations + 1,
        )),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    args.output_dir.mkdir(parents=True)
    config.save_config(str(args.output_dir / "holosoma_config.yaml"))
    (args.output_dir / "contract.json").write_text(json.dumps({
        "source_checkpoint": str(args.checkpoint.resolve()),
        "arm": args.arm, "mode": args.mode, "num_envs": args.num_envs,
        "iterations": args.iterations if args.mode == "train" else None,
        "rollouts": args.rollouts if args.mode == "probe" else None,
        "steps": args.steps, "max_events": args.max_events,
        "cooldown": args.cooldown, "sustain": args.sustain, "threshold": args.threshold,
        "ppo_excluded_on_recovery_active": True,
        "teacher_control_requires_valid_label": True,
    }, indent=2) + "\n")
    env, device, app = setup_simulation_environment(config)
    try:
        algo = RecoveryPPO(device=device, env=env, config=config.algo.config,
                           log_dir=str(args.output_dir), multi_gpu_cfg=None)
        algo.num_teachers = 1
        algo.setup()
        algo.attach_checkpoint_metadata(config, None)
        algo.load(str(args.checkpoint))
        algo.configure_recovery(args.arm, args.steps, args.max_events,
                                args.cooldown, args.sustain, args.threshold)
        algo.install_action_swap()
        if args.mode == "train":
            algo.learn(num_learning_iterations=args.iterations)
            (args.output_dir / "checkpoint.txt").write_text(
                str(args.output_dir / f"model_{algo.current_learning_iteration:05d}.pt") + "\n")
        else:
            algo._train_mode()
            obs = env.reset_all()
            for _ in range(args.rollouts):
                obs = algo._rollout_step(obs)
                algo.storage.clear()
        (args.output_dir / "coverage.json").write_text(json.dumps(algo.summary(), indent=2) + "\n")
        print(f"[PASS] {args.mode} {args.arm}: {args.output_dir}", flush=True)
    except BaseException:
        (args.output_dir / "error.txt").write_text(traceback.format_exc())
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
