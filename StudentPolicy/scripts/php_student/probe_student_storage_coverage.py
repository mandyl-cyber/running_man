"""Audit real DistillationPPO rollout-storage labels without updating any weights."""

from __future__ import annotations

import argparse
import dataclasses
import json
import traceback
from pathlib import Path

import numpy as np
import torch
from holosoma.utils.eval_utils import CheckpointConfig
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment

from wbt_training.utils.checkpoint import apply_preprocess_hook, load_saved_experiment_config
from wbt_training.utils.isaacsim_clock import use_php_isaacsim


def load_event_mapping(trace_dir: Path, registry: str, ends: torch.Tensor) -> tuple[list[dict], np.ndarray]:
    """Validate every replicated training segment against its source clip length."""
    from wbt_training.utils.registry import bundle_pairs

    if not registry.startswith("file://"):
        raise ValueError(f"Event mapping requires a local file:// registry, got {registry}")
    bundle = Path(registry.removeprefix("file://"))
    paths = [motion for motion, _terrain, _obj in bundle_pairs(bundle)]
    names_path = trace_dir.parent / "motion_names.txt"
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    if len(paths) != len(names) or [p.name.removesuffix("_motion.npz") for p in paths] != names:
        raise RuntimeError("Training source order does not match event trace motion_names.txt")
    lengths = []
    onsets = []
    for clip, path in enumerate(paths):
        with np.load(path) as motion:
            lengths.append(len(motion["joint_pos"]))
        trace_path = trace_dir / f"motion_{clip:02d}.npz"
        with np.load(trace_path) as trace:
            frame = np.asarray(trace["frame"])
            ankle_z = np.asarray(trace["tracked_ankle_wrist_pos_w"][:, :2, 2])
            foot_force = np.linalg.norm(trace["foot_contact_force"], axis=-1)
            ground = np.median(ankle_z[:min(40, len(ankle_z))], axis=0)
            elevated = np.any((ankle_z >= ground + 0.15) & (foot_force >= 20.0), axis=1)
            onsets.append(int(frame[np.flatnonzero(elevated)[0]]) if elevated.any() else -1)
    observed = np.diff(np.r_[-1, ends.cpu().numpy()])
    if len(observed) % len(paths) or not np.array_equal(observed, np.tile(lengths, len(observed) // len(paths))):
        raise RuntimeError("Training motion segment lengths/order differ from repeated source bundle")
    variants = len(observed) // len(paths)
    mapping = [{"training_motion_id": i, "source_clip_id": i % len(paths),
                "source_name": names[i % len(paths)], "terrain_variant": i // len(paths),
                "length": lengths[i % len(paths)], "clean_teacher_support_onset": onsets[i % len(paths)]}
               for i in range(len(observed))]
    print(f"[MAPPING] validated {len(paths)} source clips x {variants} variants = {len(observed)} training segments", flush=True)
    return mapping, np.asarray(onsets, dtype=np.int32)


def gradient_probe(algo, storage, masks: dict[str, np.ndarray], seed: int) -> dict[str, dict]:
    """Measure current-checkpoint actor gradients; never call optimizer.step()."""
    torch.manual_seed(seed)
    student_params = list(algo.policy.student.parameters())
    depth_params = list(algo.policy.depth_backbone.parameters())
    parameters = student_params + depth_params
    count_student = len(student_params)
    arrays = {key: storage[key].flatten(0, 1) for key in
              ("actor_obs", "depth_obs", "actions", "teacher_actions", "actions_log_prob",
               "advantages", "ppo_gate")}
    valid = torch.any(arrays["teacher_actions"] != 0, dim=1).cpu().numpy()
    result = {}
    for name, mask in masks.items():
        ids = np.flatnonzero(mask & valid)
        if len(ids) < 32:
            result[name] = {"available_valid_samples": len(ids), "error": "fewer than 32 valid labels"}
            continue
        chosen = torch.as_tensor(np.random.default_rng(seed).choice(ids, size=min(128, len(ids)), replace=False),
                                 device=algo.device, dtype=torch.long)
        actor = arrays["actor_obs"][chosen]
        depth = arrays["depth_obs"][chosen]
        algo.policy.act({"actor_obs": actor, "depth_obs": depth})
        mean = algo.policy.action_mean
        log_prob = algo.policy.get_actions_log_prob(arrays["actions"][chosen])
        ratio = torch.exp(log_prob - arrays["actions_log_prob"][chosen].squeeze(-1))
        advantage = arrays["advantages"][chosen].squeeze(-1)
        coef = arrays["ppo_gate"][chosen].squeeze(-1) * algo.ppo_coef
        surrogate = -advantage * ratio
        clipped = -advantage * ratio.clamp(1 - algo.config.clip_param, 1 + algo.config.clip_param)
        ppo_loss = (coef * torch.maximum(surrogate, clipped)).mean()
        dagger_loss = ((1 - coef) * algo.config.dagger_loss_coef
                       * algo._dagger_loss_per_sample(mean, arrays["teacher_actions"][chosen])).mean()
        ppo_grads = torch.autograd.grad(ppo_loss, parameters, retain_graph=True, allow_unused=True)
        dagger_grads = torch.autograd.grad(dagger_loss, parameters, allow_unused=True)
        def norms(grads):
            values = [float(torch.sqrt(sum(torch.sum(g.square()) for g in group if g is not None)).detach())
                      if any(g is not None for g in group) else 0.0
                      for group in (grads[:count_student], grads[count_student:])]
            return {"student_grad_l2": values[0], "depth_grad_l2": values[1]}
        dot = sum(torch.sum(a * b) for a, b in zip(ppo_grads, dagger_grads)
                  if a is not None and b is not None)
        ppo_norm = norms(ppo_grads)
        dagger_norm = norms(dagger_grads)
        ppo_total = (ppo_norm["student_grad_l2"] ** 2 + ppo_norm["depth_grad_l2"] ** 2) ** 0.5
        dagger_total = (dagger_norm["student_grad_l2"] ** 2 + dagger_norm["depth_grad_l2"] ** 2) ** 0.5
        result[name] = {"available_valid_samples": len(ids), "sampled": len(chosen),
                        "ppo_surrogate_loss": float(ppo_loss.detach()),
                        "dagger_loss": float(dagger_loss.detach()),
                        "importance_ratio_mean": float(ratio.detach().mean()),
                        "importance_ratio_min": float(ratio.detach().min()),
                        "importance_ratio_max": float(ratio.detach().max()),
                        "ppo_dagger_gradient_cosine": float(dot.detach()) / max(ppo_total * dagger_total, 1e-12),
                        "ppo": ppo_norm, "dagger": dagger_norm}
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollouts", type=int, default=3)
    parser.add_argument("--event-traces", type=Path,
                        help="Complete clean Teacher traces for verified source-ID mapping and pre-contact windows")
    args = parser.parse_args()
    if args.output.exists() or not args.checkpoint.is_file() or args.num_envs < 1 or args.rollouts < 1:
        parser.error("Use a fresh output, existing checkpoint, and positive sizes")
    saved, _ = load_saved_experiment_config(CheckpointConfig(checkpoint=str(args.checkpoint)))
    registry = saved.training.registry_name
    config = dataclasses.replace(
        saved,
        training=dataclasses.replace(saved.training, num_envs=args.num_envs, headless=True, export_onnx=False),
    )
    config = apply_preprocess_hook(use_php_isaacsim(config))
    env, device, app = setup_simulation_environment(config)
    try:
        algo = get_class(config.algo._target_)(
            device=device, env=env, config=config.algo.config, log_dir=None, multi_gpu_cfg=None
        )
        algo.num_teachers = 1
        algo.setup()
        algo.load(str(args.checkpoint))
        algo.policy.require_loaded_teachers()
        algo._warmup_active = False
        obs = env.reset_all()
        cmd = env.command_manager.get_state("motion_command")
        ends = torch.nonzero(cmd.motion.motion_ends, as_tuple=False).squeeze(-1)
        starts = torch.cat((torch.zeros_like(ends[:1]), ends[:-1] + 1))
        event_mapping, event_onsets = (load_event_mapping(args.event_traces, registry, ends)
                                       if args.event_traces else ([], np.empty(0, dtype=np.int32)))
        snapshots = []
        advantages = []
        rewards = []
        gradient_results = []
        original_hook = algo._update_ppo_gate_stats

        def capture_before_step(student_mean: torch.Tensor, teacher_actions: torch.Tensor) -> None:
            motion_id = torch.searchsorted(ends, cmd.time_steps, right=False)
            phase = (cmd.time_steps - starts[motion_id]).float() / (ends[motion_id] - starts[motion_id]).clamp_min(1)
            valid = torch.any(teacher_actions != 0, dim=-1)
            l2 = torch.linalg.vector_norm(student_mean - teacher_actions, dim=-1)
            gate = algo._get_ppo_gate().squeeze(-1)
            engaged = bool(cmd._adaptive_gate_table() is not None) if hasattr(cmd, "_adaptive_gate_table") else False
            snapshots.append((
                motion_id.cpu().numpy().copy(), phase.cpu().numpy().copy(),
                valid.cpu().numpy().copy(), l2.cpu().numpy().copy(), gate.cpu().numpy().copy(),
                teacher_actions.cpu().numpy().copy(),
                (cmd.time_steps - starts[motion_id]).cpu().numpy().copy(),
                np.full(motion_id.shape[0], engaged, dtype=np.bool_),
            ))
            original_hook(student_mean, teacher_actions)

        algo._update_ppo_gate_stats = capture_before_step
        max_storage_label_error = 0.0
        for index in range(args.rollouts):
            first = len(snapshots)
            obs = algo._rollout_step(obs)
            captured = np.stack([row[5] for row in snapshots[first:]])
            stored = algo.storage["teacher_actions"].cpu().numpy()
            max_storage_label_error = max(max_storage_label_error, float(np.abs(captured - stored).max()))
            if algo.storage.step != algo.config.num_steps_per_env:
                raise RuntimeError("Storage did not contain the expected number of rollout steps")
            advantages.append(algo.storage["advantages"].cpu().numpy().reshape(-1).copy())
            rewards.append(algo.storage["rewards"].cpu().numpy().reshape(-1).copy())
            if event_mapping and index >= max(args.rollouts - 3, 0):
                local_ids = np.concatenate([row[0] for row in snapshots[first:]])
                local_frame = np.concatenate([row[6] for row in snapshots[first:]])
                local_post = np.concatenate([row[7] for row in snapshots[first:]]).astype(bool)
                local_source = local_ids % len(event_onsets)
                local_onset = event_onsets[local_source]
                one_gradient = gradient_probe(algo, algo.storage, {
                    "pre_contact_proxy": local_post & (local_onset >= 0)
                    & (local_frame >= local_onset - 110) & (local_frame < local_onset - 40),
                    "early_control": local_post & (local_frame >= 20) & (local_frame < 100),
                }, seed=42 + index)
                gradient_results.append({"rollout": index + 1, "windows": one_gradient})
            algo.storage.clear()
            print(f"[STORAGE PROBE] rollout={index + 1}/{args.rollouts}", flush=True)
        ids = np.concatenate([row[0] for row in snapshots])
        phase = np.concatenate([row[1] for row in snapshots])
        valid = np.concatenate([row[2] for row in snapshots]).astype(bool)
        l2 = np.concatenate([row[3] for row in snapshots])
        gate = np.concatenate([row[4] for row in snapshots])
        frame = np.concatenate([row[6] for row in snapshots])
        post_gate = np.concatenate([row[7] for row in snapshots]).astype(bool)
        advantage = np.concatenate(advantages)
        reward = np.concatenate(rewards)
        effective_ppo = gate * float(algo.ppo_coef)
        effective_dagger = (1.0 - effective_ppo) * float(algo.config.dagger_loss_coef) * valid
        bins = np.minimum((phase * 5).astype(int), 4)

        def summarize(mask: np.ndarray) -> dict:
            selected = mask & valid
            return {
                "samples": int(mask.sum()),
                "valid_labels": int(selected.sum()),
                "valid_fraction": float(selected.sum() / mask.sum()) if mask.any() else None,
                "valid_action_l2_mean": float(l2[selected].mean()) if selected.any() else None,
                "ppo_gate_mean": float(gate[mask].mean()) if mask.any() else None,
                "effective_ppo_coef_mean": float(effective_ppo[mask].mean()) if mask.any() else None,
                "effective_dagger_weight_mean": float(effective_dagger[mask].mean()) if mask.any() else None,
                "zero_dagger_label_fraction": float((mask & ~valid).sum() / mask.sum()) if mask.any() else None,
                "advantage_abs_mean": float(np.abs(advantage[mask]).mean()) if mask.any() else None,
                "advantage_positive_fraction": float((advantage[mask] > 0).mean()) if mask.any() else None,
                "reward_mean": float(reward[mask].mean()) if mask.any() else None,
                "weighted_ppo_advantage_abs_mean": float((np.abs(advantage[mask]) * effective_ppo[mask]).mean()) if mask.any() else None,
                "weighted_dagger_residual_mean": float((np.square(l2[mask]) / 29 * effective_dagger[mask]).mean()) if mask.any() else None,
            }

        result = {
            "contract": "saved_training_config_real_rollout_storage_no_optimizer_step",
            "checkpoint": str(args.checkpoint),
            "num_envs": args.num_envs,
            "rollouts": args.rollouts,
            "steps_per_rollout": algo.config.num_steps_per_env,
            "total_samples": int(len(ids)),
            "storage_teacher_action_max_abs_error": max_storage_label_error,
            "overall": summarize(np.ones(len(ids), dtype=bool)),
            "phase": {f"{i * 20}-{(i + 1) * 20}%": summarize(bins == i) for i in range(5)},
            "frame_180_250": summarize((frame >= 180) & (frame < 250)),
            "post_gate_overall": summarize(post_gate),
            "post_gate_phase": {f"{i * 20}-{(i + 1) * 20}%": summarize(post_gate & (bins == i)) for i in range(5)},
            "post_gate_frame_180_250": summarize(post_gate & (frame >= 180) & (frame < 250)),
            "post_gate_motion_frame_180_250": {
                str(i): summarize(post_gate & (ids == i) & (frame >= 180) & (frame < 250))
                for i in range(len(ends))
            },
            "motion": {str(i): summarize(ids == i) for i in range(len(ends))},
            "motion_phase": {
                str(motion_id): {
                    f"{i * 20}-{(i + 1) * 20}%": summarize((ids == motion_id) & (bins == i))
                    for i in range(5)
                }
                for motion_id in range(len(ends))
            },
            "ppo_coef_checkpoint": float(algo.ppo_coef),
            "dagger_loss_coef": float(algo.config.dagger_loss_coef),
            "ppo_gate_warmup_samples": float(getattr(cmd, "_gate_warmup", 0.0)),
            "ppo_gate_seen_samples": float(getattr(cmd, "_gate_seen", 0.0)),
            "ppo_gate_engaged": bool(cmd._adaptive_gate_table() is not None)
            if hasattr(cmd, "_adaptive_gate_table") else False,
            "motion_source_paths": [str(path) for path in cmd._motion_files_for(cmd.motion)]
            if hasattr(cmd, "_motion_files_for") else [],
        }
        if event_mapping:
            source_id = ids % len(event_onsets)
            onset = event_onsets[source_id]
            pre = (onset >= 0) & (frame >= onset - 110) & (frame < onset - 40)
            early = (frame >= 20) & (frame < 100)
            result["event_mapping"] = event_mapping
            result["pre_contact_proxy_window"] = [-110, -40]
            result["pre_contact_proxy_post_gate"] = summarize(post_gate & pre)
            result["early_control_post_gate"] = summarize(post_gate & early)
            result["pre_contact_proxy_post_gate_by_source"] = {
                str(i): summarize(post_gate & pre & (source_id == i)) for i in range(len(event_onsets))
            }
            result["last_three_rollout_gradient_probes"] = gradient_results
            result["event_warning"] = (
                "Clean Teacher elevated-support onset is only a source-clip proxy; "
                "training terrain variants may shift actual contact timing. Gradients are current-checkpoint "
                "counterfactuals on one fresh rollout, not historical optimizer updates."
            )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(f"[PASS] {args.output}", flush=True)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
