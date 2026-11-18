"""Deterministic frame-0 full-clip report for an official PHP depth student."""

from __future__ import annotations

import dataclasses
import json
import os
import traceback
from pathlib import Path
from statistics import mean

import numpy as np
import torch
import tyro
from holosoma.config_types.eval_callback import EvalCallbacksConfig
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.utils.config_utils import CONFIG_NAME
from holosoma.utils.eval_utils import CheckpointConfig, init_eval_logging, load_checkpoint
from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
from holosoma.utils.helpers import get_class
from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
from loguru import logger

from wbt_training.training_runs.eval_student import _load_student_only
from wbt_training.training_runs.eval_teacher import _apply_eval_profile
from wbt_training.utils.checkpoint import (
    apply_preprocess_hook,
    load_saved_experiment_config,
    parse_saved_config_overrides,
    resolve_checkpoint_reference,
)
from wbt_training.utils.isaacsim_clock import use_php_isaacsim
from wbt_training.config_values.motion_command import _quat_rotate_inverse


@torch.no_grad()
def evaluate_full_clips(algo, output: Path, target: int, max_steps: int) -> None:
    algo._create_eval_callbacks()
    algo._pre_evaluate_policy()
    obs = algo.env.reset_all()
    state = {
        "done_indices": [],
        "stop": False,
        "obs": obs,
        "actions": torch.zeros(algo.env.num_envs, algo.num_act, device=algo.device),
    }
    actor_obs, _teacher_obs, critic_obs, depth_obs = algo._extract_groups(obs)
    state["obs"]["actor_obs"] = actor_obs
    state["obs"]["critic_obs"] = critic_obs
    state["obs"]["depth_obs"] = depth_obs
    policy = algo.get_inference_policy()

    command = algo.env.command_manager.get_state("motion_command")
    if command is None or not hasattr(command, "motion_ids"):
        raise RuntimeError("Full-clip evaluation requires a stateful motion_command")
    clip_ends = torch.nonzero(command.motion.motion_ends, as_tuple=False).squeeze(-1)
    count = int(clip_ends.numel())
    print(f"[STUDENT FULL EVAL] loaded_motions={count} envs={algo.env.num_envs}", flush=True)
    successes = torch.zeros(count, dtype=torch.long, device=algo.device)
    failures = torch.zeros_like(successes)
    timeouts = torch.zeros_like(successes)
    diagnostic = os.environ.get("PHP_EVAL_DIAGNOSTICS", "0") == "1"
    takeover_mode = os.environ.get("PHP_EVAL_TAKEOVER_MODE", "none")
    takeover_start_fraction = float(os.environ.get("PHP_EVAL_TAKEOVER_START_FRACTION", "0.25"))
    takeover_steps = int(os.environ.get("PHP_EVAL_TAKEOVER_STEPS", "0"))
    joint_spec = os.environ.get("PHP_EVAL_TAKEOVER_JOINT_IDS", "")
    takeover_joint_ids = tuple(int(value) for value in joint_spec.split(",") if value)
    if len(set(takeover_joint_ids)) != len(takeover_joint_ids) or any(
        value < 0 or value >= algo.num_act for value in takeover_joint_ids
    ):
        raise ValueError(f"Invalid takeover joint IDs: {joint_spec}")
    rescue_max_events = int(os.environ.get("PHP_EVAL_RESCUE_MAX_EVENTS", "3"))
    rescue_sustain = int(os.environ.get("PHP_EVAL_RESCUE_SUSTAIN", "3"))
    rescue_threshold = float(os.environ.get("PHP_EVAL_RESCUE_ACTION_THRESHOLD", "1.0"))
    rescue_cooldown_steps = int(os.environ.get("PHP_EVAL_RESCUE_COOLDOWN_STEPS", "40"))
    takeover_start_map_path = os.environ.get("PHP_EVAL_TAKEOVER_START_MAP", "")
    takeover_start_map = (
        {int(k): int(v) for k, v in json.loads(Path(takeover_start_map_path).read_text()).items()}
        if takeover_start_map_path else {}
    )
    trace_spec = os.environ.get("PHP_EVAL_TRACE_MOTION_IDS", "")
    trace_ids = (set(range(count)) if trace_spec == "all" else
                 {int(value) for value in trace_spec.split(",") if value})
    trace_stride = int(os.environ.get("PHP_EVAL_TRACE_DEPTH_STRIDE", "5"))
    if trace_ids and (not diagnostic or min(trace_ids) < 0 or max(trace_ids) >= count or trace_stride < 1):
        raise ValueError("Trace requires diagnostics, in-range motion IDs, and a positive depth stride")
    traces = {idx: {"rows": [], "depth": [], "depth_frame": []} for idx in trace_ids}
    trace_envs: dict[int, int] = {}
    traced_completed: set[int] = set()
    foot_indices = ([algo.env.simulator.body_names.index(name) for name in
                     ("left_foot_contact_point", "right_foot_contact_point")] if trace_ids else [])
    # The half-sphere hand collision geometries are fixed below the wrist yaw
    # links in the source URDF. The runtime robot uses the 32-body ordering, so
    # these wrist net forces are hand-contact proxies, not dedicated palm forces.
    hand_indices = ([algo.env.simulator.body_names.index(name) for name in
                     ("left_wrist_yaw_link", "right_wrist_yaw_link")] if trace_ids else [])
    if trace_ids:
        # Fail explicitly: a zero-filled fallback would turn missing actuator
        # telemetry into a false conclusion about insufficient pull-up torque.
        sim = algo.env.simulator
        if not hasattr(sim, "_robot") or not hasattr(sim._robot, "data") or not hasattr(sim._robot.data, "applied_torque"):
            raise RuntimeError("IsaacSim applied_torque is unavailable; cannot capture torque trace")
        if not hasattr(sim, "dof_ids"):
            raise RuntimeError("Simulator dof_ids unavailable; cannot align applied torque with actions")
        # Keep the camera pose history in the same newest-at-the-end layout as
        # WarpDepthImageObsTerm.depth_buffer.  The policy receives a latency-
        # selected image, so the current camera pose alone is not sufficient to
        # establish image/robot alignment.
        depth_term = algo.env.observation_manager._term_instances["depth_camera"]["depth_cam"]
        camera = depth_term.camera_sensor
        camera_pose_pos = camera.camera_sensor_position[:, 0].detach().clone()
        camera_pose_quat = camera.camera_sensor_orientation[:, 0].detach().clone()
        depth_pose_pos_history = camera_pose_pos[:, None, :].expand(
            -1, depth_term.buffer_len, -1
        ).clone()
        depth_pose_quat_history = camera_pose_quat[:, None, :].expand(
            -1, depth_term.buffer_len, -1
        ).clone()
        original_get_depth_images = depth_term._get_depth_images

        def traced_get_depth_images(env):
            depth_images = original_get_depth_images(env)
            new_pos = camera.camera_sensor_position[:, 0].detach()
            new_quat = camera.camera_sensor_orientation[:, 0].detach()
            now = depth_term.command.time_steps
            reset = (now < depth_term._prev_time_step) | depth_term.reset_episodes
            depth_pose_pos_history[:, :-1] = depth_pose_pos_history[:, 1:].clone()
            depth_pose_quat_history[:, :-1] = depth_pose_quat_history[:, 1:].clone()
            depth_pose_pos_history[:, -1] = new_pos
            depth_pose_quat_history[:, -1] = new_quat
            if reset.any():
                depth_pose_pos_history[reset] = new_pos[reset, None, :].expand(
                    -1, depth_term.buffer_len, -1
                )
                depth_pose_quat_history[reset] = new_quat[reset, None, :].expand(
                    -1, depth_term.buffer_len, -1
                )
            return depth_images

        # _get_depth_images is invoked as an instance attribute by __call__;
        # assigning this closure intentionally avoids changing official code.
        depth_term._get_depth_images = traced_get_depth_images
        routing_column = int(algo.policy.routing_column)
        body_names = list(command.motion_cfg.body_names_to_track)
        body_indices = torch.tensor(
            [body_names.index(name) for name in
             ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")],
            device=algo.device,
        )
    if takeover_mode not in {"none", "window", "all", "mapped_tail", "mapped_window", "rescue_events"}:
        raise ValueError(f"Invalid takeover mode: {takeover_mode}")
    if takeover_mode != "none" and not diagnostic:
        raise ValueError("Teacher takeover requires diagnostics and loaded teacher weights")
    if takeover_mode == "window" and (not 0 <= takeover_start_fraction < 1 or takeover_steps <= 0):
        raise ValueError("Window takeover requires start fraction in [0, 1) and positive steps")
    if takeover_mode in {"mapped_tail", "mapped_window"} and set(takeover_start_map) != set(range(count)):
        raise ValueError(
            f"{takeover_mode} requires one start frame for every motion; got "
            f"{len(takeover_start_map)} for {count} motions"
        )
    if takeover_mode == "mapped_window" and takeover_steps <= 0:
        raise ValueError("mapped_window requires positive takeover steps")
    if takeover_mode == "rescue_events" and (takeover_steps <= 0 or rescue_max_events <= 0 or rescue_sustain <= 0 or rescue_threshold <= 0 or rescue_cooldown_steps < 0):
        raise ValueError("rescue_events requires positive steps/max events/sustain/threshold and nonnegative cooldown")
    mapped_start_frames = (
        torch.tensor([takeover_start_map[i] for i in range(count)], device=algo.device)
        if takeover_start_map else None
    )
    clip_starts = torch.cat((torch.zeros_like(clip_ends[:1]), clip_ends[:-1] + 1))
    failure_records: list[list[dict]] = [[] for _ in range(count)]
    latest_termination: dict[str, torch.Tensor] = {}
    action_error_sum = torch.zeros(algo.env.num_envs, device=algo.device)
    action_error_count = torch.zeros_like(action_error_sum)
    takeover_count = torch.zeros_like(action_error_sum)
    rescue_remaining = torch.zeros(algo.env.num_envs, dtype=torch.long, device=algo.device)
    rescue_streak = torch.zeros_like(rescue_remaining)
    rescue_events = torch.zeros_like(rescue_remaining)
    rescue_invalid_aborts = torch.zeros_like(rescue_remaining)
    rescue_cooldown = torch.zeros_like(rescue_remaining)
    rescue_event_frames: list[list[int]] = [[] for _ in range(algo.env.num_envs)]
    takeover_completed: list[list[int]] = [[] for _ in range(count)]
    rescue_events_completed: list[list[int]] = [[] for _ in range(count)]
    rescue_aborts_completed: list[list[int]] = [[] for _ in range(count)]
    rescue_event_frames_completed: list[list[list[int]]] = [[] for _ in range(count)]
    depth_finite_last = torch.ones(algo.env.num_envs, device=algo.device)
    depth_std_last = torch.zeros_like(depth_finite_last)
    if diagnostic:
        term = algo.env.termination_manager._term_instances["bad_tracking"]
        original_check = algo.env.termination_manager.check

        def check_with_details():
            result = original_check()
            active_clip = torch.searchsorted(clip_ends, command.time_steps, right=False)
            starts = torch.cat((torch.zeros_like(clip_ends[:1]), clip_ends[:-1] + 1))
            start_pos = command.motion._body_pos_w[starts[active_clip], 0, :2]
            end_pos = command.motion._body_pos_w[clip_ends[active_clip], 0, :2]
            robot_pos = command.robot_ref_pos_w[:, :2] - algo.env.simulator.scene.env_origins[:, :2]
            path = end_pos - start_pos
            robot_progress = ((robot_pos - start_pos) * path).sum(dim=1) / path.square().sum(dim=1).clamp_min(1e-6)
            latest_termination.clear()
            latest_termination.update({
                "frame": command.time_steps.clone(),
                "ref_pos": term.bad_ref_pos(command).clone(),
                "ref_ori": term.bad_ref_ori(command).clone(),
                "body_pos": term.bad_motion_body_pos(command).clone(),
                "ref_pos_error": torch.linalg.vector_norm(
                    command.ref_pos_w - command.robot_ref_pos_w, dim=1
                ).clone(),
                "robot_progress": robot_progress.clone(),
            })
            return result

        algo.env.termination_manager.check = check_with_details
    executed = 0

    for step in range(max_steps):
        prior_ids = torch.searchsorted(clip_ends, command.time_steps, right=False)
        prior_frame = command.time_steps - clip_starts[prior_ids]
        state["step"] = step
        state = algo._pre_eval_env_step(state, policy)
        if diagnostic:
            _actor_obs, teacher_obs, _critic_obs, depth_obs = algo._extract_groups(state["obs"])
            flat_depth = depth_obs.reshape(depth_obs.shape[0], -1)
            depth_finite_last = torch.isfinite(flat_depth).float().mean(dim=1)
            depth_std_last = torch.nan_to_num(flat_depth).std(dim=1)
            teacher_actions = algo.policy.teacher_act(teacher_obs)
            valid = torch.any(teacher_actions != 0, dim=1)
            if trace_ids:
                ref_pos_error_all = torch.linalg.vector_norm(command.ref_pos_w - command.robot_ref_pos_w, dim=1)
                gravity = gravity_vector(algo.env)
                motion_g_b = _quat_rotate_inverse(command.ref_quat_w, gravity, w_last=True)
                robot_g_b = _quat_rotate_inverse(command.robot_ref_quat_w, gravity, w_last=True)
                ref_ori_error_all = torch.abs(motion_g_b[:, 2] - robot_g_b[:, 2])
                body_errors_all = torch.linalg.vector_norm(
                    command.body_pos_relative_w[:, body_indices] - command.robot_body_pos_w[:, body_indices], dim=-1
                )
                body_error_max_all = body_errors_all.max(dim=1).values
                velocity_commands = command.vel_cmd
            for idx in trace_ids - traced_completed:
                candidates = torch.where(prior_ids == idx)[0]
                if not candidates.numel():
                    continue
                env_id = trace_envs.setdefault(idx, int(candidates[0].item()))
                if int(prior_ids[env_id].item()) != idx:
                    raise RuntimeError(f"Trace motion routing changed unexpectedly for env {env_id}")
                frame = int(prior_frame[env_id].item())
                root = algo.env.simulator.robot_root_states[env_id]
                foot_forces = algo.env.simulator.contact_forces[env_id, foot_indices]
                hand_forces = algo.env.simulator.contact_forces[env_id, hand_indices]
                if not torch.isfinite(hand_forces).all():
                    raise RuntimeError(f"Nonfinite wrist/hand contact proxy for motion {idx} frame {frame}")
                body_forces = algo.env.simulator.contact_forces[env_id]
                if body_forces.shape != (len(algo.env.simulator.body_names), 3) or not torch.isfinite(body_forces).all():
                    raise RuntimeError(f"Invalid full-body contact forces for motion {idx} frame {frame}")
                student_action = state["actions"][env_id]
                teacher_action = teacher_actions[env_id]
                traces[idx]["rows"].append({
                    "frame": frame,
                    "teacher_valid": bool(valid[env_id].item()),
                    "teacher_routing_index": float(teacher_obs[env_id, routing_column].item()),
                    "expert_lost_ref_pos": bool((ref_pos_error_all[env_id] > 0.5).item()),
                    "expert_lost_ref_ori": bool((ref_ori_error_all[env_id] > 0.8).item()),
                    "expert_lost_body_pos": bool((body_errors_all[env_id] > 0.5).any().item()),
                    "ref_ori_error_gravity_z": float(ref_ori_error_all[env_id].item()),
                    "tracked_body_pos_error_max_m": float(body_error_max_all[env_id].item()),
                    "velocity_command": velocity_commands[env_id].detach().cpu().numpy().copy(),
                    "student_action": student_action.detach().cpu().numpy().copy(),
                    "teacher_action": teacher_action.detach().cpu().numpy().copy(),
                    "actor_obs": _actor_obs[env_id].detach().cpu().numpy().copy(),
                    "teacher_obs": teacher_obs[env_id].detach().cpu().numpy().copy(),
                    "robot_ref_pos_w": command.robot_ref_pos_w[env_id].detach().cpu().numpy().copy(),
                    "ref_pos_w": command.ref_pos_w[env_id].detach().cpu().numpy().copy(),
                    "root_state": root.detach().cpu().numpy().copy(),
                    "foot_contact_force": foot_forces.detach().cpu().numpy().copy(),
                    "hand_contact_force_proxy": hand_forces.detach().cpu().numpy().copy(),
                    "body_contact_force": body_forces.detach().cpu().numpy().copy(),
                    "tracked_ankle_wrist_pos_w": (
                        command.robot_body_pos_w[env_id, body_indices].detach().cpu().numpy().copy()
                    ),
                })
                if frame % trace_stride == 0:
                    image = depth_obs[env_id].detach()
                    buffer_images = depth_term.depth_buffer[env_id]
                    # _process_depth_images already adds noise/holes and
                    # normalizes before writing depth_buffer; the observation
                    # group returns that buffered tensor without another
                    # transform.
                    match_error = (buffer_images - image.unsqueeze(0)).abs().mean(dim=(1, 2))
                    buffer_index = int(match_error.argmin().item())
                    traces[idx]["depth"].append(image.cpu().numpy().copy())
                    traces[idx]["depth_frame"].append(frame)
                    traces[idx].setdefault("depth_latency_frames", []).append(
                        depth_term.buffer_len - 1 - buffer_index
                    )
                    traces[idx].setdefault("depth_buffer_match_error", []).append(
                        float(match_error[buffer_index].item())
                    )
                    traces[idx].setdefault("depth_camera_pos_w", []).append(
                        depth_pose_pos_history[env_id, buffer_index].cpu().numpy().copy()
                    )
                    traces[idx].setdefault("depth_camera_quat_xyzw", []).append(
                        depth_pose_quat_history[env_id, buffer_index].cpu().numpy().copy()
                    )
            error = torch.linalg.vector_norm(state["actions"] - teacher_actions, dim=1)
            action_error_sum += torch.where(valid, error, 0.0)
            action_error_count += valid.float()
            if takeover_mode == "all":
                takeover = valid
            elif takeover_mode == "window":
                start_frame = (takeover_start_fraction * (clip_ends[prior_ids] - clip_starts[prior_ids] + 1)).long()
                takeover = valid & (prior_frame >= start_frame) & (prior_frame < start_frame + takeover_steps)
            elif takeover_mode == "mapped_tail":
                takeover = valid & (prior_frame >= mapped_start_frames[prior_ids])
            elif takeover_mode == "mapped_window":
                start = mapped_start_frames[prior_ids]
                takeover = valid & (prior_frame >= start) & (prior_frame < start + takeover_steps)
            elif takeover_mode == "rescue_events":
                rescue_cooldown = torch.where(rescue_remaining == 0, (rescue_cooldown - 1).clamp_min(0), rescue_cooldown)
                aborted = (rescue_remaining > 0) & ~valid
                rescue_invalid_aborts += aborted.long()
                rescue_remaining[aborted] = 0
                rescue_streak = torch.where(valid & (error > rescue_threshold) & (rescue_remaining == 0), rescue_streak + 1, 0)
                trigger = ((rescue_streak >= rescue_sustain) & (rescue_remaining == 0)
                           & (rescue_cooldown == 0) & (rescue_events < rescue_max_events))
                for env_id in torch.where(trigger)[0].tolist():
                    rescue_event_frames[env_id].append(int(prior_frame[env_id].item()))
                rescue_remaining[trigger] = takeover_steps
                rescue_events += trigger.long()
                takeover = valid & (rescue_remaining > 0)
                rescue_remaining[takeover] -= 1
                rescue_cooldown[takeover & (rescue_remaining == 0)] = rescue_cooldown_steps
                rescue_streak[takeover] = 0
            else:
                takeover = torch.zeros_like(valid)
            if takeover.any():
                joint_mask = torch.zeros(algo.num_act, dtype=torch.bool, device=algo.device)
                joint_mask[list(takeover_joint_ids) if takeover_joint_ids else slice(None)] = True
                state["actions"] = torch.where(takeover[:, None] & joint_mask[None, :], teacher_actions, state["actions"])
                takeover_count += takeover.float()
            if trace_ids:
                for idx, env_id in trace_envs.items():
                    if idx not in traced_completed:
                        traces[idx]["rows"][-1]["applied_action"] = state["actions"][env_id].detach().cpu().numpy().copy()
        state = algo.env_step(state)
        if trace_ids:
            for idx, env_id in trace_envs.items():
                if idx in traced_completed:
                    continue
                row = traces[idx]["rows"][-1]
                torque = sim._robot.data.applied_torque[env_id, sim.dof_ids].detach()
                if torque.numel() != algo.num_act or not torch.isfinite(torque).all():
                    raise RuntimeError(f"Invalid applied torque for motion {idx}: shape={tuple(torque.shape)}")
                row["applied_torque_post_step"] = torque.cpu().numpy().copy()
                row["root_state_post_step"] = sim.robot_root_states[env_id].detach().cpu().numpy().copy()
        state = algo._post_eval_env_step(state)
        done = state["dones"].bool()
        timeout = state["extras"].get("time_outs", torch.zeros_like(done)).bool()
        ended = command.motion_end_reset.bool() & ~done
        for idx, env_id in trace_envs.items():
            if bool((done | ended)[env_id].item()):
                traced_completed.add(idx)
        for idx in range(count):
            belongs = prior_ids == idx
            remaining = max(target - int((successes[idx] + failures[idx]).item()), 0)
            if not remaining:
                continue
            s = min(int((ended & belongs).sum().item()), remaining)
            successes[idx] += s
            remaining -= s
            timeouts[idx] += int((done & timeout & belongs).sum().item())
            failure_envs = torch.where(done & ~timeout & belongs)[0][:remaining]
            failures[idx] += failure_envs.numel()
            if diagnostic:
                clip_start = 0 if idx == 0 else int(clip_ends[idx - 1].item()) + 1
                clip_end = int(clip_ends[idx].item())
                for env_id in failure_envs.tolist():
                    frame = int(latest_termination["frame"][env_id].item())
                    failure_records[idx].append({
                        "frame": frame - clip_start,
                        "progress_ratio": (frame - clip_start) / max(clip_end - clip_start, 1),
                        "ref_pos": bool(latest_termination["ref_pos"][env_id].item()),
                        "ref_ori": bool(latest_termination["ref_ori"][env_id].item()),
                        "body_pos": bool(latest_termination["body_pos"][env_id].item()),
                        "ref_pos_error_m": float(latest_termination["ref_pos_error"][env_id].item()),
                        "robot_path_projection_ratio": float(latest_termination["robot_progress"][env_id].item()),
                        "depth_finite_fraction": float(depth_finite_last[env_id].item()),
                        "depth_image_std": float(depth_std_last[env_id].item()),
                        "student_teacher_action_l2_mean": (
                            float((action_error_sum[env_id] / action_error_count[env_id]).item())
                            if action_error_count[env_id] > 0 else None
                        ),
                        "teacher_takeover_steps_applied": int(takeover_count[env_id].item()),
                        "rescue_events_triggered": int(rescue_events[env_id].item()),
                        "rescue_invalid_aborts": int(rescue_invalid_aborts[env_id].item()),
                    })
            if diagnostic:
                completed = torch.where((ended | done) & belongs)[0]
                takeover_completed[idx].extend(int(takeover_count[env_id].item()) for env_id in completed.tolist())
                rescue_events_completed[idx].extend(int(rescue_events[env_id].item()) for env_id in completed.tolist())
                rescue_aborts_completed[idx].extend(int(rescue_invalid_aborts[env_id].item()) for env_id in completed.tolist())
                rescue_event_frames_completed[idx].extend(rescue_event_frames[env_id].copy() for env_id in completed.tolist())
                for env_id in completed.tolist():
                    rescue_event_frames[env_id].clear()
                action_error_sum[completed] = 0.0
                action_error_count[completed] = 0.0
                takeover_count[completed] = 0.0
                rescue_remaining[completed] = 0
                rescue_streak[completed] = 0
                rescue_events[completed] = 0
                rescue_invalid_aborts[completed] = 0
                rescue_cooldown[completed] = 0
        executed = step + 1
        if bool(torch.all(successes + failures >= target)):
            break
        if step and step % 1000 == 0:
            print(f"[STUDENT FULL EVAL] step={step} completed={int((successes + failures).sum().item())}/{count * target}", flush=True)

    algo._post_evaluate_policy()
    if traces:
        trace_dir = Path(os.environ["PHP_EVAL_TRACE_DIR"])
        trace_dir.mkdir(parents=True, exist_ok=True)
        (trace_dir / "contact_body_names.json").write_text(
            json.dumps(list(algo.env.simulator.body_names), indent=2) + "\n"
        )
        for idx, data in traces.items():
            rows = data["rows"]
            if not rows:
                raise RuntimeError(f"No trace recorded for motion {idx}")
            arrays = {key: np.stack([row[key] for row in rows]) for key in rows[0]}
            if "applied_torque_post_step" not in arrays or not np.any(np.abs(arrays["applied_torque_post_step"]) > 1e-6):
                raise RuntimeError(f"Missing or all-zero applied torque trace for motion {idx}")
            arrays["depth"] = np.stack(data["depth"])
            arrays["depth_frame"] = np.asarray(data["depth_frame"], dtype=np.int32)
            for key in ("depth_latency_frames", "depth_buffer_match_error"):
                arrays[key] = np.asarray(data[key], dtype=np.float32)
            for key in ("depth_camera_pos_w", "depth_camera_quat_xyzw"):
                arrays[key] = np.stack(data[key]).astype(np.float32)
            np.savez_compressed(trace_dir / f"motion_{idx:02d}.npz", **arrays)
        logger.info(f"Wrote {len(traces)} paired-action traces to {trace_dir}")
    names_path = Path(os.environ["PHP_EVAL_MOTION_NAMES_FILE"])
    names = [line.strip() for line in names_path.read_text().splitlines() if line.strip()]
    if len(names) != count:
        raise RuntimeError(f"Motion names ({len(names)}) do not match loaded clips ({count})")
    rows = []
    for idx, name in enumerate(names):
        success = int(successes[idx].item())
        failure = int(failures[idx].item())
        total = success + failure
        rows.append({
            "motion_id": idx,
            "motion": name,
            "episodes": total,
            "successes": success,
            "bad_tracking": failure,
            "simulator_timeouts_ignored": int(timeouts[idx].item()),
            "success_rate": success / total if total else None,
        })
        if diagnostic:
            records = failure_records[idx]
            rows[-1]["diagnostic"] = {
                "failure_frame_mean": mean(r["frame"] for r in records) if records else None,
                "failure_progress_mean": mean(r["progress_ratio"] for r in records) if records else None,
                "ref_pos_trigger_count": sum(r["ref_pos"] for r in records),
                "ref_ori_trigger_count": sum(r["ref_ori"] for r in records),
                "body_pos_trigger_count": sum(r["body_pos"] for r in records),
                "ref_pos_error_m_mean": mean(r["ref_pos_error_m"] for r in records) if records else None,
                "robot_path_projection_mean": mean(r["robot_path_projection_ratio"] for r in records) if records else None,
                "depth_finite_fraction_mean": mean(r["depth_finite_fraction"] for r in records) if records else None,
                "depth_image_std_mean": mean(r["depth_image_std"] for r in records) if records else None,
                "student_teacher_action_l2_mean": mean(
                    r["student_teacher_action_l2_mean"] for r in records
                    if r["student_teacher_action_l2_mean"] is not None
                ) if any(r["student_teacher_action_l2_mean"] is not None for r in records) else None,
                "teacher_takeover_steps_mean": mean(takeover_completed[idx]) if takeover_completed[idx] else None,
                "rescue_events_mean": mean(rescue_events_completed[idx]) if rescue_events_completed[idx] else None,
                "rescue_invalid_aborts_mean": mean(rescue_aborts_completed[idx]) if rescue_aborts_completed[idx] else None,
                "rescue_event_start_frames_samples": rescue_event_frames_completed[idx],
                "samples": records,
            }
    payload = {
        "schema_version": 1,
        "contract": "student_deterministic_frame0_full_clip",
        "profile": os.environ.get("PHP_EVAL_PROFILE", "clean"),
        "episodes_per_motion_target": target,
        "max_eval_steps": max_steps,
        "executed_steps": executed,
        "complete": all(row["episodes"] >= target for row in rows),
        "obstacle_seed": int(os.environ["PHP_EVAL_OBSTACLE_SEED"]),
        "diagnostics_enabled": diagnostic,
        "intervention": {
            "joint_ids": list(takeover_joint_ids) if takeover_joint_ids else "all",
            "mode": takeover_mode,
            "start_fraction": takeover_start_fraction if takeover_mode == "window" else None,
            "steps": takeover_steps if takeover_mode in {"window", "mapped_window", "rescue_events"} else None,
            "start_map": takeover_start_map if takeover_mode in {"mapped_tail", "mapped_window"} else None,
            "max_events": rescue_max_events if takeover_mode == "rescue_events" else None,
            "sustain": rescue_sustain if takeover_mode == "rescue_events" else None,
            "action_l2_threshold": rescue_threshold if takeover_mode == "rescue_events" else None,
            "cooldown_steps": rescue_cooldown_steps if takeover_mode == "rescue_events" else None,
        },
        "motions": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n")
    logger.info(f"Wrote student full-clip report to {output}")
    if not payload["complete"]:
        raise RuntimeError(f"Incomplete student full-clip report after {executed} steps")


def main() -> None:
    init_eval_logging()
    checkpoint_cfg, args = tyro.cli(CheckpointConfig, return_unknown_args=True, add_help=False)
    callbacks_cfg, args = tyro.cli(EvalCallbacksConfig, return_unknown_args=True, add_help=False, args=args)
    checkpoint_cfg = dataclasses.replace(
        checkpoint_cfg,
        checkpoint=resolve_checkpoint_reference(checkpoint_cfg.checkpoint),
    )
    saved_cfg, wandb_path = load_saved_experiment_config(checkpoint_cfg)
    config = parse_saved_config_overrides(saved_cfg.get_eval_config(), args)
    config = _apply_eval_profile(config)
    config = apply_preprocess_hook(use_php_isaacsim(config))
    env, device, app = setup_simulation_environment(config)
    try:
        log_dir = get_experiment_dir(config.logger, config.training, get_timestamp(), task_name="eval")
        log_dir.mkdir(parents=True, exist_ok=True)
        config.save_config(str(log_dir / CONFIG_NAME))
        callbacks = callbacks_cfg.collect_active_callbacks()
        if callbacks:
            object.__setattr__(config.algo.config, "eval_callbacks", callbacks)
        checkpoint = load_checkpoint(checkpoint_cfg.checkpoint, str(log_dir))
        algo = get_class(config.algo._target_)(
            device=device, env=env, config=config.algo.config, log_dir=str(log_dir), multi_gpu_cfg=None
        )
        algo.setup()
        algo.attach_checkpoint_metadata(saved_cfg, wandb_path)
        _load_student_only(algo, str(checkpoint))
        if os.environ.get("PHP_EVAL_DIAGNOSTICS", "0") == "1":
            state = torch.load(str(checkpoint), map_location=algo.device, weights_only=False)["model_state_dict"]
            teacher_state = {k[len("teachers.0."):]: v for k, v in state.items() if k.startswith("teachers.0.")}
            algo.policy.teachers[0].load_state_dict(teacher_state, strict=True)
            norm_state = {k[len("teacher_obs_normalizers.0."):]: v for k, v in state.items()
                          if k.startswith("teacher_obs_normalizers.0.")}
            algo.policy.teacher_obs_normalizers[0].load_state_dict(norm_state, strict=True)
            algo.policy.set_loaded_teachers([True])
        evaluate_full_clips(
            algo,
            Path(os.environ["PHP_EVAL_REPORT_OUTPUT"]),
            int(os.environ["PHP_EVAL_EPISODES_PER_MOTION"]),
            int(config.training.max_eval_steps),
        )
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        if app:
            close_simulation_app(app)


if __name__ == "__main__":
    main()
