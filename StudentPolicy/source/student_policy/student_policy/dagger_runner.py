"""Student distillation runner: PPO plus DAgger action supervision."""

from __future__ import annotations

import os
import time
from collections import deque

import torch
import torch.nn.functional as F
from isaaclab.managers import SceneEntityCfg
from rsl_rl.modules import ActorCritic
from rsl_rl.runners import OnPolicyRunner
from student_policy.depth_actor_critic import StudentDepthActorCritic
from whole_body_tracking.tasks.tracking import mdp


class StudentDaggerOnPolicyRunner(OnPolicyRunner):
    """RSL-RL runner with a frozen teacher queried on a separate obs group."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)
        runner_cfg = train_cfg.get("runner", {})
        policy_cfg = train_cfg.get("policy", {})
        algorithm_cfg = train_cfg.get("algorithm", {})

        self._replace_student_actor_critic(env, policy_cfg, algorithm_cfg)

        self.teacher_checkpoint = runner_cfg.get("teacher_checkpoint") or train_cfg.get("teacher_checkpoint")
        if self.teacher_checkpoint is None:
            raise ValueError("Student training requires --teacher_checkpoint.")

        self.dagger_loss_coef = float(runner_cfg.get("dagger_loss_coef", train_cfg.get("dagger_loss_coef", 10.0)))
        self.dagger_curriculum_iterations = int(
            runner_cfg.get("dagger_curriculum_iterations", train_cfg.get("dagger_curriculum_iterations", 10000))
        )
        self.expert_anchor_pos_threshold = float(
            runner_cfg.get(
                "expert_anchor_pos_threshold",
                train_cfg.get("expert_anchor_pos_threshold", train_cfg.get("expert_termination_threshold", 0.5)),
            )
        )
        self.expert_anchor_ori_threshold = float(
            runner_cfg.get("expert_anchor_ori_threshold", train_cfg.get("expert_anchor_ori_threshold", 0.8))
        )
        self.expert_ee_pos_threshold = float(
            runner_cfg.get("expert_ee_pos_threshold", train_cfg.get("expert_ee_pos_threshold", 0.8))
        )
        self.enable_kl_after_iteration = int(
            runner_cfg.get("enable_kl_after_iteration", train_cfg.get("enable_kl_after_iteration", 1000))
        )
        self._original_schedule = getattr(self.alg, "schedule", "fixed")

        teacher_obs = self._get_teacher_observations()
        teacher_obs_dict = {"policy": teacher_obs, "critic": teacher_obs}
        teacher_obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        teacher_hidden_dims = runner_cfg.get("teacher_actor_hidden_dims", [512, 256, 128])
        activation = policy_cfg.get("activation", "elu") if isinstance(policy_cfg, dict) else "elu"
        init_noise_std = policy_cfg.get("init_noise_std", 1.0) if isinstance(policy_cfg, dict) else 1.0
        self.teacher_actor_critic = ActorCritic(
            teacher_obs_dict,
            teacher_obs_groups,
            self.env.num_actions,
            actor_hidden_dims=teacher_hidden_dims,
            critic_hidden_dims=teacher_hidden_dims,
            activation=activation,
            init_noise_std=init_noise_std,
        ).to(self.device)
        self._load_teacher_checkpoint(self.teacher_checkpoint)
        self.teacher_actor_critic.eval()

    def _replace_student_actor_critic(self, env, policy_cfg: dict, algorithm_cfg: dict):
        obs = env.get_observations()
        obs_groups = self.cfg.get("obs_groups", {"policy": ["policy"], "critic": ["critic"]})
        num_actor_obs = sum(obs[group].shape[-1] for group in obs_groups["policy"])
        num_critic_obs = sum(obs[group].shape[-1] for group in obs_groups["critic"])
        num_actions = env.num_actions

        actor_critic = StudentDepthActorCritic(
            num_actor_obs=num_actor_obs,
            num_critic_obs=num_critic_obs,
            num_actions=num_actions,
            obs_groups=obs_groups,
            depth_height=policy_cfg.get("depth_height", 58),
            depth_width=policy_cfg.get("depth_width", 87),
            depth_dim=policy_cfg.get("depth_dim", 32),
            actor_hidden_dims=policy_cfg.get("actor_hidden_dims", [2048, 1024, 512, 256, 128]),
            critic_hidden_dims=policy_cfg.get("critic_hidden_dims", [512, 256, 128]),
            activation=policy_cfg.get("activation", "elu"),
            init_noise_std=policy_cfg.get("init_noise_std", 0.01),
        ).to(self.device)

        self.alg.policy = actor_critic
        self.alg.actor_critic = actor_critic
        learning_rate = algorithm_cfg.get("learning_rate", getattr(self.alg, "learning_rate", 3.0e-4))
        self.alg.optimizer = torch.optim.Adam(self.alg.policy.parameters(), lr=learning_rate)

    def _load_teacher_checkpoint(self, path: str):
        path = os.path.abspath(path)
        checkpoint = torch.load(path, map_location=self.device)
        if "actor_model_state_dict" in checkpoint:
            raise RuntimeError(
                "This is a Holosoma/PHP teacher checkpoint, but the legacy StudentPolicy "
                "runner only supports RSL-RL teachers with model_state_dict. Do not remap "
                "the tensors: the official teacher consumes a different 449-D observation "
                "contract. Use scripts/php_student/run_student.sh preflight/train instead."
            )
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model_state = self.teacher_actor_critic.state_dict()
        self._check_teacher_actor_input_dim(state_dict, model_state, path)
        compatible_state = {
            key: value for key, value in state_dict.items() if key in model_state and model_state[key].shape == value.shape
        }
        skipped = sorted(set(state_dict) - set(compatible_state))
        skipped_actor = [key for key in skipped if key.startswith("actor.")]
        if skipped_actor:
            raise RuntimeError(
                "Teacher actor weights are incompatible with the student teacher-observation group. "
                f"First skipped actor tensors: {skipped_actor[:5]}"
            )
        loaded_actor = [key for key in compatible_state if key.startswith("actor.")]
        if not loaded_actor:
            raise RuntimeError("No teacher actor tensors were loaded from the checkpoint.")
        self.teacher_actor_critic.load_state_dict(compatible_state, strict=False)
        print(f"[INFO]: Loaded frozen teacher checkpoint: {path}")
        if skipped:
            print(f"[INFO]: Skipped {len(skipped)} incompatible teacher tensors, usually critic-only weights.")

    def _check_teacher_actor_input_dim(self, state_dict: dict, model_state: dict, path: str):
        actor_input_keys = [
            key
            for key, value in state_dict.items()
            if key.startswith("actor.") and key.endswith(".weight") and hasattr(value, "ndim") and value.ndim == 2
        ]
        if not actor_input_keys:
            return

        first_actor_key = sorted(actor_input_keys)[0]
        if first_actor_key not in model_state:
            return

        checkpoint_input_dim = state_dict[first_actor_key].shape[1]
        expected_input_dim = model_state[first_actor_key].shape[1]
        if checkpoint_input_dim != expected_input_dim:
            teacher_obs_dim = self._get_teacher_observations().shape[-1]
            raise RuntimeError(
                "Teacher checkpoint actor input dimension does not match the student teacher-observation group. "
                f"Checkpoint: {path}. "
                f"Checkpoint actor input dim: {checkpoint_input_dim}; "
                f"expected: {expected_input_dim}; current teacher obs dim: {teacher_obs_dim}. "
                "Use a Tracking-Flat-G1-v0 teacher checkpoint from the current new_teacherpolicy, "
                "or update StudentObservationsCfg.TeacherCfg to mirror that teacher actor observation order."
            )

    def _unwrap_env(self):
        env = self.env
        while hasattr(env, "env"):
            env = env.env
        return getattr(env, "unwrapped", env)

    def _get_teacher_observations(self) -> torch.Tensor:
        base_env = self._unwrap_env()
        if not hasattr(base_env, "observation_manager"):
            raise RuntimeError("StudentDaggerOnPolicyRunner requires an Isaac Lab manager-based environment.")
        return base_env.observation_manager.compute_group("teacher").to(self.device)

    def _get_teacher_observation_dict(self) -> dict[str, torch.Tensor]:
        teacher_obs = self._get_teacher_observations()
        return {"policy": teacher_obs, "critic": teacher_obs}

    def _dagger_weight(self, iteration: int) -> float:
        if self.dagger_curriculum_iterations <= 0:
            return 0.1
        progress = min(max(iteration / float(self.dagger_curriculum_iterations), 0.0), 1.0)
        return max(1.0 - 0.9 * progress, 0.1)

    def _expert_mask(self) -> torch.Tensor:
        base_env = self._unwrap_env()
        # Match the teacher's actual loss-of-track contract.  The previous
        # z-only test incorrectly kept x/y-diverged expert labels in DAgger.
        expert_failed = mdp.bad_anchor_pos(
            base_env,
            command_name="motion",
            threshold=self.expert_anchor_pos_threshold,
        )
        expert_failed |= mdp.bad_anchor_ori(
            base_env,
            asset_cfg=SceneEntityCfg("robot"),
            command_name="motion",
            threshold=self.expert_anchor_ori_threshold,
        )
        expert_failed |= mdp.bad_motion_body_pos_z_only(
            base_env,
            command_name="motion",
            threshold=self.expert_ee_pos_threshold,
            body_names=[
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ],
        )
        return (~expert_failed).float().unsqueeze(-1)

    def _actor_mean(self, obs: torch.Tensor) -> torch.Tensor:
        self.alg.policy.act(obs)
        return self.alg.policy.action_mean

    def _update_dagger(self, obs_buf: torch.Tensor, teacher_action_buf: torch.Tensor, mask_buf: torch.Tensor, iteration: int):
        obs = obs_buf.flatten(0, 1)
        teacher_actions = teacher_action_buf.flatten(0, 1)
        mask = mask_buf.flatten(0, 1)
        valid_count = torch.clamp(mask.sum(), min=1.0)
        weight = self._dagger_weight(iteration)

        student_actions = self._actor_mean(obs)
        per_env_loss = F.mse_loss(student_actions, teacher_actions, reduction="none").mean(dim=-1, keepdim=True)
        dagger_loss = (per_env_loss * mask).sum() / valid_count
        loss = self.dagger_loss_coef * weight * dagger_loss

        self.alg.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.alg.policy.parameters(), self.alg.max_grad_norm)
        self.alg.optimizer.step()
        return float(dagger_loss.detach().cpu()), weight, float(mask.mean().detach().cpu())

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        self._prepare_logging_writer()
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations()
        obs = {key: value.to(self.device) for key, value in obs.items()}
        self.train_mode()

        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)

        start_iter = self.current_learning_iteration
        tot_iter = self.current_learning_iteration + num_learning_iterations
        for it in range(self.current_learning_iteration, tot_iter):
            if hasattr(self.alg, "schedule") and it < self.enable_kl_after_iteration:
                self.alg.schedule = "fixed"
            elif hasattr(self.alg, "schedule"):
                self.alg.schedule = self._original_schedule

            policy_obs = self.alg.policy._get_actor_obs(obs)
            obs_buf = torch.zeros(self.num_steps_per_env, self.env.num_envs, policy_obs.shape[1], device=self.device)
            teacher_action_buf = torch.zeros(
                self.num_steps_per_env, self.env.num_envs, self.env.num_actions, device=self.device
            )
            mask_buf = torch.zeros(self.num_steps_per_env, self.env.num_envs, 1, device=self.device)

            start = time.time()
            with torch.inference_mode():
                for step in range(self.num_steps_per_env):
                    teacher_actions = self.teacher_actor_critic.act_inference(self._get_teacher_observation_dict())
                    obs_buf[step] = self.alg.policy._get_actor_obs(obs)
                    teacher_action_buf[step] = teacher_actions
                    mask_buf[step] = self._expert_mask()

                    actions = self.alg.act(obs)
                    obs, rewards, dones, infos = self.env.step(actions.to(self.env.device))
                    obs = {key: value.to(self.device) for key, value in obs.items()}
                    rewards, dones = rewards.to(self.device), dones.to(self.device)
                    self.alg.process_env_step(obs, rewards, dones, infos)

                    if self.log_dir is not None:
                        if "episode" in infos:
                            ep_infos.append(infos["episode"])
                        cur_reward_sum += rewards
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

            collection_time = time.time() - start
            start = time.time()
            self.alg.compute_returns(obs)
            loss_dict = self.alg.update()
            mean_dagger_loss, dagger_weight, dagger_valid_ratio = self._update_dagger(
                obs_buf, teacher_action_buf, mask_buf, it
            )
            learn_time = time.time() - start

            if self.log_dir is not None:
                self.writer.add_scalar("Loss/dagger", mean_dagger_loss, it)
                self.writer.add_scalar("Loss/dagger_weight", dagger_weight, it)
                self.writer.add_scalar("Loss/dagger_valid_ratio", dagger_valid_ratio, it)
                self.log(locals())
            if it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))
            ep_infos.clear()

        self.current_learning_iteration += num_learning_iterations
        self.save(os.path.join(self.log_dir, f"model_{self.current_learning_iteration}.pt"))
