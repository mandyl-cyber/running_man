"""Gym task registration for student policy training."""

import gymnasium as gym

from student_policy import env_cfg


gym.register(
    id="Tracking-Flat-G1-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": env_cfg.G1FlatStudentRelaxedEnvCfg,
        "rsl_rl_cfg_entry_point": "student_policy.rsl_rl_ppo_cfg:G1FlatStudentPPORunnerCfg",
    },
)

