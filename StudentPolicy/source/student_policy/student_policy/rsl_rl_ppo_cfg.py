from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlStudentDepthActorCriticCfg(RslRlPpoActorCriticCfg):
    depth_height = 58
    depth_width = 87
    depth_dim = 32


@configclass
class G1FlatStudentPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """DAgger + PPO student distillation settings from the parkour paper."""

    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "g1_flat_student"
    empirical_normalization = False
    policy = RslRlStudentDepthActorCriticCfg(
        init_noise_std=0.01,
        actor_hidden_dims=[2048, 1024, 512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.001,
        num_learning_epochs=2,
        num_mini_batches=96,
        learning_rate=3.0e-4,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )

    teacher_checkpoint = None
    dagger_loss_coef = 10.0
    dagger_curriculum_iterations = 10000
    expert_anchor_pos_threshold = 0.5
    expert_anchor_ori_threshold = 0.8
    expert_ee_pos_threshold = 0.8
    enable_kl_after_iteration = 1000
