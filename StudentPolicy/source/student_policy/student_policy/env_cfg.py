from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import MultiMeshRayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import whole_body_tracking.tasks.tracking.mdp as mdp
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.tracking_env_cfg import ObservationsCfg
from student_policy import observations as student_obs


DEPTH_HEIGHT = 58
DEPTH_WIDTH = 87
DEPTH_MAX_DISTANCE = 5.0
COMMAND_X_RANGE = (1.0, 1.0)
COMMAND_Y_RANGE = (0.0, 0.0)


@configclass
class StudentObservationsCfg(ObservationsCfg):
    """Student observations plus a frozen-teacher group for DAgger."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Student actor observations with depth immediately before previous actions."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.5, n_max=0.5))
        depth_image = ObsTerm(func=student_obs.depth_image, params={"sensor_cfg": SceneEntityCfg("depth_camera")})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class TeacherCfg(ObsGroup):
        """Frozen-teacher actor observations, matching Tracking-Flat-G1-v0 policy."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    teacher: TeacherCfg = TeacherCfg()


@configclass
class G1FlatStudentEnvCfg(G1FlatEnvCfg):
    """Student distillation environment from the parkour paper."""

    observations: StudentObservationsCfg = StudentObservationsCfg()

    def __post_init__(self):
        super().__post_init__()

        self.scene.depth_camera = MultiMeshRayCasterCameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/torso_link",
            offset=MultiMeshRayCasterCameraCfg.OffsetCfg(pos=(0.18, 0.0, 0.18), rot=(1.0, 0.0, 0.0, 0.0)),
            data_types=["distance_to_image_plane"],
            pattern_cfg=patterns.PinholeCameraPatternCfg(
                focal_length=24.0,
                horizontal_aperture=20.955,
                height=DEPTH_HEIGHT,
                width=DEPTH_WIDTH,
            ),
            mesh_prim_paths=["/World/ground"],
            max_distance=DEPTH_MAX_DISTANCE,
            update_period=1.0 / 30.0,
            debug_vis=False,
        )

        self.observations.policy.command.func = student_obs.velocity_command
        self.observations.policy.command.params = {"x_range": COMMAND_X_RANGE, "y_range": COMMAND_Y_RANGE}

        self.commands.motion.debug_vis = False
        self.commands.motion.adaptive_uniform_ratio = 1.0
        self.commands.motion.adaptive_alpha = 0.0


@configclass
class G1FlatStudentRelaxedEnvCfg(G1FlatStudentEnvCfg):
    """Student variant with relaxed tracking terminations."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.anchor_pos.params["threshold"] = 1.0
        if hasattr(self.terminations, "ee_body_pos"):
            self.terminations.ee_body_pos.params["threshold"] = 1.0
