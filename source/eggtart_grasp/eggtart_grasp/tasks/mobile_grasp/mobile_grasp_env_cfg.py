"""Base ManagerBasedRLEnvCfg for the Eggtart mobile-grasp task.

Scene: a Mecanum-wheeled mobile manipulator on a ground plane, chasing and grasping a moving
target object (a small floating cube standing in for the egg-tart payload).

The robot articulation itself is left ``MISSING`` here and filled in by the concrete config in
``config/eggtart/grasp_env_cfg.py``.
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import eggtart_grasp.tasks.mobile_grasp.mdp as mdp
from eggtart_grasp.assets.eggtart import (
    EGGTART_ARM_JOINT_NAMES,
    EGGTART_GRIPPER_JOINT_NAME,
    EGGTART_WHEEL_JOINT_NAMES,
)

# ---------------------------------------------------------------------------
# Tunable task constants (TODO tune)
# ---------------------------------------------------------------------------
GRASP_REACH_THRESHOLD = 0.05      # m -- EE within this distance counts as "at the target"
GRIPPER_CLOSED_THRESHOLD = -0.6   # rad -- gripper joint below this counts as "closed"


##
# Scene definition
##
@configclass
class MobileGraspSceneCfg(InteractiveSceneCfg):
    """Scene: ground, light, mobile manipulator, and a moving target."""

    # ground plane
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # robot -- filled in by the concrete config
    robot: ArticulationCfg = MISSING

    # moving target: a small cube with gravity disabled so it coasts at constant height
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.05, 0.05, 0.05),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.2)),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Action terms: 3-D Mecanum base velocity + arm joint positions + gripper."""

    base_velocity = mdp.MecanumBaseActionCfg(
        asset_name="robot",
        wheel_joint_names=EGGTART_WHEEL_JOINT_NAMES,
    )
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=EGGTART_ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
    gripper_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[EGGTART_GRIPPER_JOINT_NAME],
        scale=1.0,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        # proprioception
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        # task
        target_position_b = ObsTerm(
            func=mdp.target_position_in_base_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")},
        )
        ee_to_target_b = ObsTerm(
            func=mdp.ee_to_target_vector_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "target_cfg": SceneEntityCfg("target"),
            },
        )
        target_lin_vel = ObsTerm(
            func=mdp.target_lin_vel_w, params={"target_cfg": SceneEntityCfg("target")}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Reset + interval events."""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # spawn the target somewhere in front of / around the robot, at graspable height
            "pose_range": {"x": (0.4, 1.2), "y": (-0.6, 0.6), "z": (0.15, 0.30)},
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )
    # periodically change the target heading so it is a genuinely moving target
    randomize_target_velocity = EventTerm(
        func=mdp.randomize_target_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class RewardsCfg:
    """Staged reward terms (TODO tune weights)."""

    # stage 1: base approach
    base_approach = RewTerm(
        func=mdp.base_to_target_xy_tanh,
        weight=1.0,
        params={"std": 0.5, "robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")},
    )
    # stage 2: end-effector reach
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=2.0,
        params={
            "std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    ee_distance = RewTerm(
        func=mdp.ee_to_target_distance_l2,
        weight=-0.1,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    # stage 3: grasp
    grasp = RewTerm(
        func=mdp.grasp_bonus,
        weight=5.0,
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    # stage 4: retract after grasping
    retract = RewTerm(
        func=mdp.retract_bonus,
        weight=2.0,
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "std": 0.5,
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
        },
    )
    # penalties
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.001)
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.0001,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES)},
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_tipped = DoneTerm(func=mdp.base_tipped, params={"min_up_proj": 0.5})


##
# Environment configuration
##
@configclass
class MobileGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Base configuration for the Eggtart mobile-grasp environment."""

    # Scene
    scene: MobileGraspSceneCfg = MobileGraspSceneCfg(num_envs=2048, env_spacing=3.0)
    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        # general settings
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 3.0)
        # simulation settings
        self.sim.dt = 1.0 / 120.0
