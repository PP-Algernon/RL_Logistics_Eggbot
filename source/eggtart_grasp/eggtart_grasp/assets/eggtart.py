"""Eggtart mobile manipulator ``ArticulationCfg`` for Isaac Lab.

Robot specs (read from ``urdf/robot.urdf``):
  - Mobile base: 4 Mecanum wheels -> continuous joints ``wheel_001_joint`` .. ``wheel_004_joint``
        (effort 10 Nm, velocity 5 rad/s)
  - 5-axis arm: revolute joints ``link_001_joint`` .. ``link_005_joint`` (effort 10 Nm, vel 5 rad/s)
  - Gripper: ``end_effector_joint`` (1-DOF servo axis, modelled as the gripper open/close DOF)
  - End-effector body: ``end_effector``;  base body: ``base_link``

Wheel geometry (from joint origins in the URDF):
    FL = wheel_003, FR = wheel_002, RL = wheel_004, RR = wheel_001
    half wheelbase  lx ~= 0.08 m   (front<->rear, along +x)
    half track      ly ~= 0.097 m  (left<->right, along +y)
    wheel radius    r  ~= 0.043 m   (wheel axle height)
    # TODO(calibrate): re-measure lx/ly/r against the real robot before serious training.

.. note::
    The URDF wheels are plain cylinders -- they have NO Mecanum roller geometry, so true
    holonomic side-slip is not physically simulated. The :class:`MecanumBaseAction` term maps a
    commanded body velocity to the 4 wheel velocity targets using ideal Mecanum kinematics; this
    is a trainable starting point, not a high-fidelity model. Add roller meshes (or switch the base
    to a holonomic root-velocity drive) if you need faithful lateral motion.
"""

from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg

# ---------------------------------------------------------------------------
# Joint / body name groups (imported by the env + action configs)
# ---------------------------------------------------------------------------
EGGTART_ARM_JOINT_NAMES = [
    "link_001_joint",
    "link_002_joint",
    "link_003_joint",
    "link_004_joint",
    "link_005_joint",
]
EGGTART_GRIPPER_JOINT_NAME = "end_effector_joint"
# Ordered FL, FR, RL, RR for the Mecanum kinematics in MecanumBaseAction.
EGGTART_WHEEL_JOINT_NAMES = [
    "wheel_003_joint",  # front-left
    "wheel_002_joint",  # front-right
    "wheel_004_joint",  # rear-left
    "wheel_001_joint",  # rear-right
]
EGGTART_EE_BODY_NAME = "end_effector"
EGGTART_BASE_BODY_NAME = "base_link"

# ---------------------------------------------------------------------------
# Path resolution -- URDF lives next to this file under ``urdf/``
# ---------------------------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
EGGTART_URDF_PATH = os.path.join(_THIS_DIR, "urdf", "robot.urdf")

# ---------------------------------------------------------------------------
# Nominal joint positions (rad)
#   Arm tucked near zero, gripper open. TODO(tune): set a real "carry/home" pose.
# ---------------------------------------------------------------------------
EGGTART_GRIPPER_OPEN = 0.0
EGGTART_GRIPPER_CLOSED = -1.2  # TODO(tune): closed angle that grips the target

EGGTART_NOMINAL_JOINT_POS = {
    "link_001_joint": 0.0,
    "link_002_joint": -0.6,
    "link_003_joint": 0.6,
    "link_004_joint": 0.0,
    "link_005_joint": 0.0,
    "end_effector_joint": EGGTART_GRIPPER_OPEN,
    "wheel_.*_joint": 0.0,
}

# ---------------------------------------------------------------------------
# ArticulationCfg
# ---------------------------------------------------------------------------
EGGTART_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=EGGTART_URDF_PATH,
        # USD cache is written next to the URDF on first conversion.
        usd_dir=os.path.join(_THIS_DIR, "urdf", "usd_cache"),
        fix_base=False,  # mobile base -- free-floating root
        merge_fixed_joints=True,
        # Default converter drive gains; per-joint control is set by the actuators below.
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            drive_type="force",
            target_type="position",
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=40.0,
                damping=2.0,
            ),
        ),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.06),  # base spawn height; TODO(tune): match wheel radius so wheels rest on ground
        joint_pos=EGGTART_NOMINAL_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # 5-axis arm: position-controlled (stiff PD).
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["link_00[1-5]_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=40.0,
            damping=2.0,
        ),
        # Gripper servo axis: position-controlled (open/close).
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["end_effector_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=30.0,
            damping=1.5,
        ),
        # Mecanum wheels: velocity-controlled (stiffness=0 -> pure velocity tracking).
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=["wheel_00[1-4]_joint"],
            effort_limit=10.0,
            velocity_limit=5.0,
            stiffness=0.0,
            damping=2.0,
        ),
    },
    soft_joint_pos_limit_factor=0.95,
)
