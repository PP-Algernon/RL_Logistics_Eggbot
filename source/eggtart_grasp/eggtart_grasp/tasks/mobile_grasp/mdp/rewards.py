"""Staged reward terms for the Eggtart mobile-grasp task.

Stage shaping (all dense unless noted):
  1. ``base_to_target_xy_tanh``  -- drive the base toward the target (horizontal plane)
  2. ``ee_to_target_tanh``       -- reach the end-effector to the target
  3. ``grasp_bonus``             -- sparse-ish bonus when the EE is close AND the gripper is closed
  4. ``retract_bonus``           -- once "grasping", reward pulling the arm back to its home pose

.. note::
    Without a latched grasp flag or a physical attach constraint, "grasping" is evaluated
    instantaneously each step and the target is not yet rigidly attached to the gripper.
    See the env-cfg TODOs for upgrading to a true pick-and-carry.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _ee_to_target_distance(
    env: ManagerBasedRLEnv, ee_cfg: SceneEntityCfg, target_cfg: SceneEntityCfg
) -> torch.Tensor:
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    return torch.norm(ee_pos_w - target.data.root_pos_w, dim=1)


def base_to_target_xy_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward driving the base toward the target in the horizontal (xy) plane."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    dist_xy = torch.norm(robot.data.root_pos_w[:, :2] - target.data.root_pos_w[:, :2], dim=1)
    return 1.0 - torch.tanh(dist_xy / std)


def ee_to_target_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Reward reaching the end-effector to the target (tanh kernel)."""
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg)
    return 1.0 - torch.tanh(dist / std)


def ee_to_target_distance_l2(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """L2 distance from end-effector to target (use with a negative weight)."""
    return _ee_to_target_distance(env, ee_cfg, target_cfg)


def grasp_bonus(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    gripper_closed_threshold: float,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Bonus (1.0) when the EE is within ``reach_threshold`` AND the gripper joint is closed."""
    robot: Articulation = env.scene[ee_cfg.name]
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    is_near = dist < reach_threshold
    is_closed = gripper_pos < gripper_closed_threshold
    return (is_near & is_closed).float()


def retract_bonus(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    gripper_closed_threshold: float,
    std: float,
    arm_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["link_00[1-5]_joint"]),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    gripper_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"]),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Once "grasping", reward the arm joints returning to their home (default) pose."""
    robot: Articulation = env.scene[arm_cfg.name]
    dist = _ee_to_target_distance(env, ee_cfg, target_cfg)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    grasping = (dist < reach_threshold) & (gripper_pos < gripper_closed_threshold)

    arm_pos = robot.data.joint_pos[:, arm_cfg.joint_ids]
    arm_home = robot.data.default_joint_pos[:, arm_cfg.joint_ids]
    home_err = torch.norm(arm_pos - arm_home, dim=1)
    return grasping.float() * (1.0 - torch.tanh(home_err / std))
