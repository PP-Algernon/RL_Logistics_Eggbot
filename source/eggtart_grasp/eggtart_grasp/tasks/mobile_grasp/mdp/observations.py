"""Custom observation terms for the Eggtart mobile-grasp task.

All task-relative quantities are expressed in the robot base frame so the policy is invariant to
the robot's world pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import Camera
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def target_position_in_base_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Position of the moving target expressed in the robot base frame. Shape (N, 3)."""
    robot: Articulation = env.scene[robot_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target.data.root_pos_w
    )
    return target_pos_b


def ee_position_in_base_frame(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
) -> torch.Tensor:
    """Position of the end-effector body expressed in the robot base frame. Shape (N, 3)."""
    robot: Articulation = env.scene[ee_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_to_target_vector_base_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """Vector from the end-effector to the target, in base frame. Shape (N, 3)."""
    return target_position_in_base_frame(env, robot_cfg, target_cfg) - ee_position_in_base_frame(env, ee_cfg)


def target_lin_vel_w(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """World-frame linear velocity of the target. Shape (N, 3)."""
    target: RigidObject = env.scene[target_cfg.name]
    return target.data.root_lin_vel_w


def camera_rgb(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
) -> torch.Tensor:
    """RGB image from camera. Shape (N, H, W, 3)."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    # Get RGB data and normalize to [0, 1]
    rgb_data = camera.data.output["rgb"]  # Shape: (N, H, W, 3) or (N, H, W, 4) with alpha
    # Remove alpha channel if present
    if rgb_data.shape[-1] == 4:
        rgb_data = rgb_data[..., :3]
    return rgb_data


def camera_depth(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("wrist_camera"),
) -> torch.Tensor:
    """Depth image from camera. Shape (N, H, W, 1)."""
    camera: Camera = env.scene.sensors[sensor_cfg.name]
    depth_data = camera.data.output["distance_to_camera"]
    # Add channel dimension if not present
    if len(depth_data.shape) == 3:
        depth_data = depth_data.unsqueeze(-1)
    return depth_data
