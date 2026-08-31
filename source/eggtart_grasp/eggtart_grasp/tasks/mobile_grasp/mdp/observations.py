"""Custom observation terms for the Eggtart mobile-grasp task.

All task-relative quantities are expressed in the robot base frame so the policy is invariant to
the robot's world pose.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms

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
    grasp_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Position of the grasp point expressed in the robot base frame. Shape (N, 3).

    ``grasp_offset`` 是真实抓取点相对 ``end_effector`` body 原点的局部偏移
    （两爪之间，见 assets/eggtart.py 的 EGGTART_EE_GRASP_OFFSET）。
    观测必须和奖励用同一个点，否则策略看到的"手在哪"和被奖励的点差 2.8 cm。
    """
    robot: Articulation = env.scene[ee_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    if grasp_offset is not None:
        ee_quat_w = robot.data.body_quat_w[:, ee_cfg.body_ids[0]]
        off = torch.tensor(grasp_offset, device=ee_pos_w.device, dtype=ee_pos_w.dtype)
        ee_pos_w = ee_pos_w + quat_apply(ee_quat_w, off.unsqueeze(0).expand(ee_pos_w.shape[0], -1))
    ee_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, ee_pos_w)
    return ee_pos_b


def ee_to_target_vector_base_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="end_effector"),
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
    grasp_offset: tuple[float, float, float] | None = None,
) -> torch.Tensor:
    """Vector from the grasp point to the target, in base frame. Shape (N, 3)."""
    return target_position_in_base_frame(env, robot_cfg, target_cfg) - ee_position_in_base_frame(
        env, ee_cfg, grasp_offset
    )


def target_lin_vel_w(
    env: ManagerBasedRLEnv,
    target_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> torch.Tensor:
    """World-frame linear velocity of the target. Shape (N, 3)."""
    target: RigidObject = env.scene[target_cfg.name]
    return target.data.root_lin_vel_w
