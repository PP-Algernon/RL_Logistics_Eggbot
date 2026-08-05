"""Custom termination terms for the Eggtart mobile-grasp task."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def base_tipped(
    env: ManagerBasedRLEnv,
    min_up_proj: float = 0.5,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Terminate when the base has tipped over.

    Projects the world up-axis onto the robot base and terminates when the base z-axis no longer
    points sufficiently upward (``< min_up_proj``).
    """
    robot: Articulation = env.scene[robot_cfg.name]
    # Base local z-axis expressed in world frame.
    base_z_axis = torch.zeros((env.num_envs, 3), device=env.device)
    base_z_axis[:, 2] = 1.0
    base_z_world = quat_apply(robot.data.root_quat_w, base_z_axis)
    return base_z_world[:, 2] < min_up_proj
