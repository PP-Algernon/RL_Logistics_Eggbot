"""Moving-target event terms for the Eggtart mobile-grasp task.

The target is spawned with gravity disabled (see the env scene cfg), so once given a horizontal
velocity it coasts in a straight line at constant height -- a simple "moving target". The reset
event randomises its start pose + velocity; the interval event periodically re-randomises its
velocity so it changes direction during an episode.

For positioning/velocity at reset, the stock ``mdp.reset_root_state_uniform`` is used directly in
the env cfg; this module adds the periodic velocity re-randomisation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def randomize_target_velocity(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("target"),
) -> None:
    """Set a new random horizontal velocity on the target for the given envs.

    ``velocity_range`` maps axis name ("x", "y") to a (min, max) range in m/s. Unspecified axes
    are set to zero. Designed to be used as an ``interval`` event so the target changes heading
    mid-episode.
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = asset._ALL_INDICES  # type: ignore[attr-defined]

    n = len(env_ids)
    new_vel = torch.zeros((n, 6), device=env.device)  # (vx, vy, vz, wx, wy, wz)
    axis_to_col = {"x": 0, "y": 1, "z": 2}
    for axis, (lo, hi) in velocity_range.items():
        col = axis_to_col[axis]
        new_vel[:, col] = torch.empty(n, device=env.device).uniform_(lo, hi)

    root_vel = asset.data.root_vel_w[env_ids].clone()
    root_vel[:] = new_vel
    asset.write_root_velocity_to_sim(root_vel, env_ids=env_ids)
