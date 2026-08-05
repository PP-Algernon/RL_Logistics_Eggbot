"""Custom Mecanum base ActionTerm for the Eggtart mobile manipulator.

The policy outputs a 3-D body velocity command ``(v_x, v_y, omega_z)`` which is mapped to the
4 Mecanum wheel velocity targets via ideal inverse kinematics:

    omega_i = (1 / r) * [ v_x  -+ v_y  -+ (lx + ly) * omega_z ]   (signs per wheel)

Wheel order is FL, FR, RL, RR (see ``EGGTART_WHEEL_JOINT_NAMES`` in assets/eggtart.py).

.. note::
    The URDF wheels are plain cylinders without roller geometry, so lateral (v_y) commands do
    not produce physically-correct side-slip. Treat this term as a trainable approximation and
    calibrate ``wheel_radius`` / ``half_wheelbase`` / ``half_track`` / ``wheel_spin_sign`` against
    the real platform (all marked TODO).
"""

from __future__ import annotations

from dataclasses import MISSING, field
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class MecanumBaseAction(ActionTerm):
    """Maps a 3-D body velocity command to 4 Mecanum wheel velocity targets."""

    cfg: MecanumBaseActionCfg
    _asset: Articulation

    def __init__(self, cfg: MecanumBaseActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        # Resolve the 4 wheel joints in FL, FR, RL, RR order.
        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            cfg.wheel_joint_names, preserve_order=True
        )
        assert len(self._wheel_ids) == 4, f"Expected 4 wheel joints, got {self._wheel_names}"

        # Geometry constants.
        self._r = cfg.wheel_radius
        self._k = cfg.half_wheelbase + cfg.half_track  # (lx + ly)

        # Command scaling: policy output (~[-1, 1]) -> body velocity.
        self._vel_scale = torch.tensor(
            [cfg.max_lin_vel_x, cfg.max_lin_vel_y, cfg.max_ang_vel_z], device=env.device
        )

        # Per-wheel spin sign to absorb the URDF joint-axis orientation (see module docstring).
        self._spin_sign = torch.tensor(cfg.wheel_spin_sign, device=env.device)

        # Mecanum inverse-kinematics sign matrix, rows = [FL, FR, RL, RR], cols = [vx, vy, wz].
        # omega_i = (1/r) * (s_vx * vx + s_vy * vy + s_wz * (lx+ly) * wz)
        self._ik = torch.tensor(
            [
                [1.0, -1.0, -1.0],  # FL
                [1.0, +1.0, +1.0],  # FR
                [1.0, +1.0, -1.0],  # RL
                [1.0, -1.0, +1.0],  # RR
            ],
            device=env.device,
        )

        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._wheel_vel = torch.zeros(env.num_envs, 4, device=env.device)

    # ------------------------------------------------------------------
    # ActionTerm interface
    # ------------------------------------------------------------------
    @property
    def action_dim(self) -> int:
        return 3  # (v_x, v_y, omega_z)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        # Clip to [-1, 1] then scale to body velocities (num_envs, 3).
        body_vel = torch.clamp(actions, -1.0, 1.0) * self._vel_scale
        self._processed_actions = body_vel

        # body_vel scaled by IK weights: apply (lx+ly) only to the wz column.
        scaled = body_vel.clone()
        scaled[:, 2] = scaled[:, 2] * self._k
        # wheel angular velocity targets: (num_envs, 4)
        wheel_vel = (scaled @ self._ik.T) / self._r
        self._wheel_vel = wheel_vel * self._spin_sign

    def apply_actions(self) -> None:
        self._asset.set_joint_velocity_target(self._wheel_vel, joint_ids=self._wheel_ids)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._wheel_vel.zero_()
        else:
            self._wheel_vel[env_ids] = 0.0


@configclass
class MecanumBaseActionCfg(ActionTermCfg):
    """Configuration for :class:`MecanumBaseAction`."""

    class_type: type[ActionTerm] = MecanumBaseAction

    asset_name: str = MISSING
    """Name of the articulation asset in the scene."""

    wheel_joint_names: list[str] = MISSING
    """Wheel joint names in FL, FR, RL, RR order."""

    # --- geometry (TODO calibrate against the real robot) ---
    wheel_radius: float = 0.043
    """Wheel radius r (m)."""

    half_wheelbase: float = 0.08
    """Half front<->rear wheel separation lx (m)."""

    half_track: float = 0.097
    """Half left<->right wheel separation ly (m)."""

    wheel_spin_sign: list[float] = field(default_factory=lambda: [1.0, 1.0, -1.0, -1.0])
    """Per-wheel sign (FL, FR, RL, RR) absorbing the URDF joint-axis direction."""

    # --- command limits ---
    max_lin_vel_x: float = 0.6
    """Max forward/backward body velocity (m/s)."""

    max_lin_vel_y: float = 0.6
    """Max lateral body velocity (m/s)."""

    max_ang_vel_z: float = 1.5
    """Max yaw rate (rad/s)."""
