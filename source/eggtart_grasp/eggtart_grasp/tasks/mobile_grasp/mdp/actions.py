"""Eggtart 移动机械臂的自定义动作项

包含两种底盘控制方式：
1. MecanumBaseAction: 麦轮逆运动学控制（已弃用，保留作为参考）
2. HolonomicBaseAction: 全向速度控制（当前使用）

全向控制模式：
    策略输出 3 维体速度指令 (v_x, v_y, omega_z)，直接设置为底盘根节点的速度。
    不模拟真实麦轮动力学，简化仿真，提供理想全向移动能力。

麦轮控制模式（已弃用）：
    策略输出 3 维体速度指令 (v_x, v_y, omega_z)，通过理想逆运动学映射到 4 个麦轮速度：
        omega_i = (1 / r) * [ v_x  ± v_y  ± (lx + ly) * omega_z ]   (符号随轮子而异)
    轮子顺序: FL, FR, RL, RR (参见 assets/eggtart.py 中的 EGGTART_WHEEL_JOINT_NAMES)

注意：
    URDF 中的轮子是简单圆柱体，没有麦轮滚子几何，因此侧向滑移 (v_y) 不符合物理真实性。
    如果需要高保真侧向运动，考虑添加滚子网格或使用全向根速度驱动。
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import field
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


class MecanumBaseAction(ActionTerm):
    """将 3 维体速度指令映射到 4 个麦轮速度目标（已弃用，保留作为参考）
    注意：当前项目使用 HolonomicBaseAction,此类保留仅供参考。
    """

    cfg: MecanumBaseActionCfg
    _asset: Articulation

    def __init__(self, cfg: MecanumBaseActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        # 解析 4 个轮子关节，顺序：FL, FR, RL, RR
        self._wheel_ids, self._wheel_names = self._asset.find_joints(
            cfg.wheel_joint_names, preserve_order=True
        )
        assert len(self._wheel_ids) == 4, f"Expected 4 wheel joints, got {self._wheel_names}"

        # 几何常数
        self._r = cfg.wheel_radius
        self._k = cfg.half_wheelbase + cfg.half_track  # (lx + ly)

        # 指令缩放：策略输出 (~[-1, 1]) -> 体速度
        self._vel_scale = torch.tensor(
            [cfg.max_lin_vel_x, cfg.max_lin_vel_y, cfg.max_ang_vel_z], device=env.device
        )

        # 每个轮子的旋转符号，吸收 URDF 关节轴方向（见模块文档）
        self._spin_sign = torch.tensor(cfg.wheel_spin_sign, device=env.device)

        # 麦轮逆运动学符号矩阵，行 = [FL, FR, RL, RR], 列 = [vx, vy, wz]
        # omega_i = (1/r) * (s_vx * vx + s_vy * vy + s_wz * (lx+ly) * wz)
        self._ik = torch.tensor(
            [
                [1.0, -1.0, -1.0],  # FL (左前)
                [1.0, +1.0, +1.0],  # FR (右前)
                [1.0, +1.0, -1.0],  # RL (左后)
                [1.0, -1.0, +1.0],  # RR (右后)
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

    asset_name: str = 'eggtart'
    """Name of the articulation asset in the scene."""

    wheel_joint_names: list[str] = ['FL', 'FR', 'RL', 'RR']
    """Wheel joint names in FL, FR, RL, RR order."""

    # --- geometry (calibrate against the real robot) ---
    wheel_radius: float = 0.0485
    """Wheel radius r (m)."""

    half_wheelbase: float = 0.096
    """Half front<->rear wheel separation lx (m)."""

    half_track: float = 0.08859
    """Half left<->right wheel separation ly (m)."""

    wheel_spin_sign: list[float] = field(default_factory=lambda: [-1.0, 1.0, 1.0, -1.0])
    """Per-wheel sign (FL, FR, RL, RR) absorbing the URDF joint-axis direction."""

    # --- command limits ---
    max_lin_vel_x: float = 0.6
    """Max forward/backward body velocity (m/s)."""

    max_lin_vel_y: float = 0.6
    """Max lateral body velocity (m/s)."""

    max_ang_vel_z: float = 1.5
    """Max yaw rate (rad/s)."""
    
class HolonomicBaseAction(ActionTerm):
    """直接控制底盘根节点体速度（全向移动）
    简化的全向移动控制，不模拟真实麦轮动力学。
    策略输出 (v_x, v_y, omega_z)，直接设置为 base_link 的体速度。
    """

    cfg: HolonomicBaseActionCfg
    _asset: Articulation

    def __init__(self, cfg: HolonomicBaseActionCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset: Articulation = env.scene[cfg.asset_name]

        # Command scaling: policy output (~[-1, 1]) -> body velocity
        self._vel_scale = torch.tensor(
            [cfg.max_lin_vel_x, cfg.max_lin_vel_y, cfg.max_ang_vel_z],
            device=env.device
        )

        self._raw_actions = torch.zeros(env.num_envs, 3, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, 3, device=env.device)

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
        # Clip to [-1, 1] then scale to body velocities
        self._processed_actions = torch.clamp(actions, -1.0, 1.0) * self._vel_scale

    def apply_actions(self) -> None:
        """直接设置根节点体速度（全向移动）"""
        body_vel = self._processed_actions  # (num_envs, 3): vx, vy, wz

        # 获取当前姿态
        root_quat = self._asset.data.root_quat_w  # (num_envs, 4)

        # 体系线速度 (vx, vy, 0)
        lin_vel_b = torch.cat([
            body_vel[:, :2],  # vx, vy
            torch.zeros(body_vel.shape[0], 1, device=body_vel.device)  # vz=0
        ], dim=1)

        # 旋转到世界系（用 quat_apply；quat_rotate 已被 Isaac Lab 标记弃用）
        lin_vel_w = quat_apply(root_quat, lin_vel_b)

        # 角速度只有 yaw (0, 0, wz)
        ang_vel_w = torch.cat([
            torch.zeros(body_vel.shape[0], 2, device=body_vel.device),  # wx=wy=0
            body_vel[:, 2:3]  # wz
        ], dim=1)

        # 设置根节点速度: [lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]
        root_vel = torch.cat([lin_vel_w, ang_vel_w], dim=1)
        self._asset.write_root_velocity_to_sim(root_vel)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class HolonomicBaseActionCfg(ActionTermCfg):
    """Configuration for :class:`HolonomicBaseAction`."""

    class_type: type[ActionTerm] = HolonomicBaseAction

    asset_name: str = 'eggtart'
    """Name of the articulation asset in the scene."""

    # --- command limits ---
    max_lin_vel_x: float = 0.5
    """Max forward/backward body velocity (m/s)."""

    max_lin_vel_y: float = 0.5
    """Max lateral body velocity (m/s)."""

    max_ang_vel_z: float = 1.5
    """Max yaw rate (rad/s)."""


class GripperForceAction(ActionTerm):
    """夹爪力控动作项：策略输出夹持力，夹爪自动贴合物体体积

    **为什么要力控**
        位置控制下策略必须精确命令一个关节角。物体多大就该停在哪个角度是几何决定的
        （实测：20 mm 物体停 q=0.116，30 mm 停 q=0.290，40 mm 停 q=0.400），
        位置控制要么闭不到位、要么把物体挤穿。力控只给"多大的力"，最终停在哪
        由接触自然决定，换个尺寸的物体不用重训。

    **实现原理**
        ImplicitActuator 下 PhysX 的关节力矩为::

            tau = stiffness * (q_target - q) + damping * (qd_target - qd) + effort_target

        所以把该关节的 ``stiffness`` 置 0 就退化成"阻尼 + 力矩指令"，即力控。
        本项在 ``__init__`` 里直接改仿真中的增益（``write_joint_stiffness_to_sim``），
        因此 asset 配置里 gripper 那组 actuator 的 stiffness 写什么都会被覆盖。

        保留一点 ``damping``（默认 0.05）很关键：完全没有阻尼时关节在无接触段会被
        恒力矩不断加速，撞上物体时冲击过大，实测会把立方体挤穿。

    **动作语义**
        原始动作 ``a`` 先 clamp 到 [-1, 1]，再映射成力矩::

            tau = a * max_effort      (a<0 闭合，a>0 张开)

        实测合理夹持力矩是 0.3~1.0 Nm：0.3 Nm 能夹住 30 mm 立方体停在 q≈0.29，
        3.0 Nm 会把立方体挤穿一路闭到硬限位 -0.2。所以 ``max_effort`` 默认 1.0。

    **注意**
        力控下"夹爪角度低于阈值算闭合"这类判据仍然可用，但夹住物体时角度**不会**
        到硬限位（被物体挡住）。所以 ``GRIPPER_CLOSED_THRESHOLD`` 必须比
        "夹住目标物体时的稳定角度"更**大**，否则 grasp 奖励永远拿不到。
        30 mm 物体停在 q≈0.29，阈值应设在 0.29 以上（例如 0.35）。
    """

    cfg: GripperForceActionCfg

    def __init__(self, cfg: GripperForceActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]

        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        if len(self._joint_ids) == 0:
            raise ValueError(f"GripperForceAction 找不到关节: {cfg.joint_names}")

        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._processed_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)

        # 关掉位置增益 -> 纯力矩控制；留一点阻尼避免无接触段被恒力矩加速到冲击过大
        n = len(self._joint_ids)
        zeros = torch.zeros(env.num_envs, n, device=env.device)
        self._asset.write_joint_stiffness_to_sim(zeros, joint_ids=self._joint_ids)
        self._asset.write_joint_damping_to_sim(
            zeros + cfg.damping, joint_ids=self._joint_ids
        )
        print(
            f"[GripperForceAction] 关节 {self._joint_names} 切换为力控:"
            f" stiffness=0, damping={cfg.damping}, max_effort={cfg.max_effort} Nm"
        )

    @property
    def action_dim(self) -> int:
        return len(self._joint_ids)

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions = actions.clone()
        # a<0 闭合，a>0 张开
        self._processed_actions = torch.clamp(actions, -1.0, 1.0) * self.cfg.max_effort

    def apply_actions(self) -> None:
        self._asset.set_joint_effort_target(
            self._processed_actions, joint_ids=self._joint_ids
        )

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0


@configclass
class GripperForceActionCfg(ActionTermCfg):
    """:class:`GripperForceAction` 的配置"""

    class_type: type[ActionTerm] = GripperForceAction

    asset_name: str = "robot"
    """场景中 articulation 资产的名字"""

    joint_names: list[str] = field(default_factory=lambda: ["end_effector_joint"])
    """夹爪关节名（支持正则）"""

    max_effort: float = 1.0
    """动作 ±1 对应的力矩上限（Nm）

    实测参考（30 mm 立方体）：0.3 Nm 稳定停在 q≈0.29；3.0 Nm 会把物体挤穿。
    """

    damping: float = 0.05
    """力控下保留的关节阻尼

    不能为 0：无接触段恒力矩会把关节持续加速，撞上物体时冲击过大。
    """
