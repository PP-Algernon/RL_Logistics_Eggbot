"""演示用的脚本化抓取接管（scripted grasp fallback）

用途
----
训练中期策略往往已经学会"底盘靠近 + 机械臂伸到目标"，但迟迟学不会"闭爪"，
于是 ``Episode_Reward/grasp`` 一直是 0，演示时看起来就是手停在目标旁边不动。

本模块提供 :class:`ScriptedGraspOverride`：在 play/演示脚本里包一层，
当**目标已经进入可夹范围、而策略在给定时间内仍然没有闭爪**时，直接由脚本
接管动作，执行固定的 闭爪 -> 保持 -> 收回 序列。

这只是**演示/调试**工具，不参与训练（不写奖励、不改环境），
用来验证"如果会夹，整条流程能不能跑通"，以及录演示视频。

动作布局（与 ``mobile_grasp_env_cfg.ActionsCfg`` 一致）
------------------------------------------------------
    index 0:2  -- base_velocity  (vx, vy, wz)，HolonomicBaseAction
    index 3:7  -- arm_action     (5 个臂关节，JointPositionAction, scale=0.5, use_default_offset)
    index 8    -- gripper_action (1 个夹爪关节, scale=1.0, use_default_offset)

因为两个关节动作项都是 ``use_default_offset=True``：

    target_q = action * scale + default_q

所以 **arm action = 0 就等于回到 nominal 姿态**，收回动作只需把臂的动作拉到 0；
夹爪要闭到 q_closed，需要发 ``action = (q_closed - default_q) / scale``。

状态机
------
    IDLE    -- 不干预，纯看策略。目标在范围内且策略没闭爪则累计计时
    CLOSING -- 接管：命令闭爪，冻结臂和底盘
    HOLDING -- 保持闭合
    RETRACT -- 臂动作从冻结值线性插值到 0（回 nominal），夹爪保持闭合
    DONE    -- 保持，直到 episode 重置

只要策略**自己**闭了爪，计时器就清零、永不接管——脚本不会抢已经学会的行为。
"""

from __future__ import annotations

import torch

from isaaclab.managers import SceneEntityCfg

# 状态编码
IDLE, CLOSING, HOLDING, RETRACT, DONE = 0, 1, 2, 3, 4
_STATE_NAMES = {IDLE: "IDLE", CLOSING: "CLOSING", HOLDING: "HOLDING", RETRACT: "RETRACT", DONE: "DONE"}


class ScriptedGraspOverride:
    """当策略"该夹却不夹"时，接管动作执行固定的抓取+收回序列

    Args:
        env: 已 unwrap 的 ManagerBasedRLEnv
        reach_threshold: 抓取点到目标的距离小于它算"可夹范围"（米）
        gripper_closed_threshold: 夹爪关节低于它算"已闭合"（弧度）
        patience_time: 目标在范围内、策略仍不闭爪，等这么久后接管（秒）
        close_time: 闭爪动作持续时间（秒）
        hold_time: 闭合后保持时间（秒）
        retract_time: 收回（臂回 nominal）的时间（秒）
        grasp_offset: 抓取点相对 end_effector body 原点的偏移，须与奖励项一致
        verbose: 打印状态切换
    """

    def __init__(
        self,
        env,
        reach_threshold: float,
        gripper_closed_threshold: float,
        patience_time: float = 1.0,
        close_time: float = 0.3,
        hold_time: float = 0.4,
        retract_time: float = 1.0,
        grasp_offset: tuple[float, float, float] | None = None,
        verbose: bool = True,
    ):
        self.env = env
        self.reach_threshold = reach_threshold
        self.gripper_closed_threshold = gripper_closed_threshold
        self.grasp_offset = grasp_offset
        self.verbose = verbose

        self.device = env.device
        self.num_envs = env.num_envs
        dt = env.step_dt

        # 秒 -> env step
        self._patience_steps = max(1, int(round(patience_time / dt)))
        self._close_steps = max(1, int(round(close_time / dt)))
        self._hold_steps = max(1, int(round(hold_time / dt)))
        self._retract_steps = max(1, int(round(retract_time / dt)))

        self._resolve_indices()
        self._resolve_gripper_command()

        # per-env 状态
        self._state = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._near_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._phase_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # 接管瞬间冻结的臂动作，收回时从这里插值回 0
        self._frozen_arm = torch.zeros(self.num_envs, self._n_arm, device=self.device)

        if self.verbose:
            print(
                f"[ScriptedGraspOverride] reach<{reach_threshold} m 持续 {self._patience_steps} 步"
                f" 未闭爪则接管；闭爪 {self._close_steps} 步 / 保持 {self._hold_steps} 步"
                f" / 收回 {self._retract_steps} 步\n"
                f"[ScriptedGraspOverride] 动作切片 base={self._base_slice} arm={self._arm_slice}"
                f" gripper={self._grip_slice}，闭爪指令={self._close_cmd:.4f}"
            )

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _resolve_indices(self) -> None:
        """从 ActionManager 求出各动作项在拼接动作向量里的切片"""
        am = self.env.action_manager
        offset = 0
        spans: dict[str, slice] = {}
        for name, dim in zip(am.active_terms, am.action_term_dim):
            spans[name] = slice(offset, offset + dim)
            offset += dim
        self._total_dim = offset

        missing = [n for n in ("base_velocity", "arm_action", "gripper_action") if n not in spans]
        if missing:
            raise RuntimeError(
                f"ScriptedGraspOverride 需要动作项 {missing}，实际只有 {list(spans)}。"
                " 动作配置改过的话要同步改这里。"
            )
        self._base_slice = spans["base_velocity"]
        self._arm_slice = spans["arm_action"]
        self._grip_slice = spans["gripper_action"]
        self._n_arm = self._arm_slice.stop - self._arm_slice.start

        # 抓取点 / 夹爪关节 的实体配置
        self._ee_cfg = SceneEntityCfg("robot", body_names="end_effector")
        self._ee_cfg.resolve(self.env.scene)
        self._grip_joint_cfg = SceneEntityCfg("robot", joint_names=["end_effector_joint"])
        self._grip_joint_cfg.resolve(self.env.scene)

    def _resolve_gripper_command(self) -> None:
        """算出"闭爪"对应的动作值（位置控制和力控两种都支持）

        力控（:class:`GripperForceAction`）：动作直接是力矩系数，闭合方向为负，
        发 -1.0 就是最大夹持力。

        位置控制（``JointPositionAction``）：``target_q = action * scale + offset``，
        ``use_default_offset=True`` 时 offset 是 default_joint_pos，
        所以 ``action = (q_target - default_q) / scale``。
        """
        robot = self.env.scene["robot"]
        term = self.env.action_manager.get_term("gripper_action")
        jid = self._grip_joint_cfg.joint_ids[0]

        # 力控项没有 _scale/_offset，用它来区分两种模式
        if not hasattr(term, "_scale"):
            self._force_mode = True
            self._close_cmd = -1.0  # 满力闭合
            self._q_closed_target = None
            return

        self._force_mode = False
        q_lower = robot.data.joint_pos_limits[0, jid, 0].item()
        q_target = min(q_lower, self.gripper_closed_threshold - 0.02)

        scale = term._scale
        offset = term._offset
        scale_v = float(scale) if isinstance(scale, float) else float(scale.reshape(-1)[0])
        offset_v = float(offset) if isinstance(offset, float) else float(offset.reshape(-1)[0])

        self._close_cmd = float((q_target - offset_v) / scale_v)
        self._q_closed_target = q_target

    # ------------------------------------------------------------------
    # 状态查询
    # ------------------------------------------------------------------
    def _grasp_point_w(self) -> torch.Tensor:
        """抓取点世界坐标（与 rewards._grasp_point_w 同一套算法）"""
        from isaaclab.utils.math import quat_apply

        robot = self.env.scene["robot"]
        bid = self._ee_cfg.body_ids[0]
        pos = robot.data.body_pos_w[:, bid]
        if self.grasp_offset is None:
            return pos
        quat = robot.data.body_quat_w[:, bid]
        off = torch.tensor(self.grasp_offset, device=pos.device, dtype=pos.dtype)
        return pos + quat_apply(quat, off.unsqueeze(0).expand(pos.shape[0], -1))

    def _is_near(self) -> torch.Tensor:
        target = self.env.scene["target"]
        dist = torch.norm(self._grasp_point_w() - target.data.root_pos_w, dim=1)
        return dist < self.reach_threshold

    def _is_closed(self) -> torch.Tensor:
        robot = self.env.scene["robot"]
        q = robot.data.joint_pos[:, self._grip_joint_cfg.joint_ids[0]]
        return q < self.gripper_closed_threshold

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """episode 重置时清状态。env_ids=None 表示全部"""
        if env_ids is None:
            self._state.zero_()
            self._near_steps.zero_()
            self._phase_steps.zero_()
            self._frozen_arm.zero_()
        else:
            self._state[env_ids] = IDLE
            self._near_steps[env_ids] = 0
            self._phase_steps[env_ids] = 0
            self._frozen_arm[env_ids] = 0.0

    def __call__(self, actions: torch.Tensor) -> torch.Tensor:
        """改写策略动作；返回同 shape 的新动作张量（不原地改传入的）"""
        actions = actions.clone()
        is_near = self._is_near()
        is_closed = self._is_closed()

        # ---- IDLE：只计时，不干预 ----
        idle = self._state == IDLE
        # 策略自己闭上了 -> 计时清零，让策略继续主导
        self._near_steps = torch.where(
            idle & is_near & ~is_closed,
            self._near_steps + 1,
            torch.zeros_like(self._near_steps),
        )
        # 忍耐到头 -> 接管
        trigger = idle & (self._near_steps >= self._patience_steps)
        if trigger.any():
            self._frozen_arm[trigger] = actions[trigger][:, self._arm_slice]
            self._state[trigger] = CLOSING
            self._phase_steps[trigger] = 0
            if self.verbose:
                for i in torch.nonzero(trigger, as_tuple=False).flatten().tolist():
                    print(f"[ScriptedGraspOverride] env {i}: 接管 -> CLOSING")

        # ---- 已接管的 env：按阶段写动作 ----
        active = self._state != IDLE
        if active.any():
            self._phase_steps[active] += 1
            self._apply_scripted(actions)
            self._advance_states()

        return actions

    # ------------------------------------------------------------------
    # 内部：写动作 / 推进状态
    # ------------------------------------------------------------------
    def _apply_scripted(self, actions: torch.Tensor) -> None:
        """对所有非 IDLE 的 env 覆写动作（原地改 actions）"""
        st = self._state

        # 接管期间底盘一律停住：目标已在手边，再动只会把目标撞飞
        moving = st != IDLE
        actions[moving, self._base_slice] = 0.0

        # 夹爪：CLOSING 之后一直保持闭合指令
        actions[moving, self._grip_slice] = self._close_cmd

        # 臂：CLOSING/HOLDING 冻结在接管瞬间的姿态；RETRACT 线性插值回 0
        freeze = moving & ((st == CLOSING) | (st == HOLDING))
        if freeze.any():
            actions[freeze, self._arm_slice] = self._frozen_arm[freeze]

        retract = st == RETRACT
        if retract.any():
            # alpha: 0 -> 1 随阶段推进
            alpha = (self._phase_steps[retract].float() / self._retract_steps).clamp(0.0, 1.0)
            actions[retract, self._arm_slice] = self._frozen_arm[retract] * (1.0 - alpha).unsqueeze(1)

        done = st == DONE
        if done.any():
            # 收回完成：臂保持 nominal（action=0），夹爪保持闭合
            actions[done, self._arm_slice] = 0.0

    def _advance_states(self) -> None:
        """到时长就切下一个阶段"""
        st = self._state
        ph = self._phase_steps

        to_hold = (st == CLOSING) & (ph >= self._close_steps)
        to_retract = (st == HOLDING) & (ph >= self._hold_steps)
        to_done = (st == RETRACT) & (ph >= self._retract_steps)

        for mask, nxt in ((to_hold, HOLDING), (to_retract, RETRACT), (to_done, DONE)):
            if mask.any():
                self._state[mask] = nxt
                self._phase_steps[mask] = 0
                if self.verbose:
                    ids = torch.nonzero(mask, as_tuple=False).flatten().tolist()
                    print(f"[ScriptedGraspOverride] env {ids} -> {_STATE_NAMES[nxt]}")

    # ------------------------------------------------------------------
    # 观测用
    # ------------------------------------------------------------------
    @property
    def state_names(self) -> list[str]:
        return [_STATE_NAMES[s] for s in self._state.tolist()]

    def stats(self) -> dict[str, int]:
        """各状态的 env 数量，便于打印"""
        return {name: int((self._state == code).sum().item()) for code, name in _STATE_NAMES.items()}
