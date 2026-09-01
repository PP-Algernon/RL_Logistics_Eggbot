#!/usr/bin/env python3
"""
诊断grasp reward为什么一直是0

检查：
1. ee到target的实际距离
2. 夹爪的实际位置
3. grasp条件是否满足
4. curriculum权重是否已激活
"""

import torch
import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=16)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

def main():
    # Create environment
    env_cfg_entry = "Isaac-Mobile-Grasp-Eggtart-v0"
    env = gym.make(env_cfg_entry, num_envs=args_cli.num_envs)

    # Wrap for RL
    env = RslRlVecEnvWrapper(env)

    print("\n" + "="*80)
    print("GRASP REWARD 诊断")
    print("="*80)

    # Check reward manager
    print("\n当前奖励项:")
    for name, term in env.unwrapped.reward_manager._term_cfgs.items():
        weight = term.weight
        print(f"  {name:20s} weight={weight:8.3f}")

    # Check curriculum
    print("\n课程学习状态 (common_step_counter = 0):")
    if hasattr(env.unwrapped, 'curriculum_manager'):
        for name in env.unwrapped.curriculum_manager._term_cfgs.keys():
            print(f"  {name}")

    # Reset and run some steps
    obs = env.reset()

    print("\n" + "="*80)
    print("运行100步，检查关键指标...")
    print("="*80)

    grasp_distances = []
    gripper_positions = []
    grasp_rewards = []
    ee_reach_rewards = []

    for step in range(100):
        # Random action
        action = 2.0 * torch.rand(env.num_envs, env.num_actions, device=env.device) - 1.0
        obs, rewards, dones, infos = env.step(action)

        # Get diagnostics from unwrapped env
        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        target = unwrapped.scene["target"]

        # Compute EE to target distance
        from eggtart_grasp.tasks.mobile_grasp.mdp.rewards import _ee_to_target_distance
        from eggtart_grasp.assets import EGGTART_EE_GRASP_OFFSET
        from isaaclab.managers import SceneEntityCfg

        ee_cfg = SceneEntityCfg("robot", body_names="end_effector")
        target_cfg = SceneEntityCfg("target")

        dist = _ee_to_target_distance(unwrapped, ee_cfg, target_cfg, EGGTART_EE_GRASP_OFFSET)

        # Get gripper position
        gripper_joint_idx = robot.find_joints("end_effector_joint")[0]
        gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

        grasp_distances.append(dist.mean().item())
        gripper_positions.append(gripper_pos.mean().item())

        # Get rewards
        reward_log = unwrapped.reward_manager._episode_sums
        if "grasp" in reward_log:
            grasp_rewards.append(reward_log["grasp"].mean().item())
        if "ee_reach" in reward_log:
            ee_reach_rewards.append(reward_log["ee_reach"].mean().item())

    print(f"\nEE到目标距离统计 (100步):")
    print(f"  最小: {min(grasp_distances):.4f} m")
    print(f"  最大: {max(grasp_distances):.4f} m")
    print(f"  平均: {sum(grasp_distances)/len(grasp_distances):.4f} m")
    print(f"  GRASP_REACH_THRESHOLD = 0.035 m")

    print(f"\n夹爪位置统计 (100步):")
    print(f"  最小: {min(gripper_positions):.4f}")
    print(f"  最大: {max(gripper_positions):.4f}")
    print(f"  平均: {sum(gripper_positions)/len(gripper_positions):.4f}")
    print(f"  GRIPPER_CLOSED_THRESHOLD = -0.15")
    print(f"  (夹爪范围: -0.2 到 1.0)")

    if grasp_rewards:
        print(f"\nGrasp奖励统计:")
        print(f"  Episode累计: {grasp_rewards[-1]:.4f}")

    if ee_reach_rewards:
        print(f"\nEE reach奖励统计:")
        print(f"  Episode累计: {ee_reach_rewards[-1]:.4f}")

    # 读取配置文件中的值
    from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import (
        GRASP_REACH_THRESHOLD, GRIPPER_CLOSED_THRESHOLD, GRASP_DWELL_TIME
    )

    print(f"\n" + "="*80)
    print("当前配置:")
    print(f"  GRASP_REACH_THRESHOLD   = {GRASP_REACH_THRESHOLD} m")
    print(f"  GRIPPER_CLOSED_THRESHOLD = {GRIPPER_CLOSED_THRESHOLD}")
    print(f"  GRASP_DWELL_TIME        = {GRASP_DWELL_TIME} s (约 {int(GRASP_DWELL_TIME/0.0333)} 步)")
    print("="*80)

    print("\n建议:")
    if min(grasp_distances) > GRASP_REACH_THRESHOLD * 2:
        print("  ⚠️  EE从未接近目标！距离阈值的2倍以上")
        print("     → 检查ee_reach权重是否足够大")
        print("     → 检查curriculum是否已激活ee阶段")
    elif min(grasp_distances) > GRASP_REACH_THRESHOLD:
        print("  ⚠️  EE接近但未达到GRASP_REACH_THRESHOLD")
        print(f"     → 考虑放宽阈值到 {min(grasp_distances)*1.2:.4f} m")
        print("     → 或增大ee_reach权重")
    else:
        print("  ✓ EE可以到达grasp范围")

    if max(gripper_positions) < 0:
        print("  ⚠️  夹爪从未打开！")
        print("     → 检查动作空间映射")
    elif min(gripper_positions) > GRIPPER_CLOSED_THRESHOLD:
        print("  ⚠️  夹爪从未闭合到阈值")
        print(f"     → 考虑放宽阈值到 {min(gripper_positions)*0.9:.4f}")
    else:
        print("  ✓ 夹爪可以闭合")

    env.close()
    simulation_app.close()

if __name__ == "__main__":
    main()
