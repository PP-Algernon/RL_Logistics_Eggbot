#!/usr/bin/env python
"""麦克纳姆轮标定脚本

测试底盘的前进和旋转，输出建议的几何参数修正值。

用法:
    cd /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
    ./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_mecanum.py
"""

import argparse
import math
import torch
import gymnasium as gym

# 导入环境
import eggtart_grasp.tasks.mobile_grasp.config.eggtart  # noqa: F401


def test_forward(env, distance=1.0, velocity=0.3):
    """测试前进指定距离，计算实际移动距离"""
    print(f"\n{'='*60}")
    print(f"测试 1: 前进 {distance:.2f} 米")
    print(f"{'='*60}")

    obs, _ = env.reset()
    robot = env.scene["robot"]

    # 记录起始位置
    start_pos = robot.data.root_pos_w[0, :2].clone()

    # 计算需要的步数
    dt = env.step_dt  # 仿真时间步长
    time_needed = distance / velocity
    steps = int(time_needed / dt)

    print(f"发送前进命令: vx = {velocity:.2f} m/s")
    print(f"持续时间: {time_needed:.2f} s ({steps} steps)")

    # 发送前进命令
    action_dim = env.action_space.shape[0]
    for i in range(steps):
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        # 前进命令在第一个维度（base_velocity 的 vx）
        action[:, 0] = velocity / 0.6  # 归一化到 [-1, 1]，max_lin_vel_x = 0.6
        env.step(action)

        if (i + 1) % 50 == 0:
            current_pos = robot.data.root_pos_w[0, :2]
            traveled = torch.norm(current_pos - start_pos).item()
            print(f"  Step {i+1}/{steps}: 已前进 {traveled:.3f} m")

    # 计算最终位置
    end_pos = robot.data.root_pos_w[0, :2]
    actual_dist = torch.norm(end_pos - start_pos).item()
    ratio = actual_dist / distance

    print(f"\n{'─'*60}")
    print(f"✓ 命令前进距离: {distance:.3f} m")
    print(f"✓ 实际前进距离: {actual_dist:.3f} m")
    print(f"✓ 比例: {ratio:.3f}")

    # 当前参数
    current_r = 0.043
    suggested_r = current_r * ratio

    print(f"\n建议修改 wheel_radius:")
    print(f"  当前值: {current_r:.4f} m")
    print(f"  建议值: {suggested_r:.4f} m")

    return ratio


def test_rotation(env, angle_deg=90, angular_velocity=0.5):
    """测试旋转指定角度，计算实际旋转角度"""
    print(f"\n{'='*60}")
    print(f"测试 2: 旋转 {angle_deg:.1f}°")
    print(f"{'='*60}")

    obs, _ = env.reset()
    robot = env.scene["robot"]

    # 记录起始姿态
    start_quat = robot.data.root_quat_w[0].clone()

    # 计算需要的步数
    dt = env.step_dt
    target_rad = math.radians(angle_deg)
    time_needed = target_rad / angular_velocity
    steps = int(time_needed / dt)

    print(f"发送旋转命令: wz = {angular_velocity:.2f} rad/s")
    print(f"持续时间: {time_needed:.2f} s ({steps} steps)")

    # 发送旋转命令
    action_dim = env.action_space.shape[0]
    for i in range(steps):
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        # 旋转命令在第三个维度（base_velocity 的 wz）
        action[:, 2] = angular_velocity / 1.5  # 归一化，max_ang_vel_z = 1.5
        env.step(action)

        if (i + 1) % 50 == 0:
            print(f"  Step {i+1}/{steps}")

    # 计算实际旋转角度
    end_quat = robot.data.root_quat_w[0]

    # 简单方法：比较 yaw 角
    def quat_to_yaw(quat):
        # quat = [w, x, y, z]
        w, x, y, z = quat[0], quat[1], quat[2], quat[3]
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        return yaw

    start_yaw = quat_to_yaw(start_quat)
    end_yaw = quat_to_yaw(end_quat)

    actual_rad = end_yaw - start_yaw
    # 规范化到 [-pi, pi]
    while actual_rad > math.pi:
        actual_rad -= 2 * math.pi
    while actual_rad < -math.pi:
        actual_rad += 2 * math.pi

    actual_deg = math.degrees(abs(actual_rad))
    ratio = actual_deg / angle_deg

    print(f"\n{'─'*60}")
    print(f"✓ 命令旋转角度: {angle_deg:.1f}°")
    print(f"✓ 实际旋转角度: {actual_deg:.1f}°")
    print(f"✓ 比例: {ratio:.3f}")

    # 当前参数
    current_lx = 0.08
    current_ly = 0.097
    current_k = current_lx + current_ly
    suggested_k = current_k * ratio

    print(f"\n建议修改 (half_wheelbase + half_track):")
    print(f"  当前值: lx={current_lx:.4f}, ly={current_ly:.4f}, 和={current_k:.4f} m")
    print(f"  建议和: {suggested_k:.4f} m")
    print(f"  建议按比例分配: lx={current_lx * ratio:.4f}, ly={current_ly * ratio:.4f} m")

    return ratio


def test_lateral(env, distance=0.5, velocity=0.2):
    """测试横向移动（仅供参考，URDF 无滚子几何时不准确）"""
    print(f"\n{'='*60}")
    print(f"测试 3: 横向移动 {distance:.2f} 米 (参考)")
    print(f"{'='*60}")
    print("⚠️  注意: URDF 轮子无滚子几何，此测试仅供参考")

    obs, _ = env.reset()
    robot = env.scene["robot"]

    start_pos = robot.data.root_pos_w[0, :2].clone()

    dt = env.step_dt
    time_needed = distance / velocity
    steps = int(time_needed / dt)

    print(f"发送横向命令: vy = {velocity:.2f} m/s")

    action_dim = env.action_space.shape[0]
    for i in range(steps):
        action = torch.zeros(env.num_envs, action_dim, device=env.device)
        action[:, 1] = velocity / 0.6  # vy 在第二个维度
        env.step(action)

    end_pos = robot.data.root_pos_w[0, :2]
    actual_dist = torch.norm(end_pos - start_pos).item()

    print(f"\n✓ 命令横移距离: {distance:.3f} m")
    print(f"✓ 实际移动距离: {actual_dist:.3f} m")
    print(f"✓ 比例: {actual_dist / distance:.3f}")


def main():
    parser = argparse.ArgumentParser(description="麦克纳姆轮几何参数标定")
    parser.add_argument("--task", type=str, default="Isaac-Mobile-Grasp-Eggtart-Play-v0")
    parser.add_argument("--forward_dist", type=float, default=1.0, help="前进测试距离 (m)")
    parser.add_argument("--rotation_angle", type=float, default=90.0, help="旋转测试角度 (度)")
    parser.add_argument("--lateral_dist", type=float, default=0.5, help="横移测试距离 (m)")
    args = parser.parse_args()

    print("\n" + "="*60)
    print("麦克纳姆轮几何标定工具")
    print("="*60)
    print(f"环境: {args.task}")
    print("="*60)

    # 创建环境（单个环境用于测试）
    env = gym.make(args.task, num_envs=1)

    # 运行测试
    forward_ratio = test_forward(env, distance=args.forward_dist)
    rotation_ratio = test_rotation(env, angle_deg=args.rotation_angle)
    test_lateral(env, distance=args.lateral_dist)

    # 总结
    print("\n" + "="*60)
    print("标定总结")
    print("="*60)

    print("\n📝 修改建议:")
    print(f"\n在文件: eggtart_grasp/tasks/mobile_grasp/mdp/actions.py")
    print(f"找到 MecanumBaseActionCfg 类，修改以下参数:\n")

    current_r = 0.043
    current_lx = 0.08
    current_ly = 0.097

    suggested_r = current_r * forward_ratio
    suggested_lx = current_lx * rotation_ratio
    suggested_ly = current_ly * rotation_ratio

    print(f"    wheel_radius: float = {suggested_r:.4f}  # 原值: {current_r:.4f}")
    print(f"    half_wheelbase: float = {suggested_lx:.4f}  # 原值: {current_lx:.4f}")
    print(f"    half_track: float = {suggested_ly:.4f}  # 原值: {current_ly:.4f}")

    print("\n🔍 如果修正后仍有偏差，可能需要:")
    print("  1. 检查 wheel_spin_sign 是否正确")
    print("  2. 测量更精确的实物尺寸")
    print("  3. 考虑轮子打滑、摩擦等物理因素")

    print("\n" + "="*60)
    print("标定完成！")
    print("="*60)

    env.close()


if __name__ == "__main__":
    main()
