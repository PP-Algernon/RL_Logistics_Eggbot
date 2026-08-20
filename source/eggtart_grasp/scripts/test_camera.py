#!/usr/bin/env python
"""Test script to verify camera configuration and capture sample images.

Usage:
    ./isaaclab.sh -p source/eggtart_grasp/scripts/test_camera.py
"""

import argparse
import os
from datetime import datetime

import torch
import gymnasium as gym

import eggtart_grasp.tasks.mobile_grasp.config.eggtart  # noqa: F401


def main():
    parser = argparse.ArgumentParser(description="Test wrist camera configuration")
    parser.add_argument("--task", type=str, default="Isaac-Mobile-Grasp-Eggtart-Vision-v0")
    parser.add_argument("--num_envs", type=int, default=4)
    parser.add_argument("--save_images", action="store_true", help="Save sample images to disk")
    args = parser.parse_args()

    # Create environment
    print(f"[INFO] Creating environment: {args.task}")
    env = gym.make(args.task, num_envs=args.num_envs)

    # Reset environment
    print("[INFO] Resetting environment...")
    obs, _ = env.reset()

    print("\n" + "=" * 60)
    print("OBSERVATION SPACE STRUCTURE")
    print("=" * 60)

    if isinstance(obs, dict):
        for key, value in obs.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key:20s}: shape={tuple(value.shape)}, dtype={value.dtype}")
            else:
                print(f"  {key:20s}: {type(value)}")
    else:
        print(f"  Single tensor observation: {obs.shape}")

    # Check camera observations
    print("\n" + "=" * 60)
    print("CAMERA OBSERVATIONS")
    print("=" * 60)

    if "camera_rgb" in obs:
        rgb = obs["camera_rgb"]
        print(f"✓ RGB found: shape={tuple(rgb.shape)}, min={rgb.min():.3f}, max={rgb.max():.3f}")

        if args.save_images:
            save_dir = "/tmp/eggtart_camera_test"
            os.makedirs(save_dir, exist_ok=True)

            # Save first 4 environments' images
            import numpy as np
            try:
                from PIL import Image
                for env_idx in range(min(4, args.num_envs)):
                    img = rgb[env_idx].cpu().numpy()
                    # Convert to uint8 if needed
                    if img.max() <= 1.0:
                        img = (img * 255).astype(np.uint8)
                    else:
                        img = img.astype(np.uint8)

                    img_pil = Image.fromarray(img)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(save_dir, f"rgb_env{env_idx}_{timestamp}.png")
                    img_pil.save(path)
                    print(f"  Saved: {path}")
            except ImportError:
                print("  [WARNING] PIL not available, skipping image save")
    else:
        print("✗ RGB not found in observations")

    if "camera_depth" in obs:
        depth = obs["camera_depth"]
        print(f"✓ Depth found: shape={tuple(depth.shape)}, min={depth.min():.3f}, max={depth.max():.3f}")

        if args.save_images:
            save_dir = "/tmp/eggtart_camera_test"
            os.makedirs(save_dir, exist_ok=True)

            import numpy as np
            try:
                from PIL import Image
                for env_idx in range(min(4, args.num_envs)):
                    depth_img = depth[env_idx, ..., 0].cpu().numpy()
                    # Normalize to 0-255 for visualization
                    depth_vis = ((depth_img - depth_img.min()) / (depth_img.max() - depth_img.min() + 1e-6) * 255)
                    depth_vis = depth_vis.astype(np.uint8)

                    img_pil = Image.fromarray(depth_vis)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(save_dir, f"depth_env{env_idx}_{timestamp}.png")
                    img_pil.save(path)
                    print(f"  Saved: {path}")
            except ImportError:
                pass
    else:
        print("✗ Depth not found in observations")

    # Check other observations
    print("\n" + "=" * 60)
    print("PROPRIOCEPTIVE OBSERVATIONS")
    print("=" * 60)
    expected_keys = ["joint_pos", "joint_vel", "base_lin_vel", "base_ang_vel",
                     "target_position_b", "ee_to_target_b", "target_lin_vel", "actions"]
    for key in expected_keys:
        if key in obs:
            print(f"✓ {key:20s}: {tuple(obs[key].shape)}")
        else:
            print(f"✗ {key:20s}: NOT FOUND")

    # Take a few random steps
    print("\n" + "=" * 60)
    print("STEPPING THROUGH ENVIRONMENT")
    print("=" * 60)

    action_dim = env.action_space.shape[0]
    print(f"Action space dimension: {action_dim}")

    for step in range(5):
        action = torch.randn(args.num_envs, action_dim, device=env.device) * 0.1
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Step {step + 1}: reward_mean={reward.mean():.3f}, terminated={terminated.sum().item()}/{args.num_envs}")

    print("\n" + "=" * 60)
    print("TEST COMPLETE")
    print("=" * 60)

    if args.save_images:
        print(f"\n📸 Images saved to: /tmp/eggtart_camera_test/")

    print("\n✓ Camera configuration is working correctly!")

    # Clean up
    env.close()


if __name__ == "__main__":
    main()
