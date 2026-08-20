"""
摄像头训练配置 - 快速参考
=====================================

✅ 已完成的配置
--------------
1. 添加手腕摄像头传感器（84x84 RGB+Depth）
2. 新增视觉观测函数（camera_rgb, camera_depth）
3. 创建视觉训练配置（512 envs, 降低学习率）
4. 注册新的 Gym 环境

🎯 两种训练模式
--------------
【模式 1】纯状态（原版）
  环境: Isaac-Mobile-Grasp-Eggtart-v0
  优点: 快速，适合算法调试
  缺点: 不利于 sim-to-real

【模式 2】状态+视觉（新增）
  环境: Isaac-Mobile-Grasp-Eggtart-Vision-v0
  优点: 准备 sim-to-real，更鲁棒
  缺点: 慢 10 倍，需要更多内存

🚀 快速启动命令
--------------
# 测试摄像头配置
./isaaclab.sh -p source/eggtart_grasp/scripts/test_camera.py --save_images

# 训练（视觉版）
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Vision-v0 \
    --num_envs 512 --headless

# 播放（视觉版）
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Vision-Play-v0 \
    --num_envs 16 --checkpoint /path/to/model.pt

📊 关键参数对比
--------------
参数             | 纯状态版  | 视觉版
----------------|----------|--------
num_envs        | 2048     | 512
学习率          | 1e-3     | 3e-4
熵系数          | 0.005    | 0.01
max_iterations  | 1500     | 3000
观测归一化      | False    | True

⚙️ 自定义配置位置
--------------
摄像头参数:    mobile_grasp_env_cfg.py:81-102
观测配置:      mobile_grasp_env_cfg.py:134-142
训练超参:      agents/rsl_rl_ppo_vision_cfg.py
环境注册:      config/eggtart/__init__.py

📁 新增文件清单
--------------
✓ mdp/observations.py              (添加 camera_rgb/depth 函数)
✓ mobile_grasp_env_cfg.py          (添加 wrist_camera 传感器)
✓ config/eggtart/grasp_env_vision_cfg.py
✓ config/eggtart/agents/rsl_rl_ppo_vision_cfg.py
✓ scripts/test_camera.py
✓ VISION_TRAINING.md

⚠️ 重要注意事项
--------------
1. 摄像头安装在 link_005 上，确保该链节存在
2. 视觉训练需要至少 24GB GPU 内存
3. concatenate_terms = False（观测为字典格式）
4. 可能需要自定义 CNN 编码器以提升性能

🔧 下一步优化方向
--------------
□ 实现自定义 CNN Actor-Critic 网络
□ 添加域随机化（光照、纹理、噪声）
□ 尝试更高分辨率（128x128）
□ 添加第三人称摄像头
□ 实现视觉注意力机制

📚 详细文档
-----------
参见: source/eggtart_grasp/VISION_TRAINING.md
"""
