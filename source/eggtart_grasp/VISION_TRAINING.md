# 视觉训练配置说明

本文档说明如何使用摄像头进行 Eggtart 移动抓取任务的训练。

## 📷 已添加的功能

### 1. **摄像头传感器**
- **位置**: 安装在机械臂末端 `link_005/WristCamera`
- **分辨率**: 84x84 (可调整)
- **数据类型**: RGB + Depth
- **更新频率**: 10Hz
- **视角**: 向前看，略微朝下

### 2. **视觉观测**
新增两个观测项：
- `camera_rgb`: RGB 图像 (N, 84, 84, 3)
- `camera_depth`: 深度图像 (N, 84, 84, 1)

### 3. **训练配置**
- **原版**: `Isaac-Mobile-Grasp-Eggtart-v0` (纯状态，2048 envs)
- **视觉版**: `Isaac-Mobile-Grasp-Eggtart-Vision-v0` (状态+视觉，512 envs)

---

## 🚀 如何训练

### **方式 1: 纯状态训练（原版，快速）**
```bash
cd /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
conda activate my_isaac_env
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-v0 \
    --num_envs 2048
```

### **方式 2: 视觉训练（新增，慢但 sim-to-real）**
```bash
cd /home/pu/RL-ws/ProjectLearning/Eggtart-logistics-robot
conda activate my_isaac_env
./isaaclab.sh -p source/standalone/workflows/rsl_rl/train.py \
    --task Isaac-Mobile-Grasp-Eggtart-Vision-v0 \
    --num_envs 512 \
    --headless
```

### **播放训练好的模型**
```bash
# 纯状态版本
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 32 \
    --checkpoint /path/to/model.pt

# 视觉版本
./isaaclab.sh -p source/standalone/workflows/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Vision-Play-v0 \
    --num_envs 16 \
    --checkpoint /path/to/vision_model.pt
```

---

## ⚙️ 配置调整

### **修改摄像头分辨率**
编辑 [mobile_grasp_env_cfg.py:81-102](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L81-L102)：
```python
wrist_camera = CameraCfg(
    height=128,  # 从 84 改为 128
    width=128,
    ...
)
```

### **修改摄像头位置/朝向**
同样位置，修改 `offset`:
```python
offset=CameraCfg.OffsetCfg(
    pos=(0.1, 0.0, 0.03),  # 调整相对于 link_005 的位置
    rot=(0.707, 0.0, 0.0, 0.707),  # 四元数旋转
    convention="ros",
)
```

### **只使用 RGB 或只使用 Depth**
编辑 [mobile_grasp_env_cfg.py:134-142](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L134-L142)，注释掉不需要的观测：
```python
# 只保留 RGB
camera_rgb = ObsTerm(...)
# camera_depth = ObsTerm(...)  # 注释掉
```

---

## 📊 性能对比

| 配置 | 并行环境数 | FPS (估算) | 收敛时间 | 内存占用 |
|------|-----------|-----------|---------|---------|
| 纯状态 | 2048 | ~30,000 | 1-2 小时 | ~8GB |
| 视觉 | 512 | ~3,000 | 8-12 小时 | ~24GB |

---

## ⚠️ 注意事项

1. **训练速度**: 视觉训练比纯状态慢约 10 倍
2. **内存需求**: 至少需要 24GB GPU 内存（512 envs）
3. **网络架构**: 目前使用默认 RSL-RL 网络，可能需要自定义 CNN 编码器
4. **观测空间**: 设置了 `concatenate_terms = False`，需要修改 Actor-Critic 网络以处理字典观测

---

## 🔧 下一步优化

### **1. 自定义 CNN 编码器**
当前使用 RSL-RL 默认网络，建议实现自定义编码器：
```python
class VisionActorCritic(ActorCritic):
    def __init__(self, ...):
        # CNN for RGB
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2),
            nn.ReLU(),
            nn.Flatten(),
        )
        # Concat with proprioception
        # MLP head
```

### **2. 域随机化**
增强 sim-to-real 鲁棒性：
- 光照随机化
- 纹理随机化
- 摄像头噪声

### **3. 多摄像头**
添加第三人称摄像头：
```python
third_person_camera = CameraCfg(
    prim_path="{ENV_REGEX_NS}/ThirdPersonCamera",
    offset=CameraCfg.OffsetCfg(pos=(-2.0, 0.0, 1.5)),
    ...
)
```

---

## 📁 文件结构

```
eggtart_grasp/tasks/mobile_grasp/
├── mdp/
│   └── observations.py          # 新增 camera_rgb/camera_depth
├── mobile_grasp_env_cfg.py      # 新增 wrist_camera 传感器
├── config/eggtart/
│   ├── grasp_env_vision_cfg.py  # 新增视觉环境配置
│   ├── agents/
│   │   └── rsl_rl_ppo_vision_cfg.py  # 新增视觉训练配置
│   └── __init__.py              # 注册新环境
```

---

## 🐛 常见问题

### Q1: 报错 "Camera sensor not found"
**A**: 确保机器人 URDF/USD 中存在 `link_005` 链节。检查：
```bash
./isaaclab.sh -p -m usd_viewer /path/to/robot.usd
```

### Q2: 内存溢出
**A**: 减少并行环境数：
```python
self.scene.num_envs = 256  # 从 512 减少
```

### Q3: 图像全黑
**A**: 检查光照和材质：
- 增加 `DomeLightCfg` 的 `intensity`
- 确保目标物体有 `visual_material`

---

## 📚 参考资料

- IsaacLab Camera 文档: https://isaac-sim.github.io/IsaacLab/
- RSL-RL 官方仓库: https://github.com/leggedrobotics/rsl_rl
- 视觉 RL 论文: "Learning Dexterous In-Hand Manipulation" (OpenAI, 2018)
