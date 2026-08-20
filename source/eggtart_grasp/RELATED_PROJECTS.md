# 相关开源项目参考清单

根据你的 Eggtart 移动抓取任务（麦克纳姆轮底盘 + 机械臂 + 视觉输入 + 强化学习），我整理了以下开源项目供你学习借鉴。

---

## 🎯 **最相关的项目（强烈推荐）**

### 1. **TIERS/isaac-marl-mobile-manipulation** ⭐⭐⭐⭐⭐
- **链接**: https://github.com/TIERS/isaac-marl-mobile-manipulation
- **描述**: 使用 NVIDIA Isaac Sim 的多智能体强化学习移动抓取
- **相关性**: 
  - ✅ Isaac Sim 平台（与你相同）
  - ✅ 移动底盘 + 机械臂
  - ✅ 多智能体 RL（分离底盘和机械臂策略）
- **可学习点**:
  - 底盘和机械臂的解耦控制
  - 奖励函数设计
  - 多智能体协调策略

### 2. **YinpeiDai/rlmmbp** ⭐⭐⭐⭐⭐
- **链接**: https://github.com/YinpeiDai/rlmmbp
- **描述**: 通过强化学习学习移动抓取行为
- **相关性**:
  - ✅ 移动抓取（mobile manipulation）
  - ✅ 强化学习
  - ✅ 底盘定位 + 抓取决策
- **可学习点**:
  - 底盘导航与抓取的协同策略
  - 分层强化学习方法
  - 任务分解（移动 → 抓取）

### 3. **debi-ml/Summit_ws** ⭐⭐⭐⭐
- **链接**: https://github.com/debi-ml/Summit_ws
- **描述**: Summit XL 全向移动平台 + Franka Panda 机械臂 in Isaac Sim
- **相关性**:
  - ✅ Isaac Sim 环境
  - ✅ 全向移动底盘（与麦克纳姆轮类似）
  - ✅ Franka Panda 机械臂
- **可学习点**:
  - 全向底盘的运动学建模
  - 机械臂与底盘的坐标变换
  - Isaac Sim 环境配置

### 4. **qaz9517532846/zm_robot** ⭐⭐⭐⭐
- **链接**: https://github.com/qaz9517532846/zm_robot
- **描述**: 四麦克纳姆轮驱动的 AGV in Isaac Sim
- **相关性**:
  - ✅ Isaac Sim 平台
  - ✅ **麦克纳姆轮驱动**（与你的底盘完全一致）
  - ✅ 集成 2D-Lidar + RGB-D 相机 + IMU
- **可学习点**:
  - 麦克纳姆轮的物理建模
  - 传感器集成（相机、激光雷达）
  - 底盘控制算法

---

## 📷 **视觉强化学习相关**

### 5. **NathanWu7/isaacLab.manipulation** ⭐⭐⭐⭐
- **链接**: https://github.com/NathanWu7/isaacLab.manipulation
- **描述**: 基于 IsaacLab 的机械臂和灵巧手操作扩展
- **相关性**:
  - ✅ IsaacLab 框架（你正在使用的）
  - ✅ 机械臂操作任务
  - ✅ 提供模板和示例
- **可学习点**:
  - IsaacLab 的任务配置模板
  - 观测空间和动作空间设计
  - 训练脚本组织结构

### 6. **Ericonaldo/visual_wholebody** ⭐⭐⭐⭐
- **链接**: https://github.com/Ericonaldo/visual_wholebody
- **描述**: 基于视觉的全身控制（四足机器人 + 机械臂）
- **相关性**:
  - ✅ 视觉输入 + 强化学习
  - ✅ 移动平台 + 操作臂协同
  - ✅ 全身控制（whole-body control）
- **可学习点**:
  - **视觉编码器架构**（CNN）
  - 视觉特征与本体感知的融合
  - 分层控制策略

### 7. **yubohann/RoboCupVisionRL_IsaacLab_ROS2** ⭐⭐⭐⭐
- **链接**: https://github.com/yubohan79-glitch/RoboCupVisionRL_IsaacLab_ROS2
- **描述**: 物体中心的世界模型 + Flow RL，IsaacLab + ROS2
- **相关性**:
  - ✅ IsaacLab 平台
  - ✅ 视觉导航
  - ✅ 多智能体对抗
- **可学习点**:
  - IsaacLab 与 ROS2 集成
  - 视觉世界模型
  - 物体检测与策略结合

### 8. **IsaacLab 官方 Cartpole Camera 示例** ⭐⭐⭐⭐⭐
- **链接**: https://github.com/isaac-sim/IsaacLab/blob/main/source/isaaclab_tasks/isaaclab_tasks/manager_based/classic/cartpole/cartpole_camera_env_cfg.py
- **描述**: IsaacLab 官方的视觉输入示例
- **相关性**:
  - ✅ IsaacLab 框架
  - ✅ **Camera 传感器配置**
  - ✅ 视觉观测处理
- **可学习点**:
  - **标准的摄像头配置方式**（官方最佳实践）
  - 观测空间的字典格式处理
  - `concatenate_terms=False` 的使用

---

## 🤖 **Sim-to-Real 相关**

### 9. **ByteDance-Seed/manip-as-in-sim-suite** ⭐⭐⭐⭐⭐
- **链接**: https://github.com/ByteDance-Seed/manip-as-in-sim-suite
- **描述**: ManipAsInSim 项目的 sim-to-real 代码
- **相关性**:
  - ✅ Sim-to-real 转移
  - ✅ 深度图清洗（CDM）
  - ✅ 仿真策略直接迁移到真实机器人
- **可学习点**:
  - **域随机化策略**
  - 深度图噪声处理
  - Sim-to-real 迁移技巧

### 10. **THU-VCLab/Part-Guided-3D-RL-for-Sim2Real** ⭐⭐⭐⭐
- **链接**: https://github.com/THU-VCLab/Part-Guided-3D-RL-for-Sim2Real-Articulated-Object-Manipulation
- **描述**: 基于部件引导的 3D RL sim-to-real 物体操作
- **相关性**:
  - ✅ Sim-to-real 转移
  - ✅ 3D 视觉 + 强化学习
  - ✅ 域随机化 + 背景随机化
- **可学习点**:
  - 域随机化实现细节
  - 真实数据测试方法
  - 3D 点云处理

### 11. **allenai/MolmoBot** ⭐⭐⭐⭐
- **链接**: https://github.com/allenai/MolmoBot
- **描述**: 大规模仿真实现零样本操作（zero-shot manipulation）
- **相关性**:
  - ✅ 大规模仿真训练
  - ✅ 零样本 sim-to-real 转移
  - ✅ 多场景泛化
- **可学习点**:
  - 大规模仿真策略
  - 场景多样性设计
  - 零样本泛化技巧

---

## 🧠 **算法与框架**

### 12. **leggedrobotics/rsl_rl** ⭐⭐⭐⭐⭐
- **链接**: https://github.com/leggedrobotics/rsl_rl
- **描述**: GPU 加速的轻量级机器人学习库（你正在使用的）
- **相关性**:
  - ✅ 你当前的训练库
  - ✅ PPO 算法实现
  - ✅ GPU 加速
- **可学习点**:
  - **PPO 算法细节**
  - Actor-Critic 网络结构
  - 自定义网络扩展方法
  - LogWriter 自定义（新版本）

### 13. **leggedrobotics/rsl_rl_rwm** ⭐⭐⭐⭐
- **链接**: https://github.com/leggedrobotics/rsl_rl_rwm
- **描述**: RSL-RL + 机器人世界模型（Model-Based RL）
- **相关性**:
  - ✅ RSL-RL 的扩展
  - ✅ 世界模型
  - ✅ 提升样本效率
- **可学习点**:
  - 世界模型在机器人中的应用
  - Model-based RL
  - 不确定性估计

### 14. **pytorch/rl - ISAACLAB.md** ⭐⭐⭐⭐
- **链接**: https://github.com/pytorch/rl/blob/main/knowledge_base/ISAACLAB.md
- **描述**: PyTorch RL 库关于 IsaacLab 的知识库
- **相关性**:
  - ✅ IsaacLab 集成
  - ✅ 像素观测处理
  - ✅ 相机传感器配置
- **可学习点**:
  - **IsaacLab 与 TorchRL 集成**
  - 像素观测的正确添加方式
  - 观测空间管理

---

## 🏭 **工程实践**

### 15. **ethz-asl/moma** ⭐⭐⭐⭐
- **链接**: https://github.com/ethz-asl/moma
- **描述**: 移动抓取（Franka arm + Ridgeback base）
- **相关性**:
  - ✅ 移动底盘 + 机械臂
  - ✅ 手腕相机标定
  - ✅ 真实机器人部署
- **可学习点**:
  - 手腕相机标定方法
  - 移动抓取系统集成
  - ROS 框架组织

### 16. **UT-Austin-RobIn/telemoma** ⭐⭐⭐⭐
- **链接**: https://github.com/UT-Austin-RobIn/telemoma
- **描述**: 模块化移动操作遥操作系统
- **相关性**:
  - ✅ 移动操作
  - ✅ 多种遥操作接口（视觉、VR）
  - ✅ 数据收集
- **可学习点**:
  - 遥操作数据收集
  - 专家演示系统
  - 多模态接口设计

### 17. **RobotecAI/agentic-mobile-manipulator** ⭐⭐⭐⭐
- **链接**: https://github.com/RobotecAI/agentic-mobile-manipulator
- **描述**: 自主智能移动抓取机器人（仓库场景）
- **相关性**:
  - ✅ 移动抓取
  - ✅ 端到端感知、推理
  - ✅ 自然语言控制
- **可学习点**:
  - 端到端系统架构
  - 感知与规划集成
  - 仓库场景任务设计

---

## 📚 **学习路径建议**

### **阶段 1: 理解基础框架（1-2 周）**
1. 仔细阅读 **IsaacLab Cartpole Camera 示例** (#8)
2. 研究 **rsl_rl** 源码 (#12)，理解 PPO 实现
3. 学习 **NathanWu7/isaacLab.manipulation** (#5) 的项目结构

### **阶段 2: 移动抓取专题（2-3 周）**
4. 深入研究 **TIERS/isaac-marl-mobile-manipulation** (#1)
5. 参考 **YinpeiDai/rlmmbp** (#2) 的任务分解方法
6. 学习 **qaz9517532846/zm_robot** (#4) 的麦克纳姆轮建模

### **阶段 3: 视觉强化学习（2-3 周）**
7. 研究 **Ericonaldo/visual_wholebody** (#6) 的 CNN 架构
8. 参考 **pytorch/rl ISAACLAB.md** (#14) 的观测处理
9. 实现自定义 Actor-Critic 网络

### **阶段 4: Sim-to-Real 准备（1-2 周）**
10. 学习 **ByteDance-Seed/manip-as-in-sim-suite** (#9) 的域随机化
11. 参考 **THU-VCLab sim2real** (#10) 的迁移技巧
12. 实现光照、纹理、噪声随机化

---

## 🔍 **关键技术点对照表**

| 技术点 | 你的需求 | 推荐项目 |
|--------|---------|---------|
| **麦克纳姆轮建模** | ✅ | #4 zm_robot |
| **移动抓取协同** | ✅ | #1 isaac-marl, #2 rlmmbp |
| **IsaacLab Camera** | ✅ | #8 Cartpole Camera (官方) |
| **视觉 CNN 编码器** | ✅ | #6 visual_wholebody |
| **RSL-RL 网络扩展** | ✅ | #12 rsl_rl, #13 rsl_rl_rwm |
| **域随机化** | ⚠️ | #9 manip-as-in-sim, #10 sim2real |
| **手腕相机标定** | ⚠️ | #15 moma |
| **ROS2 集成** | ⚠️ | #7 RoboCupVisionRL |

---

## 💡 **额外资源**

### **论文**
- **Integration of the TIAGo Robot into Isaac Sim** (2024) - 麦克纳姆轮建模
  - https://arxiv.org/abs/2510.10273
- **Visual Sim-to-Real at Scale for Humanoid Loco-Manipulation** (2024)
  - https://arxiv.org/abs/2511.15200

### **官方文档**
- IsaacLab Camera 文档: https://isaac-sim.github.io/IsaacLab/source/overview/sensors/camera.html
- IsaacLab Observation Manager: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.managers.observation_manager.html

### **相关 Discussions/Issues**
- IsaacLab #2712: 非对称 Actor-Critic + 特权信息
  - https://github.com/isaac-sim/IsaacLab/issues/2712
- IsaacLab #4080: 如何录制带图像观测的演示
  - https://github.com/isaac-sim/IsaacLab/discussions/4080

---

## 🎯 **直接可用的代码片段来源**

| 需求 | 直接参考 |
|------|---------|
| 摄像头配置 | #8 IsaacLab cartpole_camera_env_cfg.py |
| 麦克纳姆轮 URDF | #4 zm_robot |
| 移动抓取奖励函数 | #1 isaac-marl-mobile-manipulation |
| 视觉 CNN Actor-Critic | #6 visual_wholebody |
| 域随机化配置 | #9 manip-as-in-sim-suite |

---

希望这份清单对你有帮助！建议从标记 ⭐⭐⭐⭐⭐ 的项目开始深入研究。
