# TODO 完成指南

本文档详细说明如何完成 README.md 中的待办事项（TODO）。

---

## **TODO 1: 标定麦轮几何与旋向** ⭐⭐⭐⭐⭐ (最重要)

### **目标**
让底盘的 `(vx, vy, ωz)` 体速度命令与真实运动一致。

### **需要标定的参数**

位置: [mdp/actions.py:127-138](eggtart_grasp/tasks/mobile_grasp/mdp/actions.py#L127-L138)

```python
# 当前值（需要测量真实机器人后修改）
wheel_radius: float = 0.043         # 轮子半径 r (m)
half_wheelbase: float = 0.08        # 半轴距 lx (前后轮间距的一半, m)
half_track: float = 0.097           # 半轮距 ly (左右轮间距的一半, m)
wheel_spin_sign: list[float] = [1.0, 1.0, -1.0, -1.0]  # 各轮旋转方向
```

### **标定步骤**

#### **步骤 1: 测量实物尺寸**

准备工具：卷尺、游标卡尺

```
测量项目：
1. 轮子直径 D → wheel_radius = D / 2
2. 前轮到后轮的距离 L_wheelbase → half_wheelbase = L_wheelbase / 2
3. 左轮到右轮的距离 L_track → half_track = L_track / 2
```

**测量示意图**:
```
     ← L_track →
    ┌─────────────┐  ↑
    │ FL      FR  │  │
    │  ●      ●   │  │ L_wheelbase
    │  ●      ●   │  │
    │ RL      RL  │  ↓
    └─────────────┘

轮子: ⊙ (直径 D)
```

#### **步骤 2: 确定轮子旋转方向**

**方法 A: 查看 URDF**
```bash
cd source/eggtart_grasp/eggtart_grasp/assets/urdf
grep -A 5 "wheel_.*_joint" robot.urdf
```

查看每个轮子的 `<axis xyz="..."/>`：
- `xyz="0 0 1"` → 正向旋转，sign = +1.0
- `xyz="0 0 -1"` → 反向旋转，sign = -1.0

**方法 B: 实物测试（推荐）**

创建测试脚本：

```python
# test_wheel_direction.py
import torch

# 假设参数
r = 0.043
lx = 0.08
ly = 0.097

# 测试：纯前进 (vx=1, vy=0, wz=0)
vx, vy, wz = 1.0, 0.0, 0.0

# 理论轮速（不考虑 spin_sign）
# FL: (vx - vy - (lx+ly)*wz) / r
# FR: (vx + vy + (lx+ly)*wz) / r
# RL: (vx + vy - (lx+ly)*wz) / r
# RR: (vx - vy + (lx+ly)*wz) / r

omega_FL = vx / r
omega_FR = vx / r
omega_RL = vx / r
omega_RR = vx / r

print(f"纯前进时，所有轮子应该同方向旋转: {omega_FL:.2f} rad/s")
```

在 Isaac Sim 中运行：
1. 给所有轮子发送相同的正速度命令
2. 观察机器人是否直线前进
3. 如果后退 → 对应轮子的 `spin_sign` 需要改为 -1.0
4. 如果侧移 → 某些轮子方向错误

**典型配置**（标准麦克纳姆轮）:
```python
# 配置 1: 常见布局
wheel_spin_sign = [1.0, 1.0, 1.0, 1.0]  # 所有轮子同向

# 配置 2: 对角布局
wheel_spin_sign = [1.0, -1.0, -1.0, 1.0]  # FL & RR 同向, FR & RL 反向

# 配置 3: 前后对称
wheel_spin_sign = [1.0, 1.0, -1.0, -1.0]  # 前轮同向，后轮反向（当前默认）
```

#### **步骤 3: 标定速度映射系数**

创建标定脚本：

```python
# calibrate_mecanum.py
"""
麦克纳姆轮标定脚本
用法: ./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_mecanum.py
"""
import torch
import time
import gymnasium as gym
import eggtart_grasp.tasks.mobile_grasp.config.eggtart

def test_forward(env, distance=1.0):
    """测试前进 distance 米，记录实际移动距离"""
    obs, _ = env.reset()
    start_pos = env.scene["robot"].data.root_pos_w[0, :2].clone()
    
    # 发送前进命令
    vx_cmd = 0.5  # m/s
    steps = int(distance / vx_cmd / env.step_dt)
    
    for _ in range(steps):
        action = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
        action[:, 0] = vx_cmd / 0.6  # 归一化到 [-1, 1]
        env.step(action)
    
    end_pos = env.scene["robot"].data.root_pos_w[0, :2]
    actual_dist = torch.norm(end_pos - start_pos).item()
    
    print(f"命令前进: {distance:.3f} m")
    print(f"实际前进: {actual_dist:.3f} m")
    print(f"比例: {actual_dist / distance:.3f}")
    print(f"建议 wheel_radius 修正: {0.043 * (actual_dist / distance):.4f} m")

def test_rotation(env, angle_deg=90):
    """测试旋转 angle_deg 度，记录实际旋转角度"""
    import math
    obs, _ = env.reset()
    start_quat = env.scene["robot"].data.root_quat_w[0].clone()
    
    # 发送旋转命令
    wz_cmd = 0.5  # rad/s
    target_rad = math.radians(angle_deg)
    steps = int(target_rad / wz_cmd / env.step_dt)
    
    for _ in range(steps):
        action = torch.zeros(env.num_envs, env.action_space.shape[0], device=env.device)
        action[:, 2] = wz_cmd / 1.5  # 归一化
        env.step(action)
    
    end_quat = env.scene["robot"].data.root_quat_w[0]
    # 计算实际旋转角度（简化版）
    from isaaclab.utils.math import quat_diff_rad
    actual_rad = quat_diff_rad(start_quat.unsqueeze(0), end_quat.unsqueeze(0))[0].item()
    actual_deg = math.degrees(abs(actual_rad))
    
    print(f"\n命令旋转: {angle_deg:.1f} °")
    print(f"实际旋转: {actual_deg:.1f} °")
    print(f"比例: {actual_deg / angle_deg:.3f}")
    k_current = 0.08 + 0.097  # lx + ly
    print(f"建议 (lx + ly) 修正: {k_current * (actual_deg / angle_deg):.4f} m")

if __name__ == "__main__":
    env = gym.make("Isaac-Mobile-Grasp-Eggtart-Play-v0", num_envs=1)
    
    print("=" * 60)
    print("麦克纳姆轮标定测试")
    print("=" * 60)
    
    test_forward(env, distance=1.0)
    test_rotation(env, angle_deg=90)
    
    env.close()
    
    print("\n" + "=" * 60)
    print("标定完成！请根据上述建议修改 mdp/actions.py 中的参数")
    print("=" * 60)
```

运行标定：
```bash
./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_mecanum.py
```

#### **步骤 4: 应用标定结果**

修改 [mdp/actions.py:127-138](eggtart_grasp/tasks/mobile_grasp/mdp/actions.py#L127-L138)：

```python
# 示例：假设标定结果显示实际前进了 1.2 倍距离
wheel_radius: float = 0.043 * 1.2  # = 0.0516

# 假设旋转测试显示实际旋转了 0.9 倍角度
# 当前 lx + ly = 0.177
# 修正后 = 0.177 * 0.9 = 0.159
# 重新分配（假设按原比例）
half_wheelbase: float = 0.08 * 0.9   # = 0.072
half_track: float = 0.097 * 0.9      # = 0.087

# 根据测试结果修改
wheel_spin_sign: list[float] = [1.0, 1.0, 1.0, 1.0]
```

---

## **TODO 2: 标定初始状态** ⭐⭐⭐⭐

### **目标**
确保机器人初始姿态合理（不碰撞地面、机械臂在可达范围内）

### **需要标定的参数**

位置: [assets/eggtart.py](eggtart_grasp/assets/eggtart.py)

#### **2.1 底盘离地高度**

查看当前配置：
```bash
cd source/eggtart_grasp/eggtart_grasp/assets
grep -A 2 "init_state" eggtart.py
```

**测量方法**:
1. 在 Isaac Sim 中加载机器人
2. 检查 `base_link` 底面到地面的距离
3. 观察轮子是否触地

**调整**:
```python
# 在 eggtart.py 的 EGGTART_CFG 中
init_state=ArticulationCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.15),  # Z 值 = 轮子半径 + 底盘厚度 + 安全间隙
    joint_pos={...}
)
```

**快速测试**:
```bash
# 启动可视化环境
./isaaclab.sh -p scripts/rsl_rl/play.py \
    --task Isaac-Mobile-Grasp-Eggtart-Play-v0 \
    --num_envs 1

# 观察：
# - 机器人是否悬空？→ Z 值太大
# - 底盘是否陷入地面？→ Z 值太小
# - 轮子是否正常接触地面？
```

#### **2.2 机械臂 Home 位姿**

**目标**: 让机械臂初始姿态舒展、无自碰撞、在工作空间内

**方法 A: 在 Isaac Sim 中手动调整**
```bash
# 1. 打开 Isaac Sim
cd /home/pu/isaac-sim
./isaac-sim.sh

# 2. 加载 URDF
# File → Import → robot.urdf

# 3. 手动旋转关节到合理位置
# 在 Property 面板调整各关节角度

# 4. 记录关节角度值
# 复制 Joint Position 值
```

**方法 B: 代码测试**

创建测试脚本：
```python
# test_home_pose.py
import torch
import gymnasium as gym
import eggtart_grasp.tasks.mobile_grasp.config.eggtart

env = gym.make("Isaac-Mobile-Grasp-Eggtart-Play-v0", num_envs=1)
obs, _ = env.reset()

robot = env.scene["robot"]

# 当前 home 位姿
home_pos = robot.data.default_joint_pos[0]
print("当前 Home 关节角度:")
for name, pos in zip(robot.joint_names, home_pos):
    print(f"  {name}: {pos:.3f} rad ({pos * 180 / 3.14159:.1f}°)")

# 测试末端位置
ee_pos = robot.data.body_pos_w[0, robot.find_bodies("end_effector")[0][0]]
print(f"\n末端位置: x={ee_pos[0]:.3f}, y={ee_pos[1]:.3f}, z={ee_pos[2]:.3f}")

# 检查自碰撞
# （需要手动观察可视化界面）

env.close()
```

**推荐 Home 姿态**（通用机械臂）:
```python
joint_pos={
    "link_001_joint": 0.0,      # 基座旋转，向前
    "link_002_joint": -0.5,     # 肩关节，略微抬起
    "link_003_joint": 1.0,      # 肘关节，弯曲
    "link_004_joint": 0.5,      # 腕关节 1
    "link_005_joint": 0.0,      # 腕关节 2
    "end_effector_joint": 0.8,  # 夹爪张开
}
```

#### **2.3 夹爪开/合角度**

**目标**: 确定夹爪完全张开和完全闭合时的关节角度

**测试方法**:
```python
# test_gripper_range.py
import torch
import gymnasium as gym
import eggtart_grasp.tasks.mobile_grasp.config.eggtart

env = gym.make("Isaac-Mobile-Grasp-Eggtart-Play-v0", num_envs=1)
env.reset()

robot = env.scene["robot"]
gripper_id = robot.find_joints("end_effector_joint")[0][0]

# 测试范围
print("测试夹爪范围...")
for angle in [-1.5, -1.0, -0.6, 0.0, 0.5, 1.0]:
    target = torch.tensor([[angle]], device=env.device)
    robot.set_joint_position_target(target, joint_ids=[gripper_id])
    
    # 步进仿真
    for _ in range(50):
        env.sim.step()
    
    actual = robot.data.joint_pos[0, gripper_id].item()
    print(f"  目标: {angle:.2f} rad → 实际: {actual:.2f} rad")
    input("按 Enter 继续...")

env.close()

print("\n根据观察，确定:")
print("  - 夹爪完全张开角度: ??? rad")
print("  - 夹爪完全闭合角度: ??? rad")
print("  - 更新 GRIPPER_CLOSED_THRESHOLD 在 mobile_grasp_env_cfg.py:38")
```

**应用结果**:

修改 [mobile_grasp_env_cfg.py:38](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L38)：
```python
GRIPPER_CLOSED_THRESHOLD = -0.6  # 改为实测的闭合角度
```

---

## **TODO 3: 真实抓取（刚性附着）** ⭐⭐⭐

### **目标**
实现真正的 pick-and-carry：目标物体被抓取后刚性附着到夹爪

### **当前问题**
- 抓取判定是瞬时的：`is_near & is_closed`
- 目标物体不会真正"粘"在夹爪上
- 目标物体可能掉落或穿透

### **解决方案 A: 使用 Fixed Joint 约束（物理方法）**

创建自定义环境类：

```python
# eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_with_attach.py
"""
扩展版环境，支持物理 attach 约束
"""
from isaaclab.envs import ManagerBasedRLEnv
import torch
from pxr import UsdPhysics, PhysxSchema

class MobileGraspEnvWithAttach(ManagerBasedRLEnv):
    """支持刚性附着的抓取环境"""
    
    def __init__(self, cfg, **kwargs):
        super().__init__(cfg, **kwargs)
        
        # 抓取状态 latch
        self._is_grasped = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._attach_joints = []  # 存储创建的约束
    
    def _attach_object(self, env_idx: int):
        """物理附着目标到夹爪"""
        # 获取夹爪和目标的 prim
        gripper_prim = self.scene["robot"].root_physx_view.get_link_path(
            env_idx, "end_effector"
        )
        target_prim = self.scene["target"].root_physx_view.get_object_path(env_idx)
        
        # 创建 Fixed Joint
        stage = self.sim.stage
        joint_path = f"{target_prim}/attach_joint"
        joint = UsdPhysics.FixedJoint.Define(stage, joint_path)
        joint.CreateBody0Rel().SetTargets([gripper_prim])
        joint.CreateBody1Rel().SetTargets([target_prim])
        
        self._attach_joints.append(joint_path)
        self._is_grasped[env_idx] = True
        print(f"[ENV {env_idx}] Object attached!")
    
    def _detach_object(self, env_idx: int):
        """分离物体"""
        if env_idx < len(self._attach_joints):
            stage = self.sim.stage
            stage.RemovePrim(self._attach_joints[env_idx])
            self._is_grasped[env_idx] = False
    
    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)
        
        # 检查抓取条件
        robot = self.scene["robot"]
        target = self.scene["target"]
        ee_pos = robot.data.body_pos_w[:, robot.find_bodies("end_effector")[0][0]]
        dist = torch.norm(ee_pos - target.data.root_pos_w, dim=1)
        
        gripper_id = robot.find_joints("end_effector_joint")[0][0]
        gripper_pos = robot.data.joint_pos[:, gripper_id]
        
        # 抓取触发条件
        can_grasp = (dist < 0.05) & (gripper_pos < -0.6) & (~self._is_grasped)
        
        for env_idx in torch.where(can_grasp)[0]:
            self._attach_object(env_idx.item())
        
        return obs, reward, terminated, truncated, info
    
    def reset(self, seed=None, options=None):
        # 重置时分离所有物体
        for env_idx in range(self.num_envs):
            if self._is_grasped[env_idx]:
                self._detach_object(env_idx)
        
        return super().reset(seed, options)
```

**注意**: 这需要深入理解 Isaac Sim 的约束系统，可能较复杂。

### **解决方案 B: 使用 Latch 状态 + 位置跟随（简化方法）**

修改奖励函数，添加 latch 逻辑：

```python
# mdp/rewards.py - 添加新函数
def grasp_with_latch(
    env: ManagerBasedRLEnv,
    reach_threshold: float,
    gripper_closed_threshold: float,
    ee_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    target_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """
    带 latch 的抓取奖励：一旦抓住就持续给奖励
    需要在 env 中维护 self._grasped_latch
    """
    robot: Articulation = env.scene[ee_cfg.name]
    target: RigidObject = env.scene[target_cfg.name]
    
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids[0]]
    dist = torch.norm(ee_pos_w - target.data.root_pos_w, dim=1)
    gripper_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids[0]]
    
    # 当前帧是否满足抓取条件
    is_grasping = (dist < reach_threshold) & (gripper_pos < gripper_closed_threshold)
    
    # Latch: 一旦抓住就锁定
    if not hasattr(env, '_grasped_latch'):
        env._grasped_latch = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    
    env._grasped_latch = env._grasped_latch | is_grasping
    
    # 检测松开（夹爪张开）
    gripper_opened = gripper_pos > (gripper_closed_threshold + 0.2)
    env._grasped_latch = env._grasped_latch & (~gripper_opened)
    
    return env._grasped_latch.float()
```

### **解决方案 C: 使用 IsaacLab 的 Attach Marker（推荐）**

IsaacLab 可能提供 `RigidObjectAttachment` 功能，查看文档：
```bash
# 搜索 IsaacLab 中的 attach 功能
cd /home/pu/isaac-sim/exts/isaaclab
grep -r "attach" --include="*.py" | grep -i "rigid"
```

参考官方示例（如果有）。

---

## **TODO 4: 添加滚子几何（获得真实横移）** ⭐⭐

### **目标**
让麦克纳姆轮在仿真中有真实的侧向滑动能力

### **当前问题**
URDF 中的轮子是纯圆柱体，没有麦轮特有的滚子几何，导致横向 `vy` 命令在物理上不正确。

### **解决方案 A: 添加滚子几何（复杂）**

需要在 URDF/USD 中为每个轮子添加 12-16 个小滚子：

```xml
<!-- 麦克纳姆轮的滚子定义（示例） -->
<link name="wheel_FL_roller_01">
  <visual>
    <geometry>
      <cylinder radius="0.008" length="0.02"/>
    </geometry>
  </visual>
  <collision>
    <geometry>
      <cylinder radius="0.008" length="0.02"/>
    </geometry>
  </collision>
</link>

<!-- 重复 12-16 次，每个滚子旋转 45° -->
```

**推荐工具**: 使用 CAD 软件导出完整的麦轮模型，或使用现成的麦轮 URDF。

**参考项目**:
- [zm_robot](https://github.com/qaz9517532846/zm_robot) - 麦克纳姆轮在 Isaac Sim 中的实现
- ROS mecanum_wheel URDF 包

### **解决方案 B: 使用全向驱动（推荐，简单）**

不使用轮速控制，直接控制底盘根节点速度：

修改 `MecanumBaseAction` 为 `HolonomicBaseAction`：

```python
# mdp/actions.py - 新增
class HolonomicBaseAction(ActionTerm):
    """直接控制底盘根节点体速度（全向移动）"""
    
    def apply_actions(self) -> None:
        # 直接设置根节点体速度
        body_vel = self._processed_actions  # (num_envs, 3): vx, vy, wz
        
        # 转换为世界系速度
        root_quat = self._asset.data.root_quat_w
        lin_vel_b = body_vel[:, :2]  # (vx, vy) in body frame
        ang_vel_z = body_vel[:, 2:3]  # wz
        
        # 旋转到世界系
        from isaaclab.utils.math import quat_rotate
        lin_vel_w = quat_rotate(root_quat, torch.cat([lin_vel_b, torch.zeros_like(ang_vel_z)], dim=1))
        
        # 设置速度
        self._asset.write_root_velocity_to_sim(
            torch.cat([lin_vel_w[:, :2], torch.zeros_like(ang_vel_z), 
                      torch.zeros_like(lin_vel_b), ang_vel_z], dim=1)
        )
```

**优点**:
- 物理上完美全向移动
- 不需要复杂的轮子几何
- 训练更快

**缺点**:
- 不符合真实物理（轮子不转动）
- sim-to-real 需要额外映射层

---

## **TODO 5: 调整奖励权重和课程学习** ⭐⭐⭐

### **目标**
优化训练效率和策略性能

### **当前权重**

位置: [mobile_grasp_env_cfg.py:183-242](eggtart_grasp/tasks/mobile_grasp/mobile_grasp_env_cfg.py#L183-L242)

```python
base_approach = RewTerm(..., weight=1.0)     # 底盘接近
ee_reach = RewTerm(..., weight=2.0)          # 末端 reach
ee_distance = RewTerm(..., weight=-0.1)      # 距离惩罚
grasp = RewTerm(..., weight=5.0)             # 抓取 bonus
retract = RewTerm(..., weight=2.0)           # 收臂
action_rate = RewTerm(..., weight=-0.001)    # 动作平滑
joint_vel = RewTerm(..., weight=-0.0001)     # 关节速度
```

### **调参建议**

#### **方法 1: 手动调参**

创建参数扫描脚本：
```python
# scripts/tune_rewards.py
import itertools

# 定义参数网格
params = {
    "base_approach": [0.5, 1.0, 2.0],
    "ee_reach": [1.0, 2.0, 4.0],
    "grasp": [3.0, 5.0, 10.0],
}

for combo in itertools.product(*params.values()):
    config = dict(zip(params.keys(), combo))
    print(f"Testing: {config}")
    # 运行训练...
```

**经验法则**:
- 阶段性奖励按重要性递增：底盘 < 到达 < 抓取 < 收臂
- 比例建议：1 : 2 : 5 : 2
- 惩罚项权重应该小（-0.001 ~ -0.01）

#### **方法 2: 课程学习（Curriculum）**

分阶段训练：

```python
# config/eggtart/grasp_env_curriculum_cfg.py
@configclass
class CurriculumEnvCfg(EggtartMobileGraspEnvCfg):
    """课程学习版本"""
    
    def __post_init__(self):
        super().__post_init__()
        
        # 阶段 1 (0-500 iter): 只奖励底盘接近
        # 阶段 2 (500-1000 iter): 加入末端 reach
        # 阶段 3 (1000+ iter): 完整任务
        
        # 需要在训练脚本中动态调整权重
```

训练脚本修改：
```python
# scripts/rsl_rl/train.py - 添加 curriculum 逻辑
def update_curriculum(env, iteration):
    if iteration < 500:
        env.reward_manager.terms["ee_reach"].weight = 0.0
        env.reward_manager.terms["grasp"].weight = 0.0
    elif iteration < 1000:
        env.reward_manager.terms["ee_reach"].weight = 2.0
        env.reward_manager.terms["grasp"].weight = 0.0
    else:
        env.reward_manager.terms["grasp"].weight = 5.0
```

#### **方法 3: 自动调参（Optuna）**

```python
# scripts/optimize_rewards.py
import optuna

def objective(trial):
    # 定义超参数
    w_base = trial.suggest_float("w_base", 0.1, 3.0)
    w_ee = trial.suggest_float("w_ee", 0.5, 5.0)
    w_grasp = trial.suggest_float("w_grasp", 2.0, 10.0)
    
    # 修改配置
    env_cfg.rewards.base_approach.weight = w_base
    env_cfg.rewards.ee_reach.weight = w_ee
    env_cfg.rewards.grasp.weight = w_grasp
    
    # 运行训练
    final_reward = run_training(env_cfg, max_iter=500)
    return final_reward

# 运行优化
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)
```

---

## **📊 完成优先级总结**

| TODO | 重要性 | 难度 | 时间估算 | 建议顺序 |
|------|--------|------|---------|---------|
| **标定麦轮几何** | ⭐⭐⭐⭐⭐ | 中 | 2-4 小时 | **1** |
| **标定初始状态** | ⭐⭐⭐⭐ | 低 | 1-2 小时 | **2** |
| **调整奖励权重** | ⭐⭐⭐ | 低 | 持续调优 | **3** |
| **真实抓取** | ⭐⭐⭐ | 高 | 1-2 天 | **5** (可选) |
| **添加滚子几何** | ⭐⭐ | 高 | 2-3 天 | **6** (可选) |

---

## **🚀 快速开始流程**

### **第 1 天: 基础标定**
```bash
# 1. 测量实物尺寸
# 2. 运行麦轮标定脚本
./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_mecanum.py

# 3. 修改 mdp/actions.py 参数
# 4. 测试初始状态
./isaaclab.sh -p scripts/rsl_rl/play.py --task Isaac-Mobile-Grasp-Eggtart-Play-v0 --num_envs 1
```

### **第 2 天: 开始训练**
```bash
# 5. 用标定后的参数训练
./isaaclab.sh -p scripts/rsl_rl/train.py --task Isaac-Mobile-Grasp-Eggtart-v0 --num_envs 2048

# 6. 监控训练曲线，调整奖励权重
tensorboard --logdir logs/rsl_rl/eggtart_mobile_grasp
```

### **后续: 持续优化**
- 微调奖励权重
- 实现课程学习
- 考虑真实抓取和滚子几何（如果 sim-to-real）

---

## **📚 参考资料**

- 麦克纳姆轮运动学: https://research.ijcaonline.org/volume113/number3/pxc3901586.pdf
- IsaacLab Articulation API: https://isaac-sim.github.io/IsaacLab/source/api/lab/isaaclab.assets.articulation.html
- 课程学习论文: "Automatic Curriculum Learning For Deep RL" (2017)
