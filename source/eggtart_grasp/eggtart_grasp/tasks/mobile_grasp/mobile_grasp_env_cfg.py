"""Eggtart 移动抓取任务的基础 ManagerBasedRLEnvCfg 配置

场景：带麦轮的移动机械臂在地面上追逐并抓取移动目标物体（小浮动立方体，代表蛋挞载荷）。

机器人本身的 articulation 配置在此处留空（MISSING），由具体配置文件
``config/eggtart/grasp_env_cfg.py`` 填充。
"""

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import eggtart_grasp.tasks.mobile_grasp.mdp as mdp
from eggtart_grasp.assets.eggtart import (
    EGGTART_ARM_JOINT_NAMES,
    EGGTART_BASE_FORWARD_AXIS,
    EGGTART_EE_GRASP_OFFSET,
    EGGTART_GRIPPER_JOINT_NAME,
    EGGTART_WHEEL_JOINT_BODY_REGEX,
    EGGTART_WHEEL_JOINT_NAMES,
)

# ---------------------------------------------------------------------------
# Tunable task constants
# ---------------------------------------------------------------------------
# 抓取点落在这个距离内算"到达目标"
# 调整建议：如果EE一直到不了，可以放宽到0.08甚至0.10；等学会了再收紧
GRASP_REACH_THRESHOLD = 0.08      # m (放宽让policy更容易触发grasp)

# 夹爪关节低于此角度算"闭合"
#
# **改成力控后这个值必须跟着改**：位置控制时空夹能压到硬限位 -0.2，所以阈值 -0.15
# 合理；但力控下夹住物体时夹爪**被物体挡住**，根本到不了 -0.2——
# 实测 30 mm 立方体停在 q≈0.29，40 mm 停在 q≈0.40。
# 如果还用 -0.15，"夹住了"这个条件永远不成立，grasp / retract 奖励恒为 0。
#
# 0.35 的依据：目标立方体 30 mm 停在 0.29，留一点余量；同时 0.35 对应开口约 44 mm，
# 比物体宽——也就是"明显在往里夹但还没夹到"不会误判成已闭合。
# 换目标尺寸要重算：开口-角度对应关系见 assets/eggtart.py 的注释表。
GRIPPER_CLOSED_THRESHOLD = 0.35

# 抓取点必须在 GRASP_REACH_THRESHOLD 内**连续停留**这么久，闭爪才算有效抓取
# env.step_dt = (1/120) * 4 = 1/30 s，所以 0.3 s ≈ 9 步，0.6 s ≈ 18 步
# 调整建议：如果grasp一直是0，先降到0.2让policy能拿到奖励，再逐步提高要求
GRASP_DWELL_TIME = 0.2  # s (降低难度，让policy先学会基本动作)


##
# 场景定义
##
@configclass
class MobileGraspSceneCfg(InteractiveSceneCfg):
    """场景配置：地面、光照、移动机械臂和移动目标"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 机器人 -- 由具体配置填充
    robot: ArticulationCfg = MISSING

    # 移动目标：小立方体，禁用重力使其能以恒定高度漂移
    target = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Target",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                linear_damping=0.0,
                angular_damping=0.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.75, 0.25)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.1)),
    )

    # 光照
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )


##
# MDP 设置
##
@configclass
class ActionsCfg:
    """动作项配置：3 维全向底盘速度 + 机械臂关节位置 + 夹爪"""

    base_velocity = mdp.HolonomicBaseActionCfg(
        asset_name="robot",
        max_lin_vel_x=0.6,
        max_lin_vel_y=0.6,
        max_ang_vel_z=1.5,
    )
    arm_action = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=EGGTART_ARM_JOINT_NAMES,
        scale=0.5,
        use_default_offset=True,
    )
    # 夹爪改为**力控**：策略输出夹持力，最终停在哪由接触自然决定，自适应物体体积。
    # 位置控制下策略必须精确命令一个关节角，而"物体多大就该停在哪个角度"是几何决定的
    # （实测 20/30/40 mm 物体分别停在 q=0.116/0.290/0.400），位置控制要么闭不到位、
    # 要么把物体挤穿。详见 actions.py 的 GripperForceAction。
    # max_effort 实测：0.3 Nm 能稳定夹住 30 mm 立方体，3.0 Nm 会把它挤穿。
    gripper_action = mdp.GripperForceActionCfg(
        asset_name="robot",
        joint_names=[EGGTART_GRIPPER_JOINT_NAME],
        max_effort=1.0,
        damping=0.05,
    )


@configclass
class ObservationsCfg:
    """MDP 的观测规格配置"""

    @configclass
    class PolicyCfg(ObsGroup):
        # 本体感知
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.05, n_max=0.05))
        # 任务相关
        target_position_b = ObsTerm(
            func=mdp.target_position_in_base_frame,
            params={"robot_cfg": SceneEntityCfg("robot"), "target_cfg": SceneEntityCfg("target")},
        )
        # 用真实抓取点（两爪之间），和奖励保持同一个基点
        ee_to_target_b = ObsTerm(
            func=mdp.ee_to_target_vector_base_frame,
            params={
                "robot_cfg": SceneEntityCfg("robot"),
                "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
                "target_cfg": SceneEntityCfg("target"),
                "grasp_offset": EGGTART_EE_GRASP_OFFSET,
            },
        )
        target_lin_vel = ObsTerm(
            func=mdp.target_lin_vel_w, params={"target_cfg": SceneEntityCfg("target")}
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """重置和间隔事件配置"""

    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.1, 0.1), "y": (-0.1, 0.1), "yaw": (-3.14, 3.14)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.9, 1.1), "velocity_range": (0.0, 0.0)},
    )
    reset_target = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # 在机器人前方/周围生成目标，处于可抓取高度
            "pose_range": {"x": (0.4, 1.2), "y": (-0.6, 0.6), "z": (0.15, 0.30)},
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )
    # 定期改变目标运动方向，使其成为真正的移动目标
    randomize_target_velocity = EventTerm(
        func=mdp.randomize_target_velocity,
        mode="interval",
        interval_range_s=(2.0, 4.0),
        params={
            "velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)},
            "asset_cfg": SceneEntityCfg("target"),
        },
    )


@configclass
class RewardsCfg:
    """分阶段奖励项配置（权重可调整）"""

    # 阶段 1: 底盘运动
    # base_cfg 必须显式传：底盘位置要用轮心（几何中心），不能用 root_pos_w。
    # base_link 原点在车身外 0.435 m（实测轮心在 base_link 系 = -0.333, -0.282, 0.042），
    # 用 root 原点会让真正的车身停在离目标 0.435 m 的左后方。详见 rewards.py 的 _base_center_w。
    # standoff: 底盘中心停在目标外 0.35 m，留出机械臂伸展的空间（不是越近越好）。
    base_approach = RewTerm(
        func=mdp.base_to_target_xy_tanh,
        weight=1.0,
        params={
            "std": 0.5,
            "standoff": 0.35,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
        },
    )
    # 底盘朝向（工作面对准目标）
    # std 是夹角误差的高斯核宽度（弧度），0.6 rad ≈ 34°
    # forward_axis 是底盘工作面在 base_link 系中的方向，本机器人为 -Y（见 rewards.py 说明）
    base_facing = RewTerm(
        func=mdp.base_facing_target,
        weight=1.0,
        params={
            "std": 0.6,
            "forward_axis": EGGTART_BASE_FORWARD_AXIS,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
        },
    )
    # 阶段 2: 末端执行器到达
    # grasp_offset: 用真实抓取点（两爪之间），不是 end_effector body 原点。
    # 两者差 2.8 cm（主要在前方），和 GRASP_REACH_THRESHOLD=0.05 同量级——
    # 不修的话策略把 body 原点怼到目标上，抓取点其实还差 2.7 cm，夹爪合上是空的。
    ee_reach = RewTerm(
        func=mdp.ee_to_target_tanh,
        weight=3.0,
        params={
            "std": 0.1,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    ee_distance = RewTerm(
        func=mdp.ee_to_target_distance_l2,
        weight=-0.2,
        params={
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 阶段 3: 抓取
    # 用带停留计时的版本：抓取点必须在阈值内**连续停留** GRASP_DWELL_TIME 秒，
    # 之后闭爪才算有效。原来的瞬时判定让策略学会边冲边提前闭爪——闭爪零成本，
    # 早闭还能提高"某一帧恰好同时满足两条件"的概率，于是夹爪总是夹早了。
    # 计数器是 per-env 的，离开阈值立刻归零（要求连续，不是累计），
    # episode 重置时由 RewardManager 自动清零（类式奖励项才有这个能力）。

    # 引导奖励：接近目标时鼓励闭合夹爪（稠密奖励，帮助policy学会基本动作）
    gripper_close_guide = RewTerm(
        func=mdp.gripper_close_when_near,
        weight=2.0,  # 稠密引导，在grasp之前帮助policy建立"靠近->闭合"的关联
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )

    # 稀疏抓取奖励（最终目标，需要停留+闭合）
    grasp = RewTerm(
        func=mdp.grasp_bonus_dwell,
        weight=5.0,
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "dwell_time": GRASP_DWELL_TIME,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 主动惩罚"还没到位就闭爪"。
    # 只加停留计时是消极约束（拿不到分但也不亏），策略可能保持闭爪的习惯；
    # 这一项让提前闭爪真的要花钱，"张爪接近 -> 到位稳住 -> 再闭合"才最优。
    gripper_early = RewTerm(
        func=mdp.gripper_premature_close,
        weight=-0.01,
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 阶段 4: 抓取后回收
    retract = RewTerm(
        func=mdp.retract_bonus,
        weight=2.0,
        params={
            "reach_threshold": GRASP_REACH_THRESHOLD,
            "gripper_closed_threshold": GRIPPER_CLOSED_THRESHOLD,
            "std": 0.5,
            "arm_cfg": SceneEntityCfg("robot", joint_names=EGGTART_ARM_JOINT_NAMES),
            "ee_cfg": SceneEntityCfg("robot", body_names="end_effector"),
            "gripper_cfg": SceneEntityCfg("robot", joint_names=[EGGTART_GRIPPER_JOINT_NAME]),
            "target_cfg": SceneEntityCfg("target"),
            "grasp_offset": EGGTART_EE_GRASP_OFFSET,
        },
    )
    # 惩罚项
    # action_rate 权重从 -0.001 提到 -0.02。
    # 实测（恒定动作跑 200 步）关节位置峰峰值只有 ~1e-6 rad，物理侧完全干净，
    # 所以抖动是**策略在输出 bang-bang 动作**，不是仿真数值问题。
    # 旧权重下：臂+爪 6 维在 ±1 之间来回跳 -> action_rate_l2 ≈ 6*(2^2) = 24,
    # 代价 24*0.001 = 0.024/步，而 ee_reach 靠近目标时单步就有 ~2.0，
    # 抖动只花掉主奖励的 ~1%，几乎免费。-0.02 让同样的抖动代价约 0.48，量级才可比。
    action_rate = RewTerm(
        func=mdp.action_rate_l2, weight=-0.02
    )

    # joint_vel 权重 -0.0001 -> -0.002，并把**夹爪**纳入（原来只有 5 个臂关节，
    # 夹爪疯狂开合完全不受罚）。这一项罚的是关节实际速度，和 action_rate 互补：
    # action_rate 管指令的跳变，joint_vel 管真实的高频运动。
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.002,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=EGGTART_ARM_JOINT_NAMES + [EGGTART_GRIPPER_JOINT_NAME]
            )
        },
    )

    # 关节到极限位置的惩罚（用 Isaac Lab 自带项）
    # joint_pos_limits 罚的是超出 **软限位** 的部分：软限位 = soft_joint_pos_limit_factor(0.95)
    # × URDF 硬限位，也就是行程最外侧 5% 那一圈。关节在中间区域时该项恒为 0，
    # 只有压到边界才产生代价，所以不会干扰正常运动。
    # 只作用在机械臂 + 夹爪上：四个轮子关节是 continuous（URDF 里没有 limit），无极限可言。
    joint_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot", joint_names=EGGTART_ARM_JOINT_NAMES + [EGGTART_GRIPPER_JOINT_NAME]
            )
        },
    )

    # 底盘速度惩罚（治"追到目标后绕着转"）
    # 现有的 action_rate / joint_vel 治不了这个毛病：
    #   - action_rate 罚动作**变化量**，匀速绕圈时动作近似恒定，几乎不花钱；
    #   - joint_vel 只覆盖机械臂关节，而且 HolonomicBaseAction 是直接写根节点速度的，
    #     轮子关节速度恒为 0，基于关节的惩罚根本约束不到底盘。
    # 绕圈的根因是奖励退化：base_approach 在"距离==standoff"最大、base_facing 在"对准"最大，
    # 这两个条件在半径 0.35 m 的**整个圆周**上同时满足，沿切向漂移不损失奖励。
    # 该项带"到位 × 对准"门控，只在停好之后才罚速度，不和 base_approach 对抗
    # （目标每 2~4 s 重随机速度，底盘本来就得重新追）。详见 rewards.py 的 base_velocity_l2。
    base_vel = RewTerm(
        func=mdp.base_velocity_l2,
        weight=-0.05,
        params={
            "standoff": 0.35,  # 与 base_approach 保持一致
            "arrive_tol": 0.15,
            "align_tol": 0.6,  # 与 base_facing 的 std 保持一致
            "ang_vel_scale": 0.3,
            "forward_axis": EGGTART_BASE_FORWARD_AXIS,
            "robot_cfg": SceneEntityCfg("robot"),
            "target_cfg": SceneEntityCfg("target"),
            "base_cfg": SceneEntityCfg("robot", body_names=EGGTART_WHEEL_JOINT_BODY_REGEX),
        },
    )


@configclass
class CurriculumCfg:
    """课程学习配置：按训练步数分阶段打开各奖励项

    时间表用 ``common_step_counter``（每次 env.step 加 1，与 num_envs 无关）计时。
    本项目 num_steps_per_env = 24，所以 step = 迭代数 × 24：

        阶段 1 (iter 0-500,    step 0-12000)  : 只学底盘接近 + 朝向
        阶段 2 (iter 500-1000, step 12000-24000): 加入末端执行器到达
        阶段 3 (iter 1000+,    step 24000+)   : 完整任务（抓取 + 回收）

    上面 RewardsCfg 里写的权重是**最终阶段**的值，课程会在前期把还没到的阶段压成 0。
    改 num_steps_per_env 要同步改这里的阈值。
    每一项的当前权重会记到 TensorBoard 的 ``Curriculum/<name>``，可以直接看切换时机。
    """

    # 底盘两项全程开启，是后面所有阶段的基础
    base_approach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_approach", "schedule": [(0, 1.0)]},
    )
    base_facing_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_facing", "schedule": [(0, 1.0)]},
    )
    # 末端执行器：阶段 2 打开（提前到iter 250，让EE更早开始学习）
    ee_reach_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_reach", "schedule": [(0, 0.0), (9000, 5.0)]},  # 9000步≈iter 375，权重5.0加大引导
    )
    ee_distance_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "ee_distance", "schedule": [(0, 0.0), (9000, -0.2)]},
    )

    # 引导奖励：在ee_reach之后、grasp之前启用，帮助policy学会"接近->闭合"
    gripper_close_guide_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_close_guide", "schedule": [(0, 0.0), (9000, 2.0)]}, 
    )

    # 抓取和回收：阶段 3 打开（提前到iter 500）
    grasp_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "grasp", "schedule": [(0, 0.0), (12000, 10.0)]},  # 12000步≈iter 500，权重10.0让grasp更有吸引力
    )
    # "提前闭爪"的惩罚和 grasp 同步打开，但权重降低避免过度抑制
    gripper_early_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "gripper_early", "schedule": [(0, 0.0), (12000, -0.005)]},  # 降到-0.005，让policy敢尝试夹
    )
    retract_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "retract", "schedule": [(0, 0.0), (30000, 2.0)]},
    )
    # 底盘速度惩罚：延后到 step 3000（≈iter 125）再打开。
    # 一开始底盘还不会走，就罚它动会拖慢学习；等它大致学会接近了再要求"停住"。
    base_vel_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "base_vel", "schedule": [(0, 0.0), (9000, -0.05)]},
    )
    # 关节限位惩罚全程开启：从一开始就不该往限位上顶
    joint_limits_sched = CurrTerm(
        func=mdp.reward_weight_schedule,
        params={"term_name": "joint_limits", "schedule": [(0, -0.5)]},
    )

@configclass
class TerminationsCfg:
    """MDP 终止条件配置"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_tipped = DoneTerm(func=mdp.base_tipped, params={"min_up_proj": 0.5})


##
# 环境配置
##
@configclass
class MobileGraspEnvCfg(ManagerBasedRLEnvCfg):
    """Eggtart 移动抓取环境的基础配置"""

    # 场景
    scene: MobileGraspSceneCfg = MobileGraspSceneCfg(num_envs=2048, env_spacing=3.0)
    # MDP
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        # 通用设置
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.render_interval = self.decimation
        self.viewer.eye = (4.0, 4.0, 3.0)
        # 仿真设置
        self.sim.dt = 1.0 / 120.0
