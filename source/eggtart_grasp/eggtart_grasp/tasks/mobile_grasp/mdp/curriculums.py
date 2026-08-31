"""Eggtart 移动抓取任务的课程学习项

为什么课程要放在这里，而不是在 train.py 里手动调权重：
    之前 train.py 里是这么写的::

        for iteration in range(max_iterations):
            update_curriculum(env, iteration)
            runner.learn(num_learning_iterations=1, ...)

    这个写法有 bug。rsl_rl 的 ``OnPolicyRunner.learn`` 结尾是
    ``self.current_learning_iteration = it``（不是 ``it + 1``），而每次调用又从
    ``start_iter = self.current_learning_iteration`` 开始。所以每次 ``learn(1)``
    都在重跑第 0 次迭代：

    - TensorBoard 所有点都写在 step 0（曲线全平，实测两次迭代 ``Train/mean_reward``
      记录为 ``[(0, 0.032), (0, 0.008)]``）
    - checkpoint 反复覆盖 ``model_0.pt``，跑 N 次迭代也只有一个文件
    - ``rewbuffer``/``lenbuffer`` 每次调用都重建，24 步内几乎没有 episode 结束，
      ``Mean reward`` 基本是噪声

    梯度更新其实照做了，但日志完全不可用。改用 Isaac Lab 原生的 CurriculumManager：
    权重在环境 step 里改，``learn()` 只调一次，rsl_rl 的计数器/日志/缓冲区都正常。

关于步数单位：
    课程用 ``env.common_step_counter`` 计时，它每次 ``env.step()`` 加 1，与
    ``num_envs`` 无关。所以 ``迭代数 = common_step_counter / num_steps_per_env``。
    本项目 ``num_steps_per_env = 24``（见 agents/rsl_rl_ppo_cfg.py），因此：

        迭代 500  -> step 12000
        迭代 1000 -> step 24000

    改了 ``num_steps_per_env`` 就要同步改 env-cfg 里的阈值。

关于生效时机：
    ``CurriculumManager.compute()`` 只在 ``ManagerBasedRLEnv._reset_idx()`` 里被调用，
    也就是**只在有环境重置时**求值，不是每步都算。所以权重切换会比阈值滞后，
    最多滞后一个 episode（本项目 episode_length_s = 10 s，约 300 步）。
    并行环境多的时候几乎每步都有环境在重置，实际滞后可以忽略。
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

from isaaclab.managers import CurriculumTermCfg, ManagerTermBase

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

class reward_weight_schedule(ManagerTermBase):
    """按步数分段设置某个奖励项的权重

    与 Isaac Lab 自带的 ``modify_reward_weight``（单个阈值、只能切一次）不同，
    这里接受一个完整的 (起始步数, 权重) 时间表，一个课程项就能覆盖全部阶段，
    env-cfg 里每个奖励项只需写一行，阶段划分也集中在一处。

    Args:
        term_name: 要调整的奖励项名字，须与 RewardsCfg 中的字段名一致
        schedule: ``[(起始步数, 权重), ...]``，按起始步数升序。取最后一个
            ``起始步数 <= common_step_counter`` 的权重。第一项通常是 ``(0, 初始权重)``。

    Returns:
        当前权重（float），会被 CurriculumManager 记到日志里，
        TensorBoard 中可以看到 ``Curriculum/<term>`` 曲线，方便确认阶段切换时机。
    """

    def __init__(self, cfg: CurriculumTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._term_name = cfg.params["term_name"]
        self._term_cfg = env.reward_manager.get_term_cfg(self._term_name)
        # 排序一次，避免每步都排
        self._schedule = sorted(cfg.params["schedule"], key=lambda kv: kv[0])

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: Sequence[int],
        term_name: str,
        schedule: Sequence[tuple[int, float]],
    ) -> float:
        # 找当前步数对应的权重
        weight = self._schedule[0][1]
        for start_step, w in self._schedule:
            if env.common_step_counter >= start_step:
                weight = w
            else:
                break

        # 只在变化时写回，省掉每步的 set_term_cfg
        if weight != self._term_cfg.weight:
            self._term_cfg.weight = weight
            env.reward_manager.set_term_cfg(term_name, self._term_cfg)
            print(f"[Curriculum] step {env.common_step_counter}: {term_name} 权重 -> {weight}")

        return weight
