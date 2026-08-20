"""Gym environment registrations for the Eggtart mobile-grasp task."""

import gymnasium as gym

from . import agents, grasp_env_cfg, grasp_env_vision_cfg

##
# Register Gym environments.
##

gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_cfg.EggtartMobileGraspEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
    },
)

gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_cfg.EggtartMobileGraspEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EggtartMobileGraspPPORunnerCfg",
    },
)

# Vision-based variants with wrist camera
gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-Vision-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_vision_cfg.EggtartMobileGraspVisionEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_vision_cfg:EggtartMobileGraspPPOVisionRunnerCfg",
    },
)

gym.register(
    id="Isaac-Mobile-Grasp-Eggtart-Vision-Play-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": grasp_env_vision_cfg.EggtartMobileGraspVisionEnvCfg_PLAY,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_vision_cfg:EggtartMobileGraspPPOVisionRunnerCfg",
    },
)
