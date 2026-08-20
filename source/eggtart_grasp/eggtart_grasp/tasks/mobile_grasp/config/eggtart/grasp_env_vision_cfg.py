"""Vision-based Eggtart mobile-grasp environment config with wrist camera."""

from __future__ import annotations

from isaaclab.utils import configclass

from eggtart_grasp.assets.eggtart import EGGTART_CFG
from eggtart_grasp.tasks.mobile_grasp.mobile_grasp_env_cfg import MobileGraspEnvCfg


@configclass
class EggtartMobileGraspVisionEnvCfg(MobileGraspEnvCfg):
    """Vision-based training configuration with wrist-mounted camera.

    This config enables RGB+Depth visual observations for the mobile-grasp task.
    Training will be slower but prepares the policy for sim-to-real transfer.
    """

    def __post_init__(self):
        super().__post_init__()
        # Attach the Eggtart robot to the scene
        self.scene.robot = EGGTART_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # Reduce number of parallel environments for vision (memory constraint)
        self.scene.num_envs = 512  # Down from 2048

        # Enable camera rendering
        self.sim.render_interval = self.decimation  # Already set in parent


@configclass
class EggtartMobileGraspVisionEnvCfg_PLAY(EggtartMobileGraspVisionEnvCfg):
    """Reduced-scale config for visualisation / evaluation with camera."""

    def __post_init__(self):
        super().__post_init__()
        # Fewer, more-spaced environments and no observation noise
        self.scene.num_envs = 32  # Very small for interactive visualization
        self.scene.env_spacing = 3.0
        self.observations.policy.enable_corruption = False
