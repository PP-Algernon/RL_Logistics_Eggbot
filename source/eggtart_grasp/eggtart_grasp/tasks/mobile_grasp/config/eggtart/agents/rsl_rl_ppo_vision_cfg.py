"""RSL-RL PPO runner config with vision encoder for the Eggtart mobile-grasp task."""

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class EggtartMobileGraspPPOVisionRunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO configuration with CNN encoder for visual observations.

    This config uses a CNN to process RGB+Depth images from the wrist camera,
    then concatenates the visual features with proprioceptive observations.
    """

    num_steps_per_env = 24
    max_iterations = 3000  # More iterations needed for vision-based training
    save_interval = 50
    experiment_name = "eggtart_mobile_grasp_vision"
    run_name = ""

    # Enable empirical normalization for observations (helpful with vision)
    empirical_normalization = True

    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=True,  # Enable for vision
        critic_obs_normalization=True,
        # MLP layers after CNN feature extraction
        actor_hidden_dims=[256, 128, 64],
        critic_hidden_dims=[256, 128, 64],
        activation="elu",
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,  # Slightly higher entropy for exploration with vision
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=3.0e-4,  # Lower LR for vision-based training
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
