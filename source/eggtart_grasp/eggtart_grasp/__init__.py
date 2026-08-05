"""Eggtart mobile-manipulation Isaac Lab extension package.

Registers the Gym environments for the Mecanum-wheeled mobile manipulator
grasping a moving target.
"""

# Register Gym environments.
from .tasks import *  # noqa: F401, F403
