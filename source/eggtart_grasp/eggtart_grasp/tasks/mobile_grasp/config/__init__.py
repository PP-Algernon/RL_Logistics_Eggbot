"""Concrete environment configurations for the mobile-grasp task.

This file is intentionally minimal -- importing the ``eggtart`` sub-package triggers the
``gym.register`` calls for the Eggtart robot.
"""

from . import eggtart  # noqa: F401
