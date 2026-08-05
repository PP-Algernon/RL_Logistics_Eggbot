"""Package containing task implementations for the Eggtart mobile manipulator."""

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##

# The blacklist is used to prevent importing configs from sub-packages.
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package (triggers gym.register in each config sub-package).
import_packages(__name__, _BLACKLIST_PKGS)
