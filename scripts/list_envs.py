"""
Script to print all the available environments in the extension.

The script iterates over all registered environments and stores the details in a table.
It prints the name of the environment, the entry point and the config file.
"""

"""Launch Isaac Sim Simulator first."""

from isaaclab.app import AppLauncher

# launch omniverse app
app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app


"""Rest everything follows."""

import gymnasium as gym
from prettytable import PrettyTable

# Import extensions to set up environment tasks
import eggtart_grasp.tasks  # noqa: F401


def _fmt(obj) -> str:
    """Return a short readable name for a class or string entry point."""
    if isinstance(obj, type):
        return f"{obj.__module__.split('.')[-1]}.{obj.__qualname__}"
    return str(obj).split(":")[-1]


def main():
    """Print all environments registered in the guguji_locomotion extension."""
    table = PrettyTable(["No.", "Task Name", "Entry Point", "Config"])
    table.title = "Guguji Isaac Lab Environments"
    table.align["Task Name"] = "l"
    table.align["Entry Point"] = "l"
    table.align["Config"] = "l"
    table.max_width["Entry Point"] = 40
    table.max_width["Config"] = 40

    index = 0
    for task_spec in gym.registry.values():
        if "Guguji-" in task_spec.id:
            cfg = task_spec.kwargs.get("env_cfg_entry_point", "")
            table.add_row([index + 1, task_spec.id, _fmt(task_spec.entry_point), _fmt(cfg)])
            index += 1

    print(table)


if __name__ == "__main__":
    try:
        # run the main function
        main()
    except Exception as e:
        raise e
    finally:
        # close the app
        simulation_app.close()
