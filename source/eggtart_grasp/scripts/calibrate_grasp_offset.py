#!/usr/bin/env python3
"""Interactive grasp-point calibration tool.

Visualizes the current grasp point, jaw centroids, and end-effector frame,
allowing keyboard adjustment of the offset in an intuitive coordinate frame.

Usage:
    ./isaaclab.sh -p source/eggtart_grasp/scripts/calibrate_grasp_offset.py

Keyboard:
    W/S: Move grasp point forward/backward (base -Y/+Y)
    A/D: Move grasp point left/right (base -X/+X)
    Q/E: Move grasp point down/up (base -Z/+Z)
    O/C: Open/Close gripper
    R: Reset offset to default
    P: Print current offset
    ESC: Exit and print final offset
"""

import argparse
import numpy as np
import torch
from pathlib import Path

import omni
import carb

from isaaclab.app import AppLauncher

# Create argparser
parser = argparse.ArgumentParser(description="Grasp point calibration tool")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments")
parser.add_argument("--compute_auto", action="store_true",
                    help="Compute grasp offset from jaw geometry and exit")
# Add AppLauncher arguments (--headless, etc.)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Launch the simulator
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Now import isaac modules
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG, RED_ARROW_X_MARKER_CFG
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_apply, subtract_frame_transforms
from scipy.spatial.transform import Rotation as R

# Import the robot config
import eggtart_grasp.tasks.mobile_grasp  # noqa: F401
from eggtart_grasp.assets import EGGTART_CFG, EGGTART_EE_GRASP_OFFSET


def compute_mesh_centroid(stl_path: Path) -> np.ndarray:
    """Load STL and compute centroid in mesh-local coordinates."""
    import struct

    with open(stl_path, 'rb') as f:
        header = f.read(80)
        num_triangles = struct.unpack('<I', f.read(4))[0]

        vertices = []
        for _ in range(num_triangles):
            # Normal (skip)
            f.read(12)
            # Three vertices
            for __ in range(3):
                v = struct.unpack('<fff', f.read(12))
                vertices.append(v)
            # Attribute byte count (skip)
            f.read(2)

    vertices = np.array(vertices)
    centroid = vertices.mean(axis=0)
    return centroid


def compute_grasp_offset_from_geometry():
    """Compute grasp offset by loading jaw meshes and finding midpoint."""
    # Script is in source/eggtart_grasp/scripts/, need source/eggtart_grasp/eggtart_grasp/assets/urdf
    script_dir = Path(__file__).resolve().parent
    urdf_dir = script_dir.parent / "eggtart_grasp" / "assets" / "urdf"

    # Fixed jaw on link_005: part_034
    fixed_jaw_stl = urdf_dir / "meshes" / "part_034_solid_034.stl"
    # Moving jaw on end_effector: part_035
    moving_jaw_stl = urdf_dir / "meshes" / "part_035_solid_035.stl"

    print("\n=== Computing grasp offset from jaw geometry ===")

    # Mesh centroids in their mesh-local frames (mm in STL)
    fixed_centroid_mesh = compute_mesh_centroid(fixed_jaw_stl) * 0.001  # mm -> m
    moving_centroid_mesh = compute_mesh_centroid(moving_jaw_stl) * 0.001

    print(f"Fixed jaw (part_034) centroid in mesh frame: {fixed_centroid_mesh}")
    print(f"Moving jaw (part_035) centroid in mesh frame: {moving_centroid_mesh}")

    # URDF visual origins (these transform mesh -> link frame)
    # link_005: <origin xyz="0.33609379 0.47285302 -0.21011169" rpy="1.57079633 0 0"/>
    fixed_origin = np.array([0.33609379, 0.47285302, -0.21011169])
    fixed_rpy = np.array([1.57079633, 0.0, 0.0])

    # end_effector: <origin xyz="0.31597274 0.46294983 -0.18567585" rpy="1.57079633 0 0"/>
    moving_origin = np.array([0.31597274, 0.46294983, -0.18567585])
    moving_rpy = np.array([1.57079633, 0.0, 0.0])

    # Transform mesh centroids to link frames
    fixed_rot = R.from_euler('xyz', fixed_rpy).as_matrix()
    moving_rot = R.from_euler('xyz', moving_rpy).as_matrix()

    fixed_centroid_link005 = fixed_origin + fixed_rot @ fixed_centroid_mesh
    moving_centroid_ee = moving_origin + moving_rot @ moving_centroid_mesh

    print(f"\nFixed jaw centroid in link_005 frame: {fixed_centroid_link005}")
    print(f"Moving jaw centroid in end_effector frame: {moving_centroid_ee}")

    # The end_effector joint connects link_005 -> end_effector
    # Joint origin in link_005 frame: varies with joint angle
    # When closed (joint = -0.2), the moving jaw should be near the fixed jaw
    # We need to express the grasp point (midpoint) in the end_effector frame

    # Simplified approach: assume at nominal closure the jaws meet
    # The grasp point should be the midpoint in the link_005 frame,
    # then we transform it to the end_effector frame

    print("\n=== Approach: Use simulation to get actual transforms ===")
    print("We need to run the simulation to get the actual end_effector body pose")
    print("relative to link_005 at various gripper positions.")
    print("This script will do that interactively.\n")

    return None  # Signal that we need simulation


def design_scene() -> tuple[InteractiveSceneCfg, SceneEntityCfg, SceneEntityCfg, SceneEntityCfg]:
    """Design the scene with the robot."""

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0, replicate_physics=False)

    # Add ground plane (using AssetBaseCfg wrapper)
    from isaaclab.assets import AssetBaseCfg
    scene_cfg.terrain = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # Add robot
    scene_cfg.robot = EGGTART_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    scene_cfg.robot.init_state.pos = (0.0, 0.0, 0.0)

    robot_cfg = SceneEntityCfg("robot")
    ee_cfg = SceneEntityCfg("robot", body_names=["end_effector"])
    link005_cfg = SceneEntityCfg("robot", body_names=["link_005"])

    return scene_cfg, robot_cfg, ee_cfg, link005_cfg


class GraspPointCalibrator:
    """Interactive calibration tool with keyboard control and visualization."""

    def __init__(self, scene: InteractiveScene, robot_cfg, ee_cfg, link005_cfg):
        self.scene = scene
        self.robot: Articulation = scene["robot"]
        self.robot_cfg = robot_cfg
        self.ee_cfg = ee_cfg
        self.link005_cfg = link005_cfg

        # Current offset being tuned (in end_effector frame)
        self.offset = torch.tensor(EGGTART_EE_GRASP_OFFSET, device=self.robot.device, dtype=torch.float32)
        self.default_offset = self.offset.clone()

        # Gripper state
        self.gripper_pos = 1.0  # Open

        # Adjustment step size (meters)
        self.step_size = 0.001  # 1mm

        # Keyboard state
        self._setup_keyboard()

        # Markers
        self._setup_markers()

        # Load mesh geometry for jaw centroids
        self._load_jaw_geometry()

        print("\n=== Grasp Point Calibrator Ready ===")
        print(f"Current offset: {self.offset.cpu().numpy()}")
        print("\nKeyboard controls:")
        print("  W/S: Forward/Backward (base -Y/+Y)")
        print("  A/D: Left/Right (base -X/+X)")
        print("  Q/E: Down/Up (base -Z/+Z)")
        print("  O/C: Open/Close gripper")
        print("  R: Reset to default")
        print("  P: Print current offset")
        print("  ESC: Exit\n")

    def _setup_keyboard(self):
        """Subscribe to keyboard events."""
        self._appwindow = omni.appwindow.get_default_app_window()
        self._input = carb.input.acquire_input_interface()
        self._keyboard = self._appwindow.get_keyboard()

        import weakref
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

        self._key_pressed = {}
        self._exit_requested = False

    def _setup_markers(self):
        """Create visualization markers."""
        # Grasp point marker (red sphere)
        from isaaclab.markers.config import SPHERE_MARKER_CFG
        grasp_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/grasp_point")
        grasp_cfg.markers["sphere"].radius = 0.01
        grasp_cfg.markers["sphere"].visual_material.diffuse_color = (1.0, 0.0, 0.0)
        self.grasp_marker = VisualizationMarkers(grasp_cfg)

        # End-effector frame marker
        frame_cfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/ee_frame")
        frame_cfg.markers["frame"].scale = (0.05, 0.05, 0.05)
        self.ee_frame_marker = VisualizationMarkers(frame_cfg)

        # Jaw centroid markers (smaller blue spheres)
        from isaaclab.markers.config import SPHERE_MARKER_CFG
        fixed_jaw_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/fixed_jaw")
        fixed_jaw_cfg.markers["sphere"].radius = 0.008
        fixed_jaw_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 0.0, 1.0)
        self.fixed_jaw_marker = VisualizationMarkers(fixed_jaw_cfg)

        moving_jaw_cfg = SPHERE_MARKER_CFG.replace(prim_path="/Visuals/moving_jaw")
        moving_jaw_cfg.markers["sphere"].radius = 0.008
        moving_jaw_cfg.markers["sphere"].visual_material.diffuse_color = (0.0, 1.0, 1.0)
        self.moving_jaw_marker = VisualizationMarkers(moving_jaw_cfg)

    def _load_jaw_geometry(self):
        """Load jaw mesh centroids."""
        # Script is in source/eggtart_grasp/scripts/, need source/eggtart_grasp/eggtart_grasp/assets/urdf
        script_dir = Path(__file__).resolve().parent
        urdf_dir = script_dir.parent / "eggtart_grasp" / "assets" / "urdf"

        fixed_jaw_stl = urdf_dir / "meshes" / "part_034_solid_034.stl"
        moving_jaw_stl = urdf_dir / "meshes" / "part_035_solid_035.stl"

        # Centroids in mesh frames
        fixed_c_mesh = compute_mesh_centroid(fixed_jaw_stl) * 0.001
        moving_c_mesh = compute_mesh_centroid(moving_jaw_stl) * 0.001

        # URDF visual origins
        fixed_origin = np.array([0.33609379, 0.47285302, -0.21011169])
        fixed_rpy = np.array([1.57079633, 0.0, 0.0])
        moving_origin = np.array([0.31597274, 0.46294983, -0.18567585])
        moving_rpy = np.array([1.57079633, 0.0, 0.0])

        # Transform to link frames
        fixed_rot = R.from_euler('xyz', fixed_rpy).as_matrix()
        moving_rot = R.from_euler('xyz', moving_rpy).as_matrix()

        self.fixed_jaw_offset_link005 = torch.tensor(
            fixed_origin + fixed_rot @ fixed_c_mesh,
            device=self.robot.device, dtype=torch.float32
        )
        self.moving_jaw_offset_ee = torch.tensor(
            moving_origin + moving_rot @ moving_c_mesh,
            device=self.robot.device, dtype=torch.float32
        )

        print(f"Fixed jaw offset (link_005 frame): {self.fixed_jaw_offset_link005.cpu().numpy()}")
        print(f"Moving jaw offset (end_effector frame): {self.moving_jaw_offset_ee.cpu().numpy()}")

    def _on_keyboard_event(self, event, *args, **kwargs):
        """Handle keyboard events."""
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._key_pressed[event.input.name] = True

            # Immediate actions
            if event.input.name == "ESCAPE":
                self._exit_requested = True
            elif event.input.name == "R":
                self.offset = self.default_offset.clone()
                print(f"Reset offset to default: {self.offset.cpu().numpy()}")
            elif event.input.name == "P":
                print(f"Current offset: {self.offset.cpu().numpy()}")
            elif event.input.name == "O":
                self.gripper_pos = 1.0  # Open
                print("Gripper: OPEN")
            elif event.input.name == "C":
                self.gripper_pos = -0.2  # Closed
                print("Gripper: CLOSED")

        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._key_pressed[event.input.name] = False

    def update(self):
        """Update robot pose and handle continuous key presses."""
        # Handle continuous adjustments
        if self._key_pressed.get("W", False):
            self._adjust_offset_base_frame(dy=-self.step_size)
        if self._key_pressed.get("S", False):
            self._adjust_offset_base_frame(dy=self.step_size)
        if self._key_pressed.get("A", False):
            self._adjust_offset_base_frame(dx=-self.step_size)
        if self._key_pressed.get("D", False):
            self._adjust_offset_base_frame(dx=self.step_size)
        if self._key_pressed.get("Q", False):
            self._adjust_offset_base_frame(dz=-self.step_size)
        if self._key_pressed.get("E", False):
            self._adjust_offset_base_frame(dz=self.step_size)

        # Set gripper position
        gripper_idx = self.robot.find_joints("end_effector_joint")[0][0]
        self.robot.set_joint_position_target(
            torch.tensor([[self.gripper_pos]], device=self.robot.device),
            joint_ids=[gripper_idx]
        )

        # Update visualization
        self._update_markers()

        return not self._exit_requested

    def _adjust_offset_base_frame(self, dx=0.0, dy=0.0, dz=0.0):
        """Adjust offset in base_link frame and convert to end_effector frame."""
        # Get current end_effector orientation in world
        ee_pos_w = self.robot.data.body_pos_w[0, self.ee_cfg.body_ids[0]]
        ee_quat_w = self.robot.data.body_quat_w[0, self.ee_cfg.body_ids[0]]

        # Get base orientation in world
        base_quat_w = self.robot.data.root_quat_w[0]

        # Adjustment vector in base frame
        delta_base = torch.tensor([dx, dy, dz], device=self.robot.device, dtype=torch.float32)

        # Transform to world frame
        delta_w = quat_apply(base_quat_w, delta_base)

        # Transform to end_effector frame
        ee_quat_w_inv = torch.tensor(
            [ee_quat_w[0], -ee_quat_w[1], -ee_quat_w[2], -ee_quat_w[3]],
            device=self.robot.device
        )
        delta_ee = quat_apply(ee_quat_w_inv, delta_w)

        # Update offset
        self.offset = self.offset + delta_ee
        print(f"Offset adjusted: {self.offset.cpu().numpy()}")

    def _update_markers(self):
        """Update all visualization markers."""
        # Get body indices
        ee_body_idx = self.ee_cfg.body_ids[0] if isinstance(self.ee_cfg.body_ids, list) else 0
        link005_body_idx = self.link005_cfg.body_ids[0] if isinstance(self.link005_cfg.body_ids, list) else 0

        # Grasp point in world frame
        ee_pos_w = self.robot.data.body_pos_w[0, ee_body_idx]
        ee_quat_w = self.robot.data.body_quat_w[0, ee_body_idx]
        grasp_pos_w = ee_pos_w + quat_apply(ee_quat_w, self.offset)

        # End-effector frame
        self.ee_frame_marker.visualize(
            ee_pos_w.unsqueeze(0),
            ee_quat_w.unsqueeze(0)
        )

        # Grasp point
        identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=self.robot.device)
        self.grasp_marker.visualize(
            grasp_pos_w.unsqueeze(0),
            identity_quat
        )

        # Fixed jaw centroid (in link_005 frame)
        link005_pos_w = self.robot.data.body_pos_w[0, link005_body_idx]
        link005_quat_w = self.robot.data.body_quat_w[0, link005_body_idx]
        fixed_jaw_w = link005_pos_w + quat_apply(link005_quat_w, self.fixed_jaw_offset_link005)

        self.fixed_jaw_marker.visualize(
            fixed_jaw_w.unsqueeze(0),
            identity_quat
        )

        # Moving jaw centroid (in end_effector frame)
        moving_jaw_w = ee_pos_w + quat_apply(ee_quat_w, self.moving_jaw_offset_ee)

        self.moving_jaw_marker.visualize(
            moving_jaw_w.unsqueeze(0),
            identity_quat
        )

    def get_final_offset(self):
        """Return the final calibrated offset."""
        return tuple(self.offset.cpu().numpy())

    def __del__(self):
        """Clean up keyboard subscription."""
        if hasattr(self, '_keyboard_sub') and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)


def main():
    """Run the calibration tool."""

    # Compute from geometry if requested
    if args_cli.compute_auto:
        result = compute_grasp_offset_from_geometry()
        if result is not None:
            print(f"\nComputed offset: {result}")
            print("Copy to eggtart.py:")
            print(f"EGGTART_EE_GRASP_OFFSET = {result}")
        simulation_app.close()
        return

    # Create simulation context
    sim_cfg = sim_utils.SimulationCfg(dt=1/120, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view(eye=[2.5, 2.5, 2.5], target=[0.0, 0.0, 0.0])

    # Create scene
    scene_cfg, robot_cfg, ee_cfg, link005_cfg = design_scene()
    scene = InteractiveScene(scene_cfg)

    # Play the simulator to initialize everything
    sim.reset()

    # Set nominal pose
    robot: Articulation = scene["robot"]
    from eggtart_grasp.assets import EGGTART_NOMINAL_JOINT_POS

    joint_pos = robot.data.default_joint_pos.clone()
    for name_pattern, pos in EGGTART_NOMINAL_JOINT_POS.items():
        joint_indices, _ = robot.find_joints(name_pattern)
        if len(joint_indices) > 0:
            joint_pos[0, joint_indices] = pos

    robot.write_joint_state_to_sim(joint_pos, robot.data.default_joint_vel)

    # Reset scene
    scene.reset()

    # Create calibrator
    calibrator = GraspPointCalibrator(scene, robot_cfg, ee_cfg, link005_cfg)

    # Simulation loop
    sim_dt = scene.sim.get_physics_dt()
    sim_time = 0.0
    count = 0

    while simulation_app.is_running():
        # Step simulation
        if count % 4 == 0:  # Match decimation
            if not calibrator.update():
                break

        # Perform step
        scene.sim.step()
        scene.update(sim_dt)

        count += 1
        sim_time += sim_dt

    # Print final offset
    final_offset = calibrator.get_final_offset()
    print("\n" + "="*60)
    print("FINAL CALIBRATED OFFSET")
    print("="*60)
    print(f"EGGTART_EE_GRASP_OFFSET = {final_offset}")
    print("="*60)
    print("\nCopy the line above to source/eggtart_grasp/eggtart_grasp/assets/eggtart.py\n")

    # Cleanup
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        carb.log_error(f"Error in calibration tool: {e}")
        import traceback
        traceback.print_exc()
        simulation_app.close()
