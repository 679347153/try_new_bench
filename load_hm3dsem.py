import os
import sys

# Try to import habitat_sim, handle failure gracefully
try:
    import habitat_sim
    import magnum as mn
except ImportError:
    print("Error: 'habitat_sim' module not found.")
    print("Please ensure habitat-sim is installed in your environment.")
    print("You can install it using conda:")
    print("  conda install -c aihabitat -c conda-forge habitat-sim headless")
    print("Or via pip (if supported on your platform):")
    print("  pip install habitat-sim")
    # We exit here because the rest of the script depends on habitat_sim
    sys.exit(1)

import cv2
import numpy as np

def make_cfg(settings):
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.gpu_device_id = 0
    if "scene_dataset_config_file" in settings and settings["scene_dataset_config_file"]:
        sim_cfg.scene_dataset_config_file = settings["scene_dataset_config_file"]
    sim_cfg.scene_id = settings["scene"]
    sim_cfg.enable_physics = True

    # Note: all sensors must have the same resolution
    sensor_specs = []

    # RGB Sensor
    rgb_sensor_spec = habitat_sim.CameraSensorSpec()
    rgb_sensor_spec.uuid = "color_sensor"
    rgb_sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    rgb_sensor_spec.resolution = [settings["height"], settings["width"]]
    rgb_sensor_spec.position = [0.0, 1.5, 0.0]
    rgb_sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_specs.append(rgb_sensor_spec)

    # Depth Sensor
    depth_sensor_spec = habitat_sim.CameraSensorSpec()
    depth_sensor_spec.uuid = "depth_sensor"
    depth_sensor_spec.sensor_type = habitat_sim.SensorType.DEPTH
    depth_sensor_spec.resolution = [settings["height"], settings["width"]]
    depth_sensor_spec.position = [0.0, 1.5, 0.0]
    depth_sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_specs.append(depth_sensor_spec)

    # Semantic Sensor
    semantic_sensor_spec = habitat_sim.CameraSensorSpec()
    semantic_sensor_spec.uuid = "semantic_sensor"
    semantic_sensor_spec.sensor_type = habitat_sim.SensorType.SEMANTIC
    semantic_sensor_spec.resolution = [settings["height"], settings["width"]]
    semantic_sensor_spec.position = [0.0, 1.5, 0.0]
    semantic_sensor_spec.orientation = [0.0, 0.0, 0.0]
    sensor_specs.append(semantic_sensor_spec)

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = sensor_specs
    agent_cfg.action_space = {
        "move_forward": habitat_sim.agent.ActionSpec(
            "move_forward", habitat_sim.agent.ActuationSpec(amount=0.25)
        ),
        "turn_left": habitat_sim.agent.ActionSpec(
            "turn_left", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
        "turn_right": habitat_sim.agent.ActionSpec(
            "turn_right", habitat_sim.agent.ActuationSpec(amount=30.0)
        ),
    }

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def main():
    # Attempt to locate the scene file automatically
    # Start looking from the current directory's 'data' folder
    base_data_path = "data"
    scene_path = None
    dataset_config_path = None

    # Search for Dataset Config
    potential_configs = [
        # Try hm3d_new first
        os.path.join("data", "scene_datasets", "hm3d_new", "hm3d_new.scene_dataset_config.json"),
        os.path.join("data", "hm3d_annotated_basis.scene_dataset_config.json")
    ]

    for cfg in potential_configs:
        if os.path.exists(cfg):
            dataset_config_path = os.path.abspath(cfg)
            break
            
    if dataset_config_path:
        print(f"Found dataset config: {dataset_config_path}")
        config_dir = os.path.dirname(dataset_config_path)
        # Search for a scene relative to this config
        
        # We need a robust walk
        for root, dirs, files in os.walk(config_dir):
            # Sort to ensure deterministic behavior
            dirs.sort()
            files.sort()
            
            # Prioritize basis.glb over scene_instance.json for better semantic loading support
            # (Scene instances sometimes don't link semantics correctly in all dataset versions)
            basis_file = next((f for f in files if f.endswith(".basis.glb")), None)
            instance_file = next((f for f in files if f.endswith(".scene_instance.json")), None)
            
            # Prefer basis file
            target_file = basis_file if basis_file else instance_file
            
            if target_file:
                # Check if corresponding semantic file exists (strict check)
                scene_name = target_file.split('.')[0]
                semantic_file = f"{scene_name}.semantic.glb"
                if semantic_file in files:
                    print(f"Verified semantic file exists: {semantic_file}")
                else:
                    print(f"Warning: Semantic file {semantic_file} not found for {target_file}")
                
                abs_path = os.path.join(root, target_file)
                # Compute relative path
                rel_path = os.path.relpath(abs_path, config_dir)
                scene_path = rel_path.replace("\\", "/")
                print(f"Selected scene handle: {scene_path}")
                break
        
        if not scene_path:
             print("Warning: Dataset config found but no scene file found relative to it.")
    
    # Fallback to absolute path search if no config-based scene found
    if not scene_path:
        # Try finding the specific file we know exists
        potential_path = os.path.join("data", "scene_datasets", "hm3d_new", "hm3d", "val", "00800-TEEsavR23oF", "TEEsavR23oF.basis.glb")
        if os.path.exists(potential_path):
            scene_path = potential_path
        
        if not scene_path:
            # Fallback to recursively searching for any .basis.glb
            for root, dirs, files in os.walk(base_data_path):
                for file in files:
                    if file.endswith(".basis.glb"):
                        scene_path = os.path.join(root, file)
                        break
                if scene_path:
                    break

    if not scene_path:
        print(f"Error: Could not find any HM3D scene file (.basis.glb) in 'data/' directory.")
        print(f"Current working directory: {os.getcwd()}")
        return

    settings = {
        "width": 640,
        "height": 480,
        "scene": scene_path,
        "scene_dataset_config_file": dataset_config_path
    }

    print(f"Loading scene: {scene_path}")
    
    try:
        cfg = make_cfg(settings)
        sim = habitat_sim.Simulator(cfg)
    except Exception as e:
        print(f"Failed to create simulator: {e}")
        import traceback
        traceback.print_exc()
        return

    print("Simulator loaded successfully!")

    # Initialize the agent
    agent = sim.initialize_agent(0)
    
    # Check Semantic Scene 
    semantic_scene = sim.semantic_scene
    if semantic_scene and len(semantic_scene.objects) > 0:
        print(f"Semantic scene loaded successfully.")
        print(f"Number of objects: {len(semantic_scene.objects)}")
    else:
        print("\nWarning: Semantic scene not loaded or empty.")
        print("Ensure 'TEEsavR23oF.semantic.glb' is in the same directory as 'TEEsavR23oF.basis.glb'.")

    # Navigate and Capture
    print("\nSimulating agent navigation...")
    
    # Move forward a bit to ensure we are inside
    sim.step("move_forward")
    sim.step("move_forward")
    sim.step("turn_right")

    # Get Observations
    observations = sim.get_sensor_observations()
    
    # Save RGB Image
    if "color_sensor" in observations:
        rgb = observations["color_sensor"]
        if rgb.shape[2] == 4:
            rgb = rgb[..., :3] # Remove alpha
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        output_img = "hm3d_observation.png"
        cv2.imwrite(output_img, bgr)
        print(f"Saved observation to {output_img}")

    # Save Semantic Image
    if "semantic_sensor" in observations:
        semantic = observations["semantic_sensor"]
        # Normalize for visualization
        # Just use raw ID but scaled
        semantic_vis = (semantic % 255).astype(np.uint8)
        output_sem = "hm3d_semantic.png"
        cv2.imwrite(output_sem, semantic_vis)
        print(f"Saved semantic observation to {output_sem}")

    sim.close()

if __name__ == "__main__":
    main()
