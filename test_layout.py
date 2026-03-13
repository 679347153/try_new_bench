#!/usr/bin/env python3
"""
交互式布局编辑器。

用途:
1. 在固定场景中可视化添加物体。
2. 选中物体后微调位置和朝向。
3. 将当前布局保存为 configs 下的 JSON 文件。

示例:
	python test_layout.py 00800-TEEsavR23oF --layout scene_objects_v2.json
"""

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

try:
	import habitat_sim
	import habitat_sim.utils.common as utils
except ImportError:
	print("错误: 未找到 habitat_sim，请在对应环境中运行")
	sys.exit(1)


SCENES_DIR = "data/scene_datasets/hm3d_new"
SCENE_DATASET_CONFIG = "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
OBJECTS_DIR = "data/objects"

AVAILABLE_SCENES = [
	"00800-TEEsavR23oF",
	"00802-wcojb4TFT35",
	"00813-svBbv1Pavdk",
	"00814-p53SfW6mjZe",
	"00820-mL8ThkuaVTM",
	"00824-Dd4bFSTQ8gi",
	"00829-QaLdnwvtxbs",
	"00832-qyAac8rV8Zk",
	"00835-q3zU7Yy5E5s",
	"00839-zt1RVoi7PcG",
]

DISPLAY_WIDTH = 1280
DISPLAY_HEIGHT = 720
CAMERA_MOVE_SPEED = 0.15
OBJECT_MOVE_SPEED = 0.08
ROTATE_SPEED = 2.5
OBJECT_ROTATE_SPEED = 5.0
PITCH_LIMIT = 85.0
SPAWN_DISTANCE = 1.5
CAMERA_HEIGHT = 0.88


HELP_TEXT = [
	"=== 相机 ===",
	"W/S A/D E/C    前后 左右 上下移动",
	"I/K J/L        俯仰 / 左右旋转视角",
	"N/P            切换场景",
	"R              相机复位到可导航点",
	"=== 布局 ===",
	"[/]            切换布局文件",
	"Z/X            切换待添加模板",
	"V              添加当前模板到相机前方",
	"</>            切换选中物体",
	"T/G            选中物体前后移动",
	"F/Y            选中物体左右移动",
	"U/O            选中物体上下移动",
	"1/2            选中物体左/右旋转",
	"B              删除选中物体",
	"M              保存当前布局",
	"H              显示/隐藏帮助",
	"ESC/Q          退出",
]


def get_scene_dir(scene_name):
	return os.path.join(SCENES_DIR, "hm3d/val", scene_name)


def get_scene_id(scene_name):
	scene_dir = get_scene_dir(scene_name)
	if not os.path.isdir(scene_dir):
		raise FileNotFoundError(f"场景目录不存在: {scene_dir}")
	parts = scene_name.split("-", 1)
	return parts[1] if len(parts) > 1 else scene_name


def get_stage_glb_path(scene_name):
	scene_id = get_scene_id(scene_name)
	return os.path.join(get_scene_dir(scene_name), f"{scene_id}.basis.glb")


def get_configs_dir(scene_name):
	return os.path.join(get_scene_dir(scene_name), "configs")


def ensure_configs_dir(scene_name):
	os.makedirs(get_configs_dir(scene_name), exist_ok=True)


def list_layout_files(scene_name):
	config_dir = get_configs_dir(scene_name)
	if not os.path.isdir(config_dir):
		return []

	files = []
	for name in sorted(os.listdir(config_dir)):
		if name.lower().endswith(".json"):
			files.append(os.path.join(config_dir, name))

	files.sort(key=lambda p: (0 if os.path.basename(p) == "scene_objects.json" else 1, os.path.basename(p)))
	return files


def get_default_layout_path(scene_name, layout_name):
	ensure_configs_dir(scene_name)
	if not layout_name.lower().endswith(".json"):
		layout_name = f"{layout_name}.json"
	return os.path.join(get_configs_dir(scene_name), layout_name)


def scan_object_templates():
	if not os.path.isdir(OBJECTS_DIR):
		raise FileNotFoundError(f"物体模板目录不存在: {OBJECTS_DIR}")

	templates = []
	for name in sorted(os.listdir(OBJECTS_DIR)):
		if not name.endswith(".object_config.json"):
			continue
		templates.append(name[:-len(".object_config.json")])

	if not templates:
		raise RuntimeError("未在 data/objects 下发现任何 .object_config.json 模板")
	return templates


def make_sim_cfg(scene_id):
	sim_cfg = habitat_sim.SimulatorConfiguration()
	sim_cfg.scene_dataset_config_file = SCENE_DATASET_CONFIG
	sim_cfg.scene_id = scene_id
	sim_cfg.enable_physics = False
	sim_cfg.gpu_device_id = 0

	sensor = habitat_sim.CameraSensorSpec()
	sensor.uuid = "color"
	sensor.sensor_type = habitat_sim.SensorType.COLOR
	sensor.resolution = [DISPLAY_HEIGHT, DISPLAY_WIDTH]
	sensor.hfov = 90
	sensor.position = [0.0, CAMERA_HEIGHT, 0.0]

	agent_cfg = habitat_sim.agent.AgentConfiguration()
	agent_cfg.sensor_specifications = [sensor]
	agent_cfg.height = CAMERA_HEIGHT
	agent_cfg.radius = 0.18

	return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def rotation_from_euler(yaw_deg, pitch_deg):
	qy = utils.quat_from_angle_axis(math.radians(yaw_deg), np.array([0, 1, 0]))
	qx = utils.quat_from_angle_axis(math.radians(pitch_deg), np.array([1, 0, 0]))
	return qy * qx


def yaw_to_quat(yaw_deg):
	return utils.quat_from_angle_axis(math.radians(yaw_deg), np.array([0, 1, 0]))


def overlay_text(img, lines, x, y, color=(255, 255, 80), scale=0.52, thickness=1):
	line_height = 21
	for line in lines:
		cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
					scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
		cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
					scale, color, thickness, cv2.LINE_AA)
		y += line_height


def prettify_model_name(model_id):
	return model_id.replace("_", " ").title()


def resolve_template_handle(template_mgr, model_id):
	candidates = template_mgr.get_template_handles(model_id)
	if candidates:
		return candidates[0]

	if not model_id.endswith(".object_config.json"):
		candidates = template_mgr.get_template_handles(f"{model_id}.object_config.json")
		if candidates:
			return candidates[0]

	return None


def safe_remove_object(rom, item):
	handle = item.get("handle")
	if handle:
		try:
			rom.remove_object_by_handle(handle)
			return
		except Exception:
			pass

	object_id = item.get("object_id")
	if object_id is not None:
		try:
			rom.remove_object_by_id(object_id)
		except Exception:
			pass


def clear_editor_objects(sim, editor_items):
	rom = sim.get_rigid_object_manager()
	for item in list(editor_items):
		safe_remove_object(rom, item)
	editor_items.clear()


def create_editor_item(sim, model_id, position, yaw_deg=0.0):
	rom = sim.get_rigid_object_manager()
	template_mgr = sim.get_object_template_manager()
	template_handle = resolve_template_handle(template_mgr, model_id)
	if template_handle is None:
		raise RuntimeError(f"找不到模板: {model_id}")

	obj = rom.add_object_by_template_handle(template_handle)
	if obj is None:
		raise RuntimeError(f"无法实例化模板: {template_handle}")

	obj.translation = np.array(position, dtype=np.float32)
	obj.rotation = yaw_to_quat(yaw_deg)
	obj.motion_type = habitat_sim.physics.MotionType.STATIC

	return {
		"object": obj,
		"object_id": getattr(obj, "object_id", None),
		"handle": getattr(obj, "handle", None),
		"model_id": model_id,
		"name": prettify_model_name(model_id),
		"yaw_deg": float(yaw_deg),
	}


def extract_yaw_deg(rotation_value):
	if isinstance(rotation_value, list) and len(rotation_value) == 3:
		return float(rotation_value[1])
	return 0.0


def load_layout_into_editor(sim, layout_path, editor_items):
	clear_editor_objects(sim, editor_items)

	if not os.path.isfile(layout_path):
		return 0, 0

	with open(layout_path, "r", encoding="utf-8") as f:
		data = json.load(f)

	loaded = 0
	skipped = 0
	for obj_cfg in data.get("objects", []):
		model_id = obj_cfg.get("model_id") or obj_cfg.get("template_name")
		position = obj_cfg.get("position") or obj_cfg.get("translation")
		if not model_id or not isinstance(position, list) or len(position) != 3:
			skipped += 1
			continue

		yaw_deg = extract_yaw_deg(obj_cfg.get("rotation"))
		try:
			item = create_editor_item(sim, model_id, position, yaw_deg)
			editor_items.append(item)
			loaded += 1
		except Exception as exc:
			print(f"      [跳过] {model_id}: {exc}")
			skipped += 1

	return loaded, skipped


def save_layout(scene_name, layout_path, editor_items):
	ensure_configs_dir(scene_name)
	payload = {
		"scene": get_stage_glb_path(scene_name),
		"timestamp": time.time(),
		"objects": [],
	}

	for index, item in enumerate(editor_items):
		pos = item["object"].translation
		payload["objects"].append({
			"id": index,
			"name": item["name"],
			"model_id": item["model_id"],
			"position": [round(float(pos[0]), 4), round(float(pos[1]), 4), round(float(pos[2]), 4)],
			"rotation": [0.0, round(float(item["yaw_deg"]), 4), 0.0],
		})

	with open(layout_path, "w", encoding="utf-8") as f:
		json.dump(payload, f, ensure_ascii=False, indent=2)


def get_camera_vectors(yaw_deg):
	yaw_rad = math.radians(yaw_deg)
	forward = np.array([-math.sin(yaw_rad), 0.0, -math.cos(yaw_rad)], dtype=np.float32)
	right = np.array([math.cos(yaw_rad), 0.0, -math.sin(yaw_rad)], dtype=np.float32)
	return forward, right


def pick_spawn_position(camera_pos, yaw_deg):
	forward, _ = get_camera_vectors(yaw_deg)
	spawn = camera_pos.copy() + forward * SPAWN_DISTANCE
	spawn[1] = max(0.0, float(camera_pos[1] - CAMERA_HEIGHT + 0.1))
	return spawn


def cycle_index(current_idx, size, step):
	if size <= 0:
		return 0
	return (current_idx + step) % size


def find_layout_index(layout_files, target_path):
	for idx, path in enumerate(layout_files):
		if os.path.normcase(path) == os.path.normcase(target_path):
			return idx
	return 0


def refresh_layout_files(scene_name, current_layout_path):
	files = list_layout_files(scene_name)
	if current_layout_path and os.path.isfile(current_layout_path):
		normalized = {os.path.normcase(path) for path in files}
		if os.path.normcase(current_layout_path) not in normalized:
			files.append(current_layout_path)
			files.sort(key=lambda p: os.path.basename(p))
	return files


def main():
	if not os.path.isdir("data/scene_datasets"):
		print("请在项目根目录下运行此脚本")
		sys.exit(1)

	parser = argparse.ArgumentParser(description="交互式布局编辑器")
	parser.add_argument("scene", nargs="?", default="00800-TEEsavR23oF", help="场景名，例如 00800-TEEsavR23oF")
	parser.add_argument("--layout", default="scene_objects.json", help="布局文件名或完整路径")
	args = parser.parse_args()

	templates = scan_object_templates()

	scene_idx = 0
	for i, scene_name in enumerate(AVAILABLE_SCENES):
		if args.scene in scene_name:
			scene_idx = i
			break

	if os.path.isabs(args.layout):
		current_layout_path = args.layout
	else:
		current_layout_path = get_default_layout_path(AVAILABLE_SCENES[scene_idx], args.layout)

	win = "布局编辑器  [H]帮助  [M]保存  [ESC/Q]退出"
	cv2.namedWindow(win, cv2.WINDOW_NORMAL)
	cv2.resizeWindow(win, DISPLAY_WIDTH, DISPLAY_HEIGHT)

	sim = None
	scene_layout_files = []
	layout_idx = 0
	editor_items = []
	selected_idx = -1
	template_idx = 0
	show_help = True
	dirty = False
	status_text = ""
	camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
	yaw = 0.0
	pitch = 0.0

	def set_status(message):
		nonlocal status_text
		status_text = message
		print(message)

	def select_after_load():
		nonlocal selected_idx
		selected_idx = 0 if editor_items else -1

	def update_layout_catalog(scene_name):
		nonlocal scene_layout_files, layout_idx
		scene_layout_files = refresh_layout_files(scene_name, current_layout_path)
		layout_idx = find_layout_index(scene_layout_files, current_layout_path) if scene_layout_files else 0

	def load_scene(idx):
		nonlocal sim, camera_pos, yaw, pitch, dirty, current_layout_path
		scene_name = AVAILABLE_SCENES[idx]
		set_status(f"\n>>> 加载场景: {scene_name}")
		scene_id = get_scene_id(scene_name)
		cfg = make_sim_cfg(scene_id)

		if sim is not None:
			sim.close()
		sim = habitat_sim.Simulator(cfg)

		if not os.path.isabs(args.layout):
			current_layout_path = get_default_layout_path(scene_name, os.path.basename(current_layout_path))
		update_layout_catalog(scene_name)

		loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
		select_after_load()
		camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
		yaw = 0.0
		pitch = 0.0
		dirty = False
		set_status(f"    布局: {os.path.basename(current_layout_path)}  (加载 {loaded} / 跳过 {skipped})")

	load_scene(scene_idx)

	while True:
		key = cv2.waitKey(30) & 0xFF
		scene_name = AVAILABLE_SCENES[scene_idx]

		if key in (27, ord('q')):
			break

		if key == ord('n'):
			scene_idx = (scene_idx + 1) % len(AVAILABLE_SCENES)
			load_scene(scene_idx)
			continue
		if key == ord('p'):
			scene_idx = (scene_idx - 1) % len(AVAILABLE_SCENES)
			load_scene(scene_idx)
			continue
		if key == ord(']'):
			update_layout_catalog(scene_name)
			if scene_layout_files:
				layout_idx = cycle_index(layout_idx, len(scene_layout_files), 1)
				current_layout_path = scene_layout_files[layout_idx]
				loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
				select_after_load()
				dirty = False
				set_status(f"    切换布局 -> {os.path.basename(current_layout_path)} (加载 {loaded} / 跳过 {skipped})")
			continue
		if key == ord('['):
			update_layout_catalog(scene_name)
			if scene_layout_files:
				layout_idx = cycle_index(layout_idx, len(scene_layout_files), -1)
				current_layout_path = scene_layout_files[layout_idx]
				loaded, skipped = load_layout_into_editor(sim, current_layout_path, editor_items)
				select_after_load()
				dirty = False
				set_status(f"    切换布局 -> {os.path.basename(current_layout_path)} (加载 {loaded} / 跳过 {skipped})")
			continue

		if key == ord('h'):
			show_help = not show_help

		if key == ord('r'):
			camera_pos = np.array([0.0, CAMERA_HEIGHT, 0.0], dtype=np.float32)
			yaw = 0.0
			pitch = 0.0
			try:
				nav_pt = sim.pathfinder.get_random_navigable_point()
				camera_pos = np.array([nav_pt[0], nav_pt[1] + CAMERA_HEIGHT, nav_pt[2]], dtype=np.float32)
			except Exception:
				pass

		if key == ord('z'):
			template_idx = cycle_index(template_idx, len(templates), -1)
		if key == ord('x'):
			template_idx = cycle_index(template_idx, len(templates), 1)

		if key == ord('v'):
			spawn_pos = pick_spawn_position(camera_pos, yaw)
			model_id = templates[template_idx]
			try:
				item = create_editor_item(sim, model_id, spawn_pos, yaw)
				editor_items.append(item)
				selected_idx = len(editor_items) - 1
				dirty = True
				set_status(f"    添加物体 -> {model_id}")
			except Exception as exc:
				set_status(f"    添加失败 -> {model_id}: {exc}")

		if key == ord(',') and editor_items:
			selected_idx = cycle_index(selected_idx, len(editor_items), -1)
		if key == ord('.') and editor_items:
			selected_idx = cycle_index(selected_idx, len(editor_items), 1)

		if key == ord('b') and 0 <= selected_idx < len(editor_items):
			item = editor_items.pop(selected_idx)
			safe_remove_object(sim.get_rigid_object_manager(), item)
			selected_idx = min(selected_idx, len(editor_items) - 1)
			dirty = True
			set_status(f"    删除物体 -> {item['model_id']}")

		forward, right = get_camera_vectors(yaw)
		selected_item = editor_items[selected_idx] if 0 <= selected_idx < len(editor_items) else None
		if selected_item is not None:
			moved = False
			obj = selected_item["object"]
			pos = obj.translation
			if key == ord('t'):
				obj.translation = pos + forward * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('g'):
				obj.translation = pos - forward * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('f'):
				obj.translation = pos - right * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('y'):
				obj.translation = pos + right * OBJECT_MOVE_SPEED
				moved = True
			if key == ord('u'):
				obj.translation = pos + np.array([0.0, OBJECT_MOVE_SPEED, 0.0], dtype=np.float32)
				moved = True
			if key == ord('o'):
				obj.translation = pos - np.array([0.0, OBJECT_MOVE_SPEED, 0.0], dtype=np.float32)
				moved = True
			if key == ord('1'):
				selected_item["yaw_deg"] -= OBJECT_ROTATE_SPEED
				obj.rotation = yaw_to_quat(selected_item["yaw_deg"])
				moved = True
			if key == ord('2'):
				selected_item["yaw_deg"] += OBJECT_ROTATE_SPEED
				obj.rotation = yaw_to_quat(selected_item["yaw_deg"])
				moved = True
			if moved:
				dirty = True

		if key == ord('m'):
			save_layout(scene_name, current_layout_path, editor_items)
			update_layout_catalog(scene_name)
			dirty = False
			set_status(f"    已保存 -> {current_layout_path}")

		if key == ord('j'):
			yaw += ROTATE_SPEED
		if key == ord('l'):
			yaw -= ROTATE_SPEED
		if key == ord('i'):
			pitch = min(pitch + ROTATE_SPEED, PITCH_LIMIT)
		if key == ord('k'):
			pitch = max(pitch - ROTATE_SPEED, -PITCH_LIMIT)

		if key == ord('w'):
			camera_pos += forward * CAMERA_MOVE_SPEED
		if key == ord('s'):
			camera_pos -= forward * CAMERA_MOVE_SPEED
		if key == ord('a'):
			camera_pos -= right * CAMERA_MOVE_SPEED
		if key == ord('d'):
			camera_pos += right * CAMERA_MOVE_SPEED
		if key == ord('e'):
			camera_pos[1] += CAMERA_MOVE_SPEED
		if key == ord('c'):
			camera_pos[1] -= CAMERA_MOVE_SPEED

		try:
			agent = sim.get_agent(0)
			state = agent.get_state()
			state.position = camera_pos.copy()
			state.rotation = rotation_from_euler(yaw, pitch)
			agent.set_state(state, reset_sensors=False)
			obs = sim.get_sensor_observations()
			rgb = obs["color"][:, :, :3]
			frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
		except Exception as exc:
			frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
			overlay_text(frame, [f"渲染错误: {exc}"], 20, 60, color=(80, 80, 255))

		selected_label = "无"
		if 0 <= selected_idx < len(editor_items):
			item = editor_items[selected_idx]
			pos = item["object"].translation
			selected_label = f"{selected_idx}: {item['model_id']} @ ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f}) yaw={item['yaw_deg']:.1f}"

		hud = [
			f"场景: {scene_name}  ({scene_idx + 1}/{len(AVAILABLE_SCENES)})",
			f"布局: {os.path.basename(current_layout_path)}{' *' if dirty else ''}",
			f"模板: {templates[template_idx]}  ({template_idx + 1}/{len(templates)})",
			f"物体数: {len(editor_items)}  选中: {selected_label}",
			f"相机: ({camera_pos[0]:.2f}, {camera_pos[1]:.2f}, {camera_pos[2]:.2f})  yaw={yaw:.1f} pitch={pitch:.1f}",
			"[H] 帮助  [M] 保存",
		]
		overlay_text(frame, hud, 10, 22, color=(80, 255, 255))

		if status_text:
			overlay_text(frame, [status_text], 10, DISPLAY_HEIGHT - 16, color=(255, 220, 120))

		if show_help:
			y0 = DISPLAY_HEIGHT - len(HELP_TEXT) * 21 - 40
			overlay_text(frame, HELP_TEXT, 10, y0, color=(80, 255, 80))

		cv2.imshow(win, frame)

	if sim is not None:
		clear_editor_objects(sim, editor_items)
		sim.close()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	main()
