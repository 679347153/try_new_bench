#!/usr/bin/env python3
"""
交互式场景查看器 - 完全复现 vlfm 的场景加载方式，测试新导入物体是否正确加载
运行方法（在 vlfm-py39 环境，/home/adminer/vlfm/ 目录下执行）:
    python scripts/interactive_scene_viewer.py [场景名称]

按键说明:
    W/S     前进/后退
    A/D     左/右平移
    E/C     上升/下降
    I/K     上下俯仰视角
    J/L     左右旋转视角
    N/P     切换下/上一场景
    R       复位到可导航点
    H       显示/隐藏帮助
    ESC/Q   退出
"""

import argparse
import math
import os
import sys

import cv2
import numpy as np

try:
    import habitat_sim
    import habitat_sim.utils.common as utils
except ImportError:
    print("错误: 未找到 habitat_sim，请在 vlfm-py39 环境中运行")
    sys.exit(1)

# ── 完全复现 vlfm 的路径配置 ────────────────────────────────────────────────────
# vlfm 配置: scenes_dir = data/scene_datasets/hm3d_new  (见 vlfm_objectnav_hm3d.yaml)
SCENES_DIR = "data/scene_datasets/hm3d_new"

# VLFM episode 中记录的 scene_dataset_config（env.py 第105行从 episode 读取）
# 该文件通过 .object_config.json 注册物体模板，使 template_name 短名匹配生效
SCENE_DATASET_CONFIG = "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"

# scene_id 模板: episode.scene_id = "hm3d/val/{scene_name}/{glb_name}.basis.glb"
# 最终 sim scene_id = os.path.join(scenes_dir, episode.scene_id)

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

DISPLAY_WIDTH  = 1280
DISPLAY_HEIGHT = 720
MOVE_SPEED     = 0.15
ROTATE_SPEED   = 2.5
PITCH_LIMIT    = 85.0


def get_scene_id(scene_name):
    """
    返回 scene_instance 名称（而非 .glb 路径），使 habitat-sim 通过
    scene_dataset_config 中注册的 scene_instance 加载场景，
    从而正确实例化 scene_instance.json 中的 object_instances。
    例如: "00800-TEEsavR23oF" → "TEEsavR23oF"
    """
    scene_dir_abs = os.path.join(SCENES_DIR, "hm3d/val", scene_name)
    if not os.path.isdir(scene_dir_abs):
        raise FileNotFoundError(f"场景目录不存在: {scene_dir_abs}")
    # scene_instance 名称 = 场景名中 '-' 后面的部分
    parts = scene_name.split("-", 1)
    instance_name = parts[1] if len(parts) > 1 else scene_name
    return instance_name


def make_sim_cfg(scene_id):
    """完全复现 habitat_simulator.py create_sim_config 的逻辑"""
    sim_cfg = habitat_sim.SimulatorConfiguration()
    # 与 vlfm 一致
    sim_cfg.scene_dataset_config_file = SCENE_DATASET_CONFIG
    sim_cfg.scene_id = scene_id
    sim_cfg.enable_physics = False   # 必须开启物理才能实例化 scene_instance 中的物体
    sim_cfg.gpu_device_id = 0

    sensor = habitat_sim.CameraSensorSpec()
    sensor.uuid = "color"
    sensor.sensor_type = habitat_sim.SensorType.COLOR
    sensor.resolution = [DISPLAY_HEIGHT, DISPLAY_WIDTH]
    sensor.hfov = 90
    sensor.position = [0.0, 0.88, 0.0]   # 与 vlfm 默认相机高度一致

    agent_cfg = habitat_sim.agent.AgentConfiguration()
    agent_cfg.sensor_specifications = [sensor]
    agent_cfg.height = 0.88    # vlfm agent height
    agent_cfg.radius = 0.18    # vlfm agent radius

    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def rotation_from_euler(yaw_deg, pitch_deg):
    qy = utils.quat_from_angle_axis(math.radians(yaw_deg),   np.array([0, 1, 0]))
    qx = utils.quat_from_angle_axis(math.radians(pitch_deg), np.array([1, 0, 0]))
    return qy * qx


def overlay_text(img, lines, x, y, color=(255, 255, 80), scale=0.52, thickness=1):
    lh = 21
    for line in lines:
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(img, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    scale, color, thickness, cv2.LINE_AA)
        y += lh


HELP_TEXT = [
    "=== 按键说明 ===",
    "W/S    前进/后退",
    "A/D    左/右平移",
    "E/C    上升/下降",
    "I/K    俯仰视角",
    "J/L    左右旋转",
    "N/P    下/上一场景",
    "R      复位到可导航点",
    "H      显示/隐藏帮助",
    "ESC/Q  退出",
]


def main():
    # 必须在项目根目录运行（与 python -m vlfm.run 一致）
    if not os.path.isdir("data/scene_datasets"):
        print("请在项目根目录 /home/adminer/vlfm/ 下运行此脚本")
        sys.exit(1)

    parser = argparse.ArgumentParser()
    parser.add_argument("scene", nargs="?", default="00824-Dd4bFSTQ8gi")
    args = parser.parse_args()

    scene_idx = 0
    for i, s in enumerate(AVAILABLE_SCENES):
        if args.scene in s:
            scene_idx = i
            break

    win = "VLFM 场景查看器 (vlfm加载方式)  [H]帮助  [ESC/Q]退出"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    sim = None
    pos = np.array([0.0, 0.88, 0.0])
    yaw, pitch = 0.0, 0.0
    show_help = True
    scene_id_str = ""

    def load_scene(idx):
        nonlocal sim, pos, yaw, pitch, scene_id_str
        sname = AVAILABLE_SCENES[idx]
        print(f"\n>>> 加载场景 [{idx+1}/{len(AVAILABLE_SCENES)}]: {sname}")
        try:
            sid = get_scene_id(sname)
            scene_id_str = sid
            print(f"    scene_id            : {sid}")
            print(f"    scene_dataset_config: {SCENE_DATASET_CONFIG}")
            cfg = make_sim_cfg(sid)
            if sim is not None:
                sim.close()
            sim = habitat_sim.Simulator(cfg)
            # 打印已加载物体
            rom = sim.get_rigid_object_manager()
            objs = rom.get_objects_by_handle_substring()
            print(f"    已加载物体: {len(objs)} 个")
            for h, o in sorted(objs.items()):
                print(f"      {h}  pos={o.translation}")
            pos = np.array([0.0, 0.88, 0.0])
            yaw, pitch = 0.0, 0.0
            # try:
            #     nav_pt = sim.pathfinder.get_random_navigable_point()
            #     pos = np.array([nav_pt[0], nav_pt[1] + 0.88, nav_pt[2]])
            #     print(f"    导航随机起点: {pos}")
            # except Exception:
            #     pass
            print(f"    ✓ 加载成功")
        except Exception as e:
            print(f"    ✗ 加载失败: {e}")
            sim = None
            scene_id_str = "加载失败"

    load_scene(scene_idx)
    print(f"\n窗口已打开，请点击窗口获得焦点后用按键控制视角\n")

    while True:
        key = cv2.waitKey(30) & 0xFF

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
        if key == ord('r'):
            pos = np.array([0.0, 0.88, 0.0])
            yaw, pitch = 0.0, 0.0
            if sim is not None:
                try:
                    nav_pt = sim.pathfinder.get_random_navigable_point()
                    pos = np.array([nav_pt[0], nav_pt[1] + 0.88, nav_pt[2]])
                except Exception:
                    pass
        if key == ord('h'):
            show_help = not show_help

        if sim is None:
            blank = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            overlay_text(blank, ["场景加载失败", "按 N/P 切换场景"],
                         20, 60, color=(80, 80, 255))
            cv2.imshow(win, blank)
            continue

        # 旋转
        if key == ord('j'):
            yaw += ROTATE_SPEED
        if key == ord('l'):
            yaw -= ROTATE_SPEED
        if key == ord('i'):
            pitch = min(pitch + ROTATE_SPEED, PITCH_LIMIT)
        if key == ord('k'):
            pitch = max(pitch - ROTATE_SPEED, -PITCH_LIMIT)

        # 移动
        yr = math.radians(yaw)
        forward = np.array([-math.sin(yr), 0.0, -math.cos(yr)])
        right   = np.array([ math.cos(yr), 0.0, -math.sin(yr)])
        if key == ord('w'):
            pos += forward * MOVE_SPEED
        if key == ord('s'):
            pos -= forward * MOVE_SPEED
        if key == ord('a'):
            pos -= right * MOVE_SPEED
        if key == ord('d'):
            pos += right * MOVE_SPEED
        if key == ord('e'):
            pos[1] += MOVE_SPEED
        if key == ord('c'):
            pos[1] -= MOVE_SPEED

        # 渲染
        try:
            agent = sim.get_agent(0)
            state = agent.get_state()
            state.position = pos.copy()
            state.rotation = rotation_from_euler(yaw, pitch)
            agent.set_state(state, reset_sensors=False)
            obs = sim.get_sensor_observations()
            rgb = obs["color"][:, :, :3]
            frame = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2BGR)
        except Exception as e:
            frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
            overlay_text(frame, [f"渲染错误: {e}"], 20, 50, color=(80, 80, 255))

        # HUD
        sname = AVAILABLE_SCENES[scene_idx]
        hud = [
            f"场景: {sname}  ({scene_idx+1}/{len(AVAILABLE_SCENES)})",
            f"位置: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})",
            f"偏航: {yaw:.1f}  俯仰: {pitch:.1f}",
            f"[H] 帮助",
        ]
        overlay_text(frame, hud, 10, 22, color=(80, 255, 255))

        if show_help:
            y0 = DISPLAY_HEIGHT - len(HELP_TEXT) * 21 - 10
            overlay_text(frame, HELP_TEXT, 10, y0, color=(80, 255, 80))

        cv2.imshow(win, frame)

    if sim is not None:
        sim.close()
    cv2.destroyAllWindows()
    print("查看器已退出")


if __name__ == "__main__":
    main()