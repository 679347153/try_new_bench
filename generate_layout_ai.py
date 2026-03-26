#!/usr/bin/env python3
"""
Semantic Inspector & AI-Powered Object Placement Advisor for HM3D Scenes

================================================================================
核心功能：
================================================================================
1. 读取 .semantic.txt 与 .semantic.glb 元文件 → 获取场景语义信息
2. 结构化输出：终端摘要 + 可选 JSON 导出
3. OpenCV 交互式可视化：逐条浏览 semantic 条目
4. Qwen3-VL 大模型集成（可选）：上传图片 → 询问物体在该场景最可能出现的三个位置

================================================================================
使用教程 & 示例：
================================================================================

【基础用法 - 读取与展示】
    # 1. 展示某个场景的 semantic 信息（默认打开交互窗口）
    python generate_layout_ai.py 00800-TEEsavR23oF

    # 2. 一次性处理多个场景
    python generate_layout_ai.py 00800-TEEsavR23oF 00802-wcojb4TFT35

    # 3. 扫描所有场景
    python generate_layout_ai.py --all

【高级 - 结构化导出】
    # 导出完整 semantic 信息到 JSON（同时打开交互窗口）
    python generate_layout_ai.py --all --export-json --output semantic_info.json

    # 仅导出 JSON，不开窗口
    python generate_layout_ai.py --all --no-vis --export-json --output semantic_data.json

【关键词过滤】
    # 筛选只显示 "chair" 相关的 semantic 条目
    python generate_layout_ai.py 00800-TEEsavR23oF --filter chair

【AI 物体位置推荐（需要 Qwen3-VL）】
    # 上传一张图片，询问"这个物体在该场景最可能出现在哪三个位置"
    python generate_layout_ai.py 00800-TEEsavR23oF --query-ai path/to/image.jpg
    python generate_layout_ai.py --all --query-ai my_object.jpg --ai-engine qwen3-vl

================================================================================
交互窗口快捷键：
================================================================================
    [N] / [P]    - 下一个 / 上一个场景（当传入多个场景时）
    [J] / [K]    - 下一页 / 上一页
    [H]          - 显示/隐藏帮助提示
    [Q] / [ESC]  - 退出交互窗口

================================================================================
Qwen3-VL 集成工作流（高级）：
================================================================================
前置要求：
    1. 安装 transformers: pip install transformers[multimodal] torch torchvision
    2. 下载模型（首次自动，约 470GB）:
       - Qwen/Qwen3-VL-235B-A22B-Thinking
    3. 至少需要 GPU 显存 80GB（或使用量化模型）

工作流：
    步骤1：选择一个场景，脚本会读取其 .semantic.txt
    步骤2：用户提供一张图片，表示该图片内的物体类别
    步骤3：脚本自动生成该场景的"房间布局描述"（从 semantic 数据生成）
    步骤4：将场景描述 + 物体类别 + 图片一起发送给 Qwen3-VL
    步骤5：Qwen3-VL 推测"物体最可能出现在场景中的哪三个位置"

示例提问：
    图片: [一个红色椅子的照片]
    场景: 00800-TEEsavR23oF (某个客厅)
    提问: "这个红色椅子在这个客厅最可能放在哪三个位置? 为什么?"
    返回: "最可能: (1) 餐厅边 (置餐桌旁) (2) 客厅 (面向电视) (3) 书房 (靠近书架)"

================================================================================
JSON 导出格式说明：
================================================================================
semantic_summary.json 包含：
    - generated_at: 生成时间戳 (ISO 8601)
    - scene_count: 总场景数
    - scenes[]: 每个场景的结构化数据
        - scene: 场景名
        - semantic_source: 数据来源 ("habitat-sim")
        - summary: {object_count, category_count, region_count, top_categories, ...}
        - entries[]: 每条 semantic 记录 (对应场景中一个具体的 3D 物体)
            - object_id: 物体唯一 ID (int)
            - category: 物体类别名称 (str, 如 "chair", "sofa")
            - region_id: 所属区域 ID (str/int, 如 "kitchen_0")
            - region_category: 所属区域类别 (str, 如 "kitchen", "bedroom")
            - level_id: 所属楼层 ID (str/int)
            - aabb_center: [x, y, z] 物体包围盒中心坐标 (List[float]) -> **这就是位置信息**
            - aabb_size: [x, y, z] 物体包围盒尺寸 (List[float])
            - volume: 物体体积 (float)
            - raw: 原始标识字符串 (仅供参考)

问题回答：是的，修改后的 JSON 包含了每个物体的位置信息。
具体体现为 `entries` 列表下的 `aabb_center` 字段，它提供了物体在 3D 空间中的中心坐标 (x, y, z)。
此外还提供了 `aabb_size` (尺寸) 和 `volume` (体积) 等几何信息。
"""

import argparse
import json
import os
import re
import sys
from collections import Counter
from datetime import datetime

import cv2
import numpy as np

# Habitat-Sim imports
try:
    import magnum as mn
    import habitat_sim
except ImportError:
    print("Error: habitat_sim not found. Please install habitat-sim to use this script.")
    sys.exit(1)


# 可选的 Qwen3-VL 依赖
try:
    from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
    QWEN3VL_AVAILABLE = True
except ImportError:
    QWEN3VL_AVAILABLE = False
    print("[WARN] Qwen3-VL not installed. AI-powered object placement will be unavailable.")
    print("       Install with: pip install transformers[multimodal] torch torchvision")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False
    print("[WARN] Pillow not installed. Image loading may fail.")

SCENES_ROOT = "data/scene_datasets/hm3d/val"
SCENE_DATASET_CONFIG = "./data/scene_datasets/hm3d/hm3d_annotated_basis.scene_dataset_config.json"
DEFAULT_SCENES = [
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

WINDOW_TITLE = "Semantic Inspector  [N/P]Scene  [J/K]Page  [H]Help  [Q/ESC]Quit"
DISPLAY_WIDTH = 1400
DISPLAY_HEIGHT = 820
ROWS_PER_PAGE = 24


def normalize_key(raw_key):
    if raw_key < 0:
        return -1
    key = raw_key & 0xFF
    if ord("A") <= key <= ord("Z"):
        return key + 32
    return key


def parse_scene_id(scene_name):
    # e.g. "00800-TEEsavR23oF" -> "TEEsavR23oF"
    parts = scene_name.split("-", 1)
    return parts[1] if len(parts) > 1 else scene_name


def make_sim_cfg(scene_name):
    # 按照 test_layout.py 中的路径逻辑
    # SCENES_DIR = "data/scene_datasets/hm3d_new" (注意：你的 test_layout.py 与这里不同，我将尝试跟随 test_layout.py 的配置 logic)
    # 不过看你的 log，路径是 data/scene_datasets/hm3d/val
    
    scene_id_short = parse_scene_id(scene_name)  # TEEsavR23oF
    
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_dataset_config_file = SCENE_DATASET_CONFIG
    
    # 关键点：habitat-sim 加载 semantic 通常依赖 .scene_instance.json 或直接指定 .basis.glb
    # 如果只指定 glb，元数据加载器可能找不到关联的 .semantic.glb
    # 如果指定 scene handle (例如 "TEEsavR23oF")，则依赖 dataset config
    
    # 根据 test_layout.py:
    #   sim_cfg.scene_id = scene_id  (其中 scene_id 是短名 "TEEsavR23oF")
    #   SCENE_DATASET_CONFIG 指向 hm3d_annotated_basis.scene_dataset_config.json
    
    # 所以如果你正确配置了 SCENE_DATASET_CONFIG，你应该只需要传短 ID
    sim_cfg.scene_id = scene_id_short
    
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = 0
    sim_cfg.load_semantic_mesh = True
    
    # 强制覆盖场景路径搜索（如果不生效）
    # sim_cfg.override_scene_light_defaults = True
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    return habitat_sim.Configuration(sim_cfg, [agent_cfg])


def extract_habitat_semantics_info(scene_name):
    """
    Configure Habitat-Sim, load scene, and extract semantic scene graph.
    Returns structured data similar to old text parsing but richer.
    """
    entries = []
    
    try:
        cfg = make_sim_cfg(scene_name)
        sim = habitat_sim.Simulator(cfg)
        semantic_scene = sim.semantic_scene
        
        if not semantic_scene:
            sim.close()
            return entries, False

        # Iterate directly over habitat semantic objects
        for obj in semantic_scene.objects:
            if obj is None:
                continue

            # Extract basic info
            obj_id = obj.id
            category_obj = obj.category
            category_name = category_obj.name() if category_obj else "unknown"
            
            # Additional info available in habitat
            region_obj = obj.region
            region_id = region_obj.id if region_obj else None
            # Some helper to get region category name if available
            region_category = region_obj.category.name() if (region_obj and region_obj.category) else "unknown"
            level_obj = obj.region.level if region_obj else None
            level_id = level_obj.id if level_obj else None

            # Geometric properties (center, size)
            aabb = obj.aabb
            center = aabb.center()
            sizes = aabb.size()
            
            # Store formatted entry
            entry = {
                "object_id": obj_id,  # Start from semantic ID
                "category": category_name,
                "region_id": region_id,
                "region_category": region_category,
                "level_id": level_id,
                "aabb_center": [float(center.x), float(center.y), float(center.z)],
                "aabb_size": [float(sizes.x), float(sizes.y), float(sizes.z)],
                "volume": float(sizes.x * sizes.y * sizes.z),
                "raw": f"{obj_id} {region_id} {category_name}",  # Backward compact compatibility for display
            }
            entries.append(entry)

        sim.close()
        return entries, True

    except Exception as e:
        print(f"[ERROR] Failed to extract semantics for {scene_name}: {e}")
        return entries, False


def group_category_prefix(category):
    if not category:
        return "unknown"
    # "chair_12" -> "chair", "dining table" -> "dining"
    if "_" in category:
        return category.split("_", 1)[0].lower()
    return category.split(" ", 1)[0].lower()


def summarize_entries(entries):
    cat_counter = Counter()
    prefix_counter = Counter()
    region_counter = Counter()

    for e in entries:
        cat = e.get("category") or "unknown"
        cat_counter[cat] += 1
        prefix_counter[group_category_prefix(cat)] += 1
        region_id = e.get("region_id")
        if region_id is not None:
            region_counter[region_id] += 1

    return {
        "object_count": len(entries),
        "category_count": len(cat_counter),
        "region_count": len(region_counter),
        "top_categories": cat_counter.most_common(15),
        "top_prefixes": prefix_counter.most_common(15),
        "top_regions": region_counter.most_common(15),
    }


def build_scene_semantic(scene_name):
    # Retrieve data directly from Habitat-Sim
    entries, success = extract_habitat_semantics_info(scene_name)
    summary = summarize_entries(entries)

    scene_dir = os.path.join(SCENES_ROOT, scene_name)
    # Just for info/logging
    glb_path = os.path.join(scene_dir, f"{parse_scene_id(scene_name)}.semantic.glb")

    return {
        "scene": scene_name,
        "scene_dir": scene_dir,
        "semantic_source": "habitat-sim",
        "semantic_extracted": success,
        "semantic_glb_approx_path": glb_path,
        "summary": summary,
        "entries": entries,
    }


def print_scene_summary(scene_data):
    s = scene_data["summary"]
    print(f"\n=== Scene: {scene_data['scene']} ===")
    print(f"Source: {scene_data['semantic_source']}")
    print(f"Extraction Status: {'SUCCESS' if scene_data['semantic_extracted'] else 'FAILED'}")
    
    print(
        "objects={obj} categories={cat} regions={reg}".format(
            obj=s["object_count"], cat=s["category_count"], reg=s["region_count"]
        )
    )

    print("Top categories:")
    if not s["top_categories"]:
        print("  (none)")
    try:
        for name, cnt in s["top_categories"][:10]:
            print(f"  - {name}: {cnt}")
    except Exception:
        pass


def export_json(scene_data_list, output_path):
    payload = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "scene_count": len(scene_data_list),
        "scenes": scene_data_list,
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"\nExported JSON -> {output_path}")


def filter_entries(entries, keyword):
    if not keyword:
        return entries
    kw = keyword.lower()
    return [e for e in entries if kw in (e.get("category") or "").lower() or kw in e.get("raw", "").lower()]


def draw_text_block(frame, lines, x, y, color=(80, 255, 255), scale=0.53, thickness=1):
    line_h = 22
    for line in lines:
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
        cv2.putText(frame, line, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        y += line_h
    return frame


def visualize(scene_data_list, keyword=""):
    if not scene_data_list:
        return

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW_TITLE, DISPLAY_WIDTH, DISPLAY_HEIGHT)

    scene_idx = 0
    page = 0
    show_help = True

    while True:
        scene_data = scene_data_list[scene_idx]
        entries = filter_entries(scene_data["entries"], keyword)
        page_count = max(1, (len(entries) + ROWS_PER_PAGE - 1) // ROWS_PER_PAGE)
        page = max(0, min(page, page_count - 1))

        frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), dtype=np.uint8)
        # Soft gradient background.
        for y in range(DISPLAY_HEIGHT):
            ratio = y / max(1, DISPLAY_HEIGHT - 1)
            frame[y, :, 0] = int(18 + 30 * ratio)
            frame[y, :, 1] = int(20 + 35 * ratio)
            frame[y, :, 2] = int(24 + 42 * ratio)

        s = scene_data["summary"]
        source = scene_data.get("semantic_source", "unknown")
        status = "SUCCESS" if scene_data.get("semantic_extracted", False) else "FAILED"
        
        title = [
            f"Scene: {scene_data['scene']}  ({scene_idx + 1}/{len(scene_data_list)})",
            f"Source: {source}   Status: {status}",
            f"objects={s['object_count']} categories={s['category_count']} regions={s['region_count']}   filter='{keyword or 'none'}'",
            f"Entries: {len(entries)}  Page: {page + 1}/{page_count}",
        ]
        frame = draw_text_block(frame, title, 20, 36, color=(90, 255, 240), scale=0.6)

        start = page * ROWS_PER_PAGE
        end = min(len(entries), start + ROWS_PER_PAGE)
        lines = []
        for idx in range(start, end):
            e = entries[idx]
            lines.append(
                "{i:04d} | obj={obj} region={reg} | {cat}".format(
                    i=idx,
                    obj=e.get("object_id") if e.get("object_id") is not None else "-",
                    reg=e.get("region_id") if e.get("region_id") is not None else "-",
                    cat=e.get("category", "unknown"),
                )
            )

        if not lines:
            lines = ["(No entries to show under current filter)"]

        frame = draw_text_block(frame, lines, 20, 140, color=(220, 230, 245), scale=0.5)

        top_cat = [f"{k}:{v}" for k, v in s["top_categories"][:8]]
        side = ["Top categories:"] + (top_cat if top_cat else ["(none)"])
        frame = draw_text_block(frame, side, 930, 140, color=(130, 255, 130), scale=0.52)

        if show_help:
            help_lines = [
                "[N/P] next/prev scene",
                "[J/K] next/prev page",
                "[H] toggle help",
                "[Q]/[ESC] quit",
            ]
            frame = draw_text_block(frame, help_lines, 20, DISPLAY_HEIGHT - 110, color=(100, 255, 100), scale=0.52)

        cv2.imshow(WINDOW_TITLE, frame)
        key = normalize_key(cv2.waitKeyEx(30))
        if key < 0:
            continue

        if key in (27, ord("q")):
            break
        if key == ord("n"):
            scene_idx = (scene_idx + 1) % len(scene_data_list)
            page = 0
        if key == ord("p"):
            scene_idx = (scene_idx - 1) % len(scene_data_list)
            page = 0
        if key == ord("j"):
            page = min(page + 1, page_count - 1)
        if key == ord("k"):
            page = max(page - 1, 0)
        if key == ord("h"):
            show_help = not show_help

    cv2.destroyAllWindows()


def generate_scene_description(scene_data):
    """
    根据 semantic entries 生成该场景的自然语言描述。
    
    示例输出：
        "This scene contains: dining table (2), chair (6), sofa (1), desk (1), bed (1).
         Regions: kitchen (east), dining area (center), bedroom (west), office (north side)."
    """
    entries = scene_data.get("entries", [])
    summary = scene_data.get("summary", {})
    
    # 统计类别
    cat_counter = Counter()
    for e in entries:
        cat = e.get("category", "unknown")
        cat_counter[cat] += 1
    
    # 统计区域
    region_counter = Counter()
    for e in entries:
        reg = e.get("region_id")
        if reg is not None:
            region_counter[reg] += 1
    
    # 生成描述
    scenes = scene_data.get("scene", "unknown_scene")
    desc_parts = [f"Scene: {scenes}"]
    
    # 物体清单
    if cat_counter:
        obj_list = [f"{cat} ({cnt})" for cat, cnt in cat_counter.most_common(20)]
        desc_parts.append(f"Objects: {', '.join(obj_list)}")
    
    # 区域信息
    if region_counter:
        region_str = f"Regions: {len(region_counter)} distinct regions detected"
        desc_parts.append(region_str)
    
    whole_desc = "\n".join(desc_parts)
    return whole_desc


def load_qwen3vl_model(device_map="auto", dtype="auto"):
    """
    加载 Qwen3-VL 模型和 processor。
    
    首次调用会下载模型权重（约 470GB），可能花费较长时间。
    如果显存不足，可以尝试量化模型或 CPU 推理（较慢）。
    """
    if not QWEN3VL_AVAILABLE:
        raise RuntimeError(
            "Qwen3-VL not installed. Run: pip install transformers[multimodal] torch torchvision"
        )
    
    print("[INFO] Loading Qwen3-VL model (first time may take minutes to download)...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-235B-A22B-Thinking",
        dtype=dtype,
        device_map=device_map,
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-235B-A22B-Thinking")
    print("[INFO] Qwen3-VL model loaded successfully.")
    
    return model, processor


def query_qwen3vl_for_placement(model, processor, image_path, scene_description, object_name=None):
    """
    使用 Qwen3-VL 询问：这个物体在该场景最可能出现在哪三个位置？
    
    参数：
        model: Qwen3-VL 模型
        processor: Qwen3-VL processor
        image_path: 传入的物体图片路径
        scene_description: 场景的自然语言描述（从 generate_scene_description 获得）
        object_name: 物体的文本描述（如 "a red wooden chair"），若为 None 则从图片推断
    
    返回：
        (推荐位置剧本, 完整答案文本)
    """
    if not PIL_AVAILABLE:
        raise RuntimeError("Pillow required. Install with: pip install pillow")
    
    # 读取图片
    if not os.path.isfile(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    image = Image.open(image_path).convert("RGB")
    
    # 构造查询提示
    if object_name:
        prompt = f"""You are an expert interior designer. Analyze the following:

SCENE CONTEXT:
{scene_description}

OBJECT IN IMAGE: {object_name}

QUESTION: Based on the scene layout and the object shown, where are the TOP 3 MOST LIKELY positions 
in this scene to place this object? Please provide:
1. Position 1: [location] - Why?
2. Position 2: [location] - Why?
3. Position 3: [location] - Why?

Format your answer as a clear numbered list with brief reasoning."""
    else:
        # 让模型先识别物体，再推荐位置
        prompt = f"""You are an expert interior designer. Analyze the following:

SCENE CONTEXT:
{scene_description}

FIRST, identify the object in the image (type and characteristics).
THEN, determine the TOP 3 MOST LIKELY positions in this scene to place it.

Format your answer as:
OBJECT IDENTIFIED: [description]

PLACEMENT RECOMMENDATIONS:
1. Position 1: [location] - Why?
2. Position 2: [location] - Why?
3. Position 3: [location] - Why?"""
    
    # 调用 Qwen3-VL
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ],
        }
    ]
    
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    
    print("[WAIT] Qwen3-VL is reasoning... (this may take 1-3 minutes)")
    generated_ids = model.generate(**inputs, max_new_tokens=512)
    generated_ids_trimmed = [
        out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]
    answers = processor.batch_decode(
        generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    
    full_answer = answers[0] if answers else "(no response)"
    
    # 尝试提取前三个位置建议
    placements = extract_top3_placements(full_answer)
    
    return placements, full_answer


def extract_top3_placements(response_text):
    """
    从 Qwen3-VL 的回答中提取前三个位置建议。
    """
    lines = response_text.split("\n")
    placements = []
    
    for line in lines:
        if re.match(r"^\s*\d+\.\s*", line):
            # 匹配 "1. ", "2. " 等开头
            placements.append(line.strip())
    
    # 取前三条
    return placements[:3]


def run_ai_placement_advisor(scene_data_list, image_path, object_name=None):
    """
    主 AI 推荐工作流。
    """
    if not os.path.isfile(image_path):
        print(f"ERROR: Image not found: {image_path}")
        return
    
    print(f"\n[AI] Loading image: {image_path}")
    print(f"[AI] Scene count: {len(scene_data_list)}")
    
    # 逐场景询问
    results = []
    try:
        model, processor = load_qwen3vl_model()
    except Exception as e:
        print(f"ERROR loading model: {e}")
        return
    
    for idx, scene_data in enumerate(scene_data_list, 1):
        scene_name = scene_data.get("scene")
        print(f"\n[{idx}/{len(scene_data_list)}] Processing scene: {scene_name}")
        
        # 生成场景描述
        scene_desc = generate_scene_description(scene_data)
        print(f"[DESC] {scene_desc[:100]}...")
        
        try:
            placements, full_answer = query_qwen3vl_for_placement(
                model, processor, image_path, scene_desc, object_name
            )
            
            print(f"\n[RESULT] Scene: {scene_name}")
            print("Top 3 placement suggestions:")
            for p in placements:
                print(f"  {p}")
            
            results.append({
                "scene": scene_name,
                "scene_description": scene_desc,
                "placements": placements,
                "full_response": full_answer,
            })
        except Exception as e:
            print(f"[ERROR] Failed to query for {scene_name}: {e}")
            results.append({
                "scene": scene_name,
                "error": str(e),
            })
    
    # 保存结果
    output_file = "ai_placement_advice.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n[SAVE] AI advice saved to {output_file}")


def discover_available_scenes():
    if not os.path.isdir(SCENES_ROOT):
        return []
    names = []
    for name in sorted(os.listdir(SCENES_ROOT)):
        if os.path.isdir(os.path.join(SCENES_ROOT, name)):
            names.append(name)
    return names


def resolve_scene_list(args):
    if args.all:
        scenes = discover_available_scenes()
        if not scenes:
            scenes = list(DEFAULT_SCENES)
        return scenes

    if args.scenes:
        return args.scenes

    return [DEFAULT_SCENES[0]]


def main():
    parser = argparse.ArgumentParser(
        description="Advanced Semantic Inspector & AI-Powered Object Placement Advisor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  1. Browse semantic info:
     python generate_layout_ai.py 00800-TEEsavR23oF

  2. Export all scenes to JSON:
     python generate_layout_ai.py --all --export-json --output semantic_data.json

  3. Ask Qwen3-VL where an object should be placed:
     python generate_layout_ai.py 00800-TEEsavR23oF --query-ai my_chair.jpg

  4. Query multiple scenes (get placement advice for each):
     python generate_layout_ai.py --all --query-ai my_chair.jpg --object-name "a red wooden dining chair"
        """,
    )
    parser.add_argument("scenes", nargs="*", help="Scene names, e.g. 00800-TEEsavR23oF")
    parser.add_argument("--all", action="store_true", help="Use all scene folders")
    parser.add_argument("--filter", default="", help="Keyword filter for semantic entries")
    parser.add_argument("--no-vis", action="store_true", help="Disable OpenCV visualization window")
    parser.add_argument("--export-json", action="store_true", help="Export to JSON file")
    parser.add_argument(
        "--output", default="semantic_summary.json", help="Output JSON path"
    )
    
    # 新增 AI 相关参数
    parser.add_argument(
        "--query-ai",
        metavar="IMAGE_PATH",
        help="Path to image of object. Queries Qwen3-VL to suggest placement locations.",
    )
    parser.add_argument(
        "--object-name",
        metavar="DESCRIPTION",
        help="Text description of the object (e.g., 'a red wooden chair'). "
             "If not provided, Qwen3-VL will identify from the image.",
    )
    parser.add_argument(
        "--ai-engine",
        choices=["qwen3-vl"],
        default="qwen3-vl",
        help="AI model to use for placement prediction (default: qwen3-vl)",
    )
    
    args = parser.parse_args()

    if not os.path.isdir("data"):
        print("Please run this script from project root.")
        sys.exit(1)

    scene_list = resolve_scene_list(args)
    if not scene_list:
        print("No scenes selected.")
        sys.exit(1)

    scene_data_list = []
    for scene_name in scene_list:
        scene_data = build_scene_semantic(scene_name)
        scene_data_list.append(scene_data)
        print_scene_summary(scene_data)

    if args.export_json:
        export_json(scene_data_list, args.output)

    # 【新增】AI 物体位置推荐流程
    if args.query_ai:
        if not QWEN3VL_AVAILABLE:
            print(
                "\n[ERROR] Qwen3-VL is not available. "
                "Install with: pip install transformers[multimodal] torch torchvision"
            )
            sys.exit(1)
        
        print(f"\n{'='*70}")
        print("AI-POWERED OBJECT PLACEMENT ADVISOR")
        print(f"{'='*70}")
        print(f"Image: {args.query_ai}")
        print(f"Object description: {args.object_name or '(will be identified by AI)'}")
        print(f"Scenes to analyze: {len(scene_data_list)}")
        print(f"{'='*70}\n")
        
        run_ai_placement_advisor(scene_data_list, args.query_ai, args.object_name)
        return

    # 【原有流程】交互式可视化
    if not args.no_vis:
        visualize(scene_data_list, keyword=args.filter)


if __name__ == "__main__":
    main()
