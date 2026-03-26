# 技术实现细节 - Semantic Inspector 架构说明

## 📐 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    User Interface Layer                       │
├──────────────────────┬──────────────────────────────────────┤
│   OpenCV Window      │     Terminal / JSON Output            │
│   (1400x820)         │     (文本 & 结构化数据)               │
├──────────────────────┴──────────────────────────────────────┤
│                    Business Logic Layer                       │
├──────────────────────┬──────────────────────────────────────┤
│  Semantic Parsing    │     Scene Description Generator      │
│  (txt → entries)     │     (entries → natural language)     │
├──────────────────────┼──────────────────────────────────────┤
│  Visualization       │     AI Integration Layer             │
│  (draw_text_block)   │     (Qwen3-VL interface)            │
├──────────────────────┴──────────────────────────────────────┤
│                     Data Layer                               │
├──────────────────────┬──────────────────────────────────────┤
│  semantic.txt files  │     Model Weights (Qwen3-VL)        │
│  (661 entries each)  │     (auto-download, ~470GB)         │
└──────────────────────┴──────────────────────────────────────┘
```

---

## 🔄 数据流程

### 路径 A：基础浏览与导出

```
用户命令
    ↓
parse arguments (argparse)
    ↓
resolve_scene_list() → scene names
    ↓
build_scene_semantic() for each scene
    ├─ get_semantic_paths() → txt/glb paths
    ├─ read_semantic_txt() → parse entries
    │  └─ split_semantic_line() → robust parser
    ├─ summarize_entries() → Counter 统计
    └─ return scene_data dict
    ↓
print_scene_summary() → 终端输出摘要
    ↓
[可选] export_json() → semantic_summary.json
    ↓
[可选] visualize() → OpenCV 交互窗口
    └─ cv2.waitKeyEx() 热键响应循环
```

### 路径 B：AI 物体位置推荐

```
用户命令 + --query-ai IMAGE_PATH
    ↓
scene_data_list (已通过路径 A 获取)
    ↓
run_ai_placement_advisor()
    │
    ├─ for each scene:
    │   ├─ generate_scene_description(scene_data)
    │   │  └─ Counter 类别 + 区域统计
    │   │
    │   ├─ load_qwen3vl_model() [首次：下载 470GB]
    │   │  └─ Qwen3VLMoeForConditionalGeneration
    │   │
    │   └─ query_qwen3vl_for_placement()
    │      ├─ 读取图片 (PIL)
    │      ├─ 构造 prompt (CoT 链式提示)
    │      ├─ 调用 model.generate()
    │      ├─ extract_top3_placements() 后处理
    │      └─ return (placements, full_answer)
    │
    └─ 保存结果 → ai_placement_advice.json
```

---

## 🧩 核心模块详解

### 1. 解析模块 (Parsing)

#### `split_semantic_line(line: str) → dict`
**目的**：鲁棒性地解析 semantic.txt 中的一行

**逻辑**：
```python
text = line.strip()
tokens = re.split(r"\s+", text)
# 提取所有整数 token
ints = [int(t) for t in tokens if re.fullmatch(r"-?\d+", t)]

# 智能提取类别（支持多种格式）
if ":" in text:
    category = text.split(":", 1)[1]  # 冒号后内容
else:
    category = " ".join([t for t in tokens if not is_int(t)])

return {
    "object_id": ints[0] if ints else None,
    "region_id": ints[1] if len(ints) > 1 else None,
    "category": category,
    "raw_tokens": tokens,
    "raw": text
}
```

**容错机制**：
- 支持缺失字段
- 处理引号、冒号、多种分隔符
- 中文 + 英文类别名

#### `read_semantic_txt(txt_path) → list[dict]`
**目的**：逐行读取文件，累积成 entries 列表

**关键点**：
- 编码：UTF-8 with `errors='ignore'`（处理坏字符）
- 过滤：跳过 None 结果（空行）
- 标记：每条 entry 带 `line_no` 用于溯源

---

### 2. 统计模块 (Analytics)

#### `summarize_entries(entries) → dict`
**目的**：生成scene-level的统计摘要

**计算内容**：
```python
cat_counter = Counter()       # 类别频数
prefix_counter = Counter()    # 类别前缀 (e.g., "chair" from "chair_red")
region_counter = Counter()    # 区域 ID 频数

# Top-15 按频率排序
top_categories = cat_counter.most_common(15)
top_prefixes = prefix_counter.most_common(15)
top_regions = region_counter.most_common(15)
```

**返回值**：
```python
{
    "object_count": 661,
    "category_count": 42,
    "region_count": 8,
    "top_categories": [(cat, cnt), ...],
    "top_prefixes": [(prefix, cnt), ...],
    "top_regions": [(region_id, cnt), ...]
}
```

---

### 3. 场景描述生成 (Scene Description)

#### `generate_scene_description(scene_data) → str`
**目的**：从 semantic 数据生成自然语言场景描述

**格式**：
```
Scene: 00800-TEEsavR23oF
Objects: ceiling (15), wall (97), chair (12), sofa (3), ...
Regions: 8 distinct regions detected
```

**用途**：
- 作为 Qwen3-VL 的上下文输入
- 帮助 AI 理解场景布局
- 可以扩展为更复杂的离散化描述（如"东侧厨房"等）

---

### 4. 交互可视化模块 (Visualization)

#### `visualize(scene_data_list, keyword="")`
**目的**：OpenCV 交互式窗口浏览

**核心循环**：
```python
while True:
    # 1. 准备数据
    entries = filter_entries(scene_data["entries"], keyword)
    page = calculate_page()
    
    # 2. 绘制背景
    frame = np.zeros((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3))
    for y in range(DISPLAY_HEIGHT):  # 梯度背景
        ratio = y / (DISPLAY_HEIGHT - 1)
        frame[y, :, 0] = int(18 + 30 * ratio)
        ...
    
    # 3. 绘制区域
    draw_text_block(frame, title_lines, 20, 36, color=(90, 255, 240))
    draw_text_block(frame, entry_lines, 20, 140, color=(220, 230, 245))
    draw_text_block(frame, top_cat_lines, 930, 140, color=(130, 255, 130))
    
    # 4. 显示 & 响应热键
    cv2.imshow(WINDOW_TITLE, frame)
    key = normalize_key(cv2.waitKeyEx(30))
    
    # 5. 处理输入
    if key == ord('q'): break
    elif key == ord('n'): scene_idx = (scene_idx + 1) % len(...)
    elif key == ord('j'): page = min(page + 1, page_count - 1)
    ...
```

**渲染细节**：
- **分辨率**：1400 x 820 (宽屏比例)
- **帧率**：~33 FPS (waitKeyEx(30))
- **字体**：OpenCV 内置 FONT_HERSHEY_SIMPLEX
- **颜色**：BGR 格式（不是 RGB！）
- **文本渲染**：双层（阴影 + 亮色）

---

### 5. AI 大模型集成 (LLM Integration)

#### `load_qwen3vl_model() → (model, processor)`
**目的**：初始化 Qwen3-VL 模型

**关键步骤**：
```python
# 自动下载（首次 ~470GB，可能 10-30 分钟）
model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-235B-A22B-Thinking",
    dtype="auto",        # 自动选择精度
    device_map="auto"    # 自动分配到设备（GPU/CPU）
)

processor = AutoProcessor.from_pretrained(...)
```

**显存要求**：
- H100/A100 (80GB): 直接加载
- RTX 4090 (24GB): 需要量化（`dtype=torch.float8`）
- CPU: 可运行，但极慢

#### `query_qwen3vl_for_placement(model, processor, image_path, scene_description, object_name) → (placements, full_answer)`

**Prompt 工程**：
```
You are an expert interior designer. Analyze the following:

SCENE CONTEXT:
{scene_description}

OBJECT IN IMAGE: {object_name}

QUESTION: Based on the scene layout and the object shown, 
where are the TOP 3 MOST LIKELY positions in this scene 
to place this object? Please provide:
1. Position 1: [location] - Why?
2. Position 2: [location] - Why?
3. Position 3: [location] - Why?
```

**调用流程**：
```python
# 1. 构造消息
messages = [{
    "role": "user",
    "content": [
        {"type": "image", "image": PIL_Image},
        {"type": "text", "text": prompt}
    ]
}]

# 2. 处理
inputs = processor.apply_chat_template(
    messages, tokenize=True, return_dict=True, return_tensors="pt"
)

# 3. 推理
generated_ids = model.generate(**inputs, max_new_tokens=512)

# 4. 解码
answers = processor.batch_decode(generated_ids_trimmed, ...)

# 5. 后处理
placements = extract_top3_placements(answers[0])
```

**推理时间**：
- H100: ~1-2 分钟/场景
- RTX 4090: ~3-5 分钟/场景
- CPU: ~15-30 分钟/场景

---

## 📊 数据结构

### scene_data dict
```python
{
    "scene": "00800-TEEsavR23oF",
    "scene_dir": "data/scene_datasets/hm3d/val/00800-TEEsavR23oF",
    "semantic_txt": ".../.semantic.txt",
    "semantic_glb": ".../.semantic.glb",
    "semantic_txt_exists": True,
    "semantic_glb_exists": True,
    "semantic_glb_size_bytes": 12345678,
    
    "summary": {
        "object_count": 661,
        "category_count": 42,
        "region_count": 8,
        "top_categories": [("wall", 97), ("ceiling", 15), ...],
        "top_prefixes": [("wall", 97), ("ceiling", 15), ...],
        "top_regions": [(1, 50), (97, 32), ...]
    },
    
    "entries": [
        {
            "line_no": 2,
            "object_id": 1,
            "region_id": 97,
            "category": "ceiling",
            "raw_tokens": ["1", "97C517", '"ceiling"', "1"],
            "raw": "1,97C517,\"ceiling\",1"
        },
        ...
    ]
}
```

### ai_placement_advice.json
```json
[
    {
        "scene": "00800-TEEsavR23oF",
        "scene_description": "Scene: ...\nObjects: ...",
        "placements": [
            "1. Position 1: Beside dining table - Chairs naturally pair with eating surfaces",
            "2. Position 2: Living room - Facing television for comfortable viewing",
            "3. Position 3: Study corner - Workspace seating near desk"
        ],
        "full_response": "(Qwen3-VL 的完整回答...)"
    }
]
```

---

## 🔐 错误处理

### 关键容错点

1. **文件不存在**
   ```python
   if not os.path.isfile(txt_path):
       return []  # 返回空列表，继续执行
   ```

2. **编码问题**
   ```python
   with open(..., encoding="utf-8", errors="ignore"):  # 跳过坏字符
   ```

3. **模型加载失败**
   ```python
   try:
       from transformers import Qwen3VLMoeForConditionalGeneration
       QWEN3VL_AVAILABLE = True
   except ImportError:
       QWEN3VL_AVAILABLE = False
       print("[WARN] Qwen3-VL not available...")
   ```

4. **GPU 显存不足**
   ```python
   # device_map="auto" 自动回退到 CPU
   # 或使用量化模型
   ```

---

## ⚡ 性能优化

### 1. 解析优化
- **正则表达式缓存**：`re.split()` 已优化
- **流式读取**：逐行读取，不加载整个文件到内存

### 2. 可视化优化
- **帧缓冲**：使用单个 numpy 数组，避免重复分配
- **梯度背景**：预计算颜色值
- **条件渲染**：只在 `show_help=True` 时绘制帮助

### 3. AI 优化
- **模型缓存**：加载一次，多场景复用
- **批处理**：可扩展为批量处理多张图片
- **提示工程**：精心设计 prompt 减少推理时间

---

## 🚀 扩展点

1. **增强 semantic 解析**
   - 支持自定义解析规则
   - 导入 COCO/Cityscapes 标签本体

2. **3D 可视化**
   - 集成 Habitat-Sim 加载 `.glb`
   - 实时渲染语义分割结果

3. **多模型支持**
   - GPT-4V, Claude Vision, LLaVA 等
   - A/B 对比测试

4. **Web 服务**
   - FastAPI 后端
   - React 前端
   - WebSocket 实时流

5. **数据库**
   - PostgreSQL 存储历史数据
   - ElasticSearch 全文搜索

---

## 📚 参考资源

- **Transformers 文档**：https://huggingface.co/transformers/
- **OpenCV Python**：https://docs.opencv.org/master/d6/d00/tutorial_py_root.html
- **HM3D 数据集**：https://www.aihabitat.org/datasets/hm3d/
- **Qwen 模型**：https://github.com/QwenLM/Qwen-VL

---

**编写时间**：2026-03-26  
**代码版本**：1.0  
**维护者**：[Your Name]
