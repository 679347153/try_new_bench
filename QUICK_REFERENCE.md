# 🎯 Semantic Inspector - 快速参考卡

## 常用命令

### 📊 浏览 & 导出

```bash
# 浏览单个场景
python generate_layout_ai.py 00800-TEEsavR23oF

# 浏览多个场景（用 N/P 切换）
python generate_layout_ai.py 00800-TEEsavR23oF 00802-wcojb4TFT35

# 导出所有场景到 JSON（同时启动窗口）
python generate_layout_ai.py --all --export-json

# 仅导出 JSON，不开窗口
python generate_layout_ai.py --all --no-vis --export-json --output data.json

# 关键词过滤（只显示 chair 相关对象）
python generate_layout_ai.py 00800-TEEsavR23oF --filter chair
```

### 🤖 AI 物体放置建议

```bash
# 单场景 + 单图片
python generate_layout_ai.py 00800-TEEsavR23oF --query-ai chair.jpg

# 指定物体名称（更精准）
python generate_layout_ai.py 00800-TEEsavR23oF --query-ai chair.jpg --object-name "a red wooden chair"

# 多场景批量查询
python generate_layout_ai.py --all --query-ai table.jpg

# 保存为自定义路径
python generate_layout_ai.py --all --query-ai sofa.jpg --output my_results.json
```

---

## ⌨️ 窗口快捷键 (按下即生效)

| 键 | 功能 |
|----|------|
| **N** / **P** | 下一个 / 上一个场景 |
| **J** / **K** | 下一页 / 上一页 |
| **H** | 显示/隐藏帮助 |
| **Q** / **ESC** | 退出 |

---

## 📋 参数参考

```
positional arguments:
  scenes                Scene names (e.g., 00800-TEEsavR23oF)

optional arguments:
  --all                 Use all scene folders
  --filter KEYWORD      Filter by category keyword
  --no-vis              Skip visualization window
  --export-json         Export to JSON file
  --output PATH         JSON output path (default: semantic_summary.json)
  --query-ai IMAGE      AI-powered placement advice (needs Qwen3-VL)
  --object-name DESC    Object description (optional, e.g., "a red chair")
  --ai-engine ENGINE    AI model (default: qwen3-vl)
```

---

## 🛠️ 安装依赖

### 基础（必需）
```bash
pip install opencv-python numpy pillow
```

### AI 推荐（可选）
```bash
pip install transformers[multimodal] torch torchvision
```

**首次运行 `--query-ai` 时会自动下载 Qwen3-VL 模型（~470GB）**

---

## 📂 输出文件

| 文件名 | 来自 | 说明 |
|-------|------|------|
| `semantic_summary.json` | `--export-json` | 所有场景的语义数据 |
| `ai_placement_advice.json` | `--query-ai` | AI 的位置建议 |

---

## 💡 工作流示例

### 例 1：我想看某个房间有什么家具

```bash
python generate_layout_ai.py 00800-TEEsavR23oF
# 按 J/K 翻页浏览所有对象
# 按 H 显示帮助
# 按 Q 退出
```

### 例 2：我想对比多个房间的布局

```bash
python generate_layout_ai.py 00800-TEEsavR23oF 00802-wcojb4TFT35
# 按 N 切换到下一个场景
# 按 P 回到上一个
```

### 例 3：我有一张椅子的照片，想知道在某个房间哪里放最合适

```bash
python generate_layout_ai.py 00800-TEEsavR23oF --query-ai my_chair.jpg
# 等待模型推理（1-2 分钟）
# 查看 ai_placement_advice.json 中的建议
```

### 例 4：我想批量分析所有房间的椅子位置

```bash
python generate_layout_ai.py --all --query-ai chair.jpg --object-name "a wooden dining chair"
# 模型会分析每个房间，给出三个最佳位置
# 结果汇总到 ai_placement_advice.json
```

---

## 🐛 故障排除

| 问题 | 解决方案 |
|------|--------|
| `ModuleNotFoundError: cv2` | `pip install opencv-python` |
| `ModuleNotFoundError: PIL` | `pip install pillow` |
| Qwen3-VL 加载失败 | `pip install transformers[multimodal] torch torchvision` |
| GPU 显存不足 | 使用量化模型或仅使用基础功能（--no-vis） |
| semantic.txt 找不到 | 检查场景名称是否正确，确认文件在 `data/scene_datasets/hm3d/val/` 下 |

---

## 📌 记住这些

✅ **推荐方式**：
- 用 `--all --export-json --no-vis` 快速导出所有数据
- 用 `--query-ai` 配合具体图片和场景进行精准查询

❌ **避免**：
- 在没有 GPU 的情况下使用 `--query-ai`（会很慢）
- 一次性查询 >1000 张图片（考虑写脚本分批）

⚙️ **性能提示**：
- 关闭 OpenCV 窗口可加速导出（`--no-vis`）
- 使用 `--filter` 减少渲染条目数
- 在 GPU 上运行 AI 查询（快 10 倍以上）

---

**更多细节见** [SEMANTIC_INSPECTOR_USAGE.md](SEMANTIC_INSPECTOR_USAGE.md)
