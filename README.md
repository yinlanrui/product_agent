# AI 商品运营 Agent

一个面向电商商品运营场景的多模态 Agent，根据商品图片调用deepseek大模型生成商品文案

核心流程：

```text
商品图片
↓
Vision 视觉识别
↓
Query Understanding
↓
Taxonomy 商品分类解析
↓
Canonical Attribute RAG
↓
Confidence / Context Guard
↓
DeepSeek
↓
商品标题 + 核心卖点 + 商品文案
```

项目目标是：即使某个商品没有被单独写进代码或知识库，Agent 仍然可以根据图片内容、商品分类和属性知识生成较为可靠的商品文案。

---

# 一、功能介绍

当前项目主要包含以下能力：

- **商品图片理解**
  - 使用 Moondream2 对商品图片进行视觉识别。
  - 提取商品类别、名称、关键词和可见属性。

- **商品分类解析**
  - 使用自建三级 Taxonomy。
  - 支持 Exact、Alias、Semantic 和 OPEN_UNKNOWN。
  - 未知商品不会被强行归类。

- **RAG 检索增强**
  - 使用 `intfloat/multilingual-e5-small` 进行 Dense Retrieval。
  - 使用 BM25Plus 进行关键词检索。
  - 使用 RRF / Fusion 融合候选结果。
  - 使用 Attribute Family Hint 增强属性相关性。

- **Canonical Attribute Knowledge**
  - 当前属性知识覆盖颜色、材质、尺寸、品牌、安装方式、连接方式、功耗、IP 等级、净含量、配料、过敏原等。
  - 属性知识保留 Evidence Policy，用于降低模型幻觉。

- **Confidence 与 Context Guard**
  - 对检索结果进行置信度控制。
  - 检索到知识不代表商品一定具有该属性。
  - 只有有图片、文字或可信结构化数据支持的内容，才允许写成商品事实。

- **OPEN_UNKNOWN 支持**
  - 即使商品无法可靠分类，也可以继续通过属性知识生成保守文案。

- **知识库自动初始化**
  - 启动时自动检查知识文件、Manifest 与 Chroma。
  - 知识缺失或过期时自动重新构建。
  - 普通用户不需要手工初始化知识库。

- **GPU / CPU 自适应**
  - `COMPUTE_DEVICE=auto`
  - 有 CUDA 时自动使用 GPU。
  - 没有 CUDA 时自动回退 CPU。

---

# 二、项目结构

```text
AGENT_DEMO/
│
├── app/
│   ├── main.py
│   ├── product_agent.py
│   ├── schemas.py
│   └── services.py
│
├── rag/
│   ├── rag.py
│   ├── attribute_retriever.py
│   ├── knowledge_bootstrap.py
│   └── taxonomy.py
│
├── system/
│   ├── compute_device.py
│   └── config.py
│
├── knowledge/
│   ├── taxonomy.json
│   ├── canonical_attribute_properties.json
│   ├── attribute_knowledge.generated.json
│   └── knowledge_manifest.json
│
├── scripts/
│   └── rebuild_knowledge.py
│
├── chroma_db_rag_v3/
│
├── .env
└── requirements.txt
```

各部分作用：

- `app/`
  - Agent 主流程、Gradio UI、Vision / DeepSeek 服务、数据结构。

- `rag/`
  - Taxonomy、Attribute Retrieval、Confidence、Context Assembly、知识库自动检查与重建。

- `system/`
  - 全局配置、路径管理、CUDA / CPU 选择。

- `knowledge/`
  - 商品分类和属性知识数据。

- `scripts/`
  - 手动维护工具。
  - 正常运行 Agent 不需要执行。

- `chroma_db_rag_v3/`
  - Chroma 向量数据库运行目录。

---

# 三、使用说明

## 1. 克隆项目

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd agent_demo
```

建议始终在项目根目录运行命令。

---

## 2. 创建虚拟环境

推荐 Python 3.11。

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. 安装依赖

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

---

## 4. 配置 `.env`

请在项目根目录创建 `.env`。

示例：

```env
DEEPSEEK_API_KEY=your_deepseek_api_key
需要你自己买一个key
DEEPSEEK_BASE_URL=https://api.deepseek.com
DEEPSEEK_MODEL=deepseek-chat

COMPUTE_DEVICE=auto
```

建议 GitHub 仓库只提交 `.env.example`，不要提交真实 `.env`。

---

## 5. 首次模型下载

项目会自动从 Hugging Face 下载本地模型，不需要手工下载模型权重。

当前使用：

```text
Vision:
vikhyatk/moondream2
revision = 2024-08-26

Embedding:
intfloat/multilingual-e5-small
```

首次运行时：

```text
启动项目
↓
检查 Hugging Face 本地缓存
↓
模型不存在
↓
自动下载 Moondream2
↓
自动下载 multilingual-e5-small
↓
保存到本地缓存
↓
后续启动直接复用
```

Windows 默认缓存通常位于：

```text
C:\Users\<用户名>\.cache\huggingface\hub
```

Linux / macOS 默认通常位于：

```text
~/.cache/huggingface/hub
```

如果希望把模型缓存放到其他磁盘，可以在 `.env` 中设置：

```env
HF_HOME=D:/huggingface_cache
```

首次运行需要联网，模型下载完成后不会每次重复完整下载。

---

## 6. 启动项目

从项目根目录执行：

```bash
python -m app.main
```

首次启动时还会自动检查知识库。

如果知识状态正常：

```text
[KNOWLEDGE] Existing index is current.
```

如果知识缺失或过期：

```text
[KNOWLEDGE] Missing or stale knowledge index detected.
```

系统会自动生成知识并重建 Chroma，不需要手工初始化。

---

## 7. 强制重建知识库

仅在开发者需要手工强制重建时使用：

```bash
python -m scripts.rebuild_knowledge
```

普通用户不需要执行。

---

## 8. GPU / CPU

推荐保持：

```env
COMPUTE_DEVICE=auto
```

运行逻辑：

```text
检测到 CUDA
→ GPU

未检测到 CUDA
→ CPU
```

如果启动日志出现：

```text
selected=cuda
```

表示 GPU 已启用。

如果出现：

```text
selected=cpu
```

项目仍可运行，但 Vision 和 Embedding 会更慢。

---

# 四、当前主要模型

```text
Vision:
vikhyatk/moondream2

Embedding:
intfloat/multilingual-e5-small

Generation:
DeepSeek API
```

本项目不会把 Moondream2 或 E5 模型权重上传到 GitHub。

GitHub 仓库只保存：

```text
模型 ID
模型 revision
加载代码
项目代码
知识数据
```

模型权重由 Hugging Face 在首次运行时自动下载。
