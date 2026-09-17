from pathlib import Path
import os

from dotenv import load_dotenv


# ============================================================
# Project / Paths
# ============================================================
#
# Path rule after the one-level package refactor:
#   - ONLY this module derives the project root from __file__.
#   - Other modules import the path constants defined here.
#
# This prevents app/, rag/, system/ and scripts/ from each calculating
# a different project root after files are moved.
# ============================================================

PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parent
    .parent
)

# Backward-compatible alias used by a few existing modules.
BASE_DIR = PROJECT_ROOT

ENV_FILE = (
    PROJECT_ROOT / ".env"
)

load_dotenv(
    ENV_FILE
)


# ============================================================
# Knowledge Data Layer
# ============================================================

KNOWLEDGE_DIR = (
    PROJECT_ROOT / "knowledge"
)

TAXONOMY_FILE = (
    KNOWLEDGE_DIR / "taxonomy.json"
)

CANONICAL_ATTRIBUTE_FILE = (
    KNOWLEDGE_DIR
    / "canonical_attribute_properties.json"
)

GENERATED_ATTRIBUTE_FILE = (
    KNOWLEDGE_DIR
    / "attribute_knowledge.generated.json"
)

KNOWLEDGE_MANIFEST_FILE = (
    KNOWLEDGE_DIR
    / "knowledge_manifest.json"
)



# ============================================================
# DeepSeek
# ============================================================

DEEPSEEK_API_KEY = os.getenv(
    "DEEPSEEK_API_KEY"
)

DEEPSEEK_BASE_URL = os.getenv(
    "DEEPSEEK_BASE_URL",
    "https://api.deepseek.com",
)

DEEPSEEK_MODEL = os.getenv(
    "DEEPSEEK_MODEL",
    "deepseek-chat",
)


# ============================================================
# Compute Device
# ============================================================
#
# Recommended:
#   auto -> CUDA when available, otherwise CPU.
#
# Optional .env override:
#   COMPUTE_DEVICE=auto
#   COMPUTE_DEVICE=cuda
#   COMPUTE_DEVICE=cpu
#
COMPUTE_DEVICE = os.getenv(
    "COMPUTE_DEVICE",
    "auto",
).strip().lower()


# ============================================================
# Vision
# ============================================================

VISION_MODEL_ID = (
    "vikhyatk/moondream2"
)

VISION_MODEL_REVISION = (
    "2024-08-26"
)


# ============================================================
# Embedding
# ============================================================

EMBED_MODEL_NAME = (
    "intfloat/multilingual-e5-small"
)


# ============================================================
# Chroma
# ============================================================

VECTOR_DB_PATH = (
    PROJECT_ROOT / "chroma_db_rag_v3"
)



# ============================================================
# Hybrid Retrieval
# ============================================================

DENSE_TOP_K = 8

BM25_TOP_K = 8

HYBRID_TOP_K = 5

FINAL_CONTEXT_K = 3


# ============================================================
# RRF
# ============================================================

RRF_K = 60


# ============================================================
# Reranker V0.3.2
# ============================================================

# 当前 Agent 的默认任务就是：
#
# 图片 → 商品文案
#
# 以后增加用户任务输入后，可以直接让
# retrieval_profile 中携带 intent，
# 这里的 default 只是 fallback。
DEFAULT_RETRIEVAL_INTENT = "product_copy"


# ------------------------------------------------------------
# Rerank Score 权重
# ------------------------------------------------------------
#
# 总和 = 1.0
#
# 当前版本刻意保持简单且可解释。
#
# RRF：
#   保留 Dense + BM25 的原始检索判断。
#
# Document Type：
#   判断知识类型是否适合任务。
#
# Keyword：
#   query 与知识关键词/正文是否匹配。
#
# Entity：
#   product_name / brand 是否出现。
#
# Category：
#   商品类别是否完全一致。

RERANK_RRF_WEIGHT = 0.40

RERANK_DOCUMENT_TYPE_WEIGHT = 0.25

RERANK_KEYWORD_WEIGHT = 0.20

RERANK_ENTITY_WEIGHT = 0.10

RERANK_CATEGORY_WEIGHT = 0.05


# ============================================================
# Threshold / Confidence V0.3.3
# ============================================================

# ------------------------------------------------------------
# Candidate safety gate
# ------------------------------------------------------------
#
# 这是 RRF 阶段的最低候选门槛，只用于丢弃明显无效候选。
# 它不是最终的“可信度阈值”。
RAG_MIN_SCORE = 0.015

# V0.3.4-B3.2:
# Weighted hierarchical RRF has a smaller numerical scale than the
# old unweighted two-list RRF. This gate only removes effectively-zero
# candidates; final trust is still decided by Confidence.
HIERARCHICAL_RRF_MIN_SCORE = 0.005


# ------------------------------------------------------------
# Confidence level thresholds
# ------------------------------------------------------------
#
# 最终 Confidence Score 位于 0~1。
#
# HIGH:
#   可以正常使用检索知识。
#
# MEDIUM:
#   知识具有参考价值，但生成时应避免强断言。
#
# LOW:
#   只保留最相关的一条知识作为弱参考。
#
# NO_MATCH:
#   不把具体知识送给生成模型。
CONFIDENCE_HIGH_THRESHOLD = 0.78
CONFIDENCE_MEDIUM_THRESHOLD = 0.64
CONFIDENCE_LOW_THRESHOLD = 0.50


# ------------------------------------------------------------
# Confidence score weights
# ------------------------------------------------------------
#
# 总和 = 1.0
#
# rerank:
#   候选经过业务 Reranker 后的综合相关性。
#
# dense:
#   语义检索本身的相似度。
#
# agreement:
#   Dense 与 BM25 是否同时召回了 Top1。
#
# margin:
#   Top1 与 Top2 的 Rerank Score 是否拉开差距。
#
# profile:
#   Query Understanding 对商品识别/分类的置信度。
#
# category:
#   检索结果与目标 category 是否一致。
CONFIDENCE_RERANK_WEIGHT = 0.45
CONFIDENCE_DENSE_WEIGHT = 0.20
CONFIDENCE_AGREEMENT_WEIGHT = 0.10
CONFIDENCE_MARGIN_WEIGHT = 0.10
CONFIDENCE_PROFILE_WEIGHT = 0.10
CONFIDENCE_CATEGORY_WEIGHT = 0.05


# Top1 - Top2 的差值达到这个值时，
# margin signal 视为 1.0。
CONFIDENCE_MARGIN_REFERENCE = 0.15


# 如果是精确 category 检索，但 Top1 Dense similarity
# 低于该值，则最多只能判为 MEDIUM。
CONFIDENCE_MIN_DENSE_SIMILARITY = 0.55


# Generic fallback 不是该商品类别的专属知识，
# 因此置信度最高限制在 LOW 区间。
CONFIDENCE_GENERIC_CAP = 0.59


# ------------------------------------------------------------
# Context policy
# ------------------------------------------------------------
#
# Confidence 不同，给 LLM 的知识数量也不同。
CONFIDENCE_HIGH_CONTEXT_K = 3
CONFIDENCE_MEDIUM_CONTEXT_K = 2
CONFIDENCE_LOW_CONTEXT_K = 1


# ============================================================
# Generic Fallback
# ============================================================

RAG_USE_GENERIC_FALLBACK = True


# ============================================================
# Gradio
# ============================================================

GRADIO_HOST = "127.0.0.1"

GRADIO_PORT = 7860


# ============================================================
# Validation
# ============================================================

def validate_settings():

    if not DEEPSEEK_API_KEY:

        raise RuntimeError(
            "\n没有读取到 DEEPSEEK_API_KEY。\n"
            "请检查项目根目录 .env 文件。\n"
        )


def validate_knowledge_settings():

    if not KNOWLEDGE_DIR.exists():

        raise RuntimeError(
            f"知识目录不存在：\n"
            f"{KNOWLEDGE_DIR}"
        )

    missing = [
        path
        for path in (
            TAXONOMY_FILE,
            CANONICAL_ATTRIBUTE_FILE,
            GENERATED_ATTRIBUTE_FILE,
        )
        if not path.exists()
    ]

    if missing:

        formatted = "\n".join(
            str(path)
            for path in missing
        )

        raise RuntimeError(
            "缺少正式知识文件：\n"
            + formatted
        )

