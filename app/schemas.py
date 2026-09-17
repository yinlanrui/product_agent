from typing import TypedDict
from PIL import Image


class RetrievalProfile(
    TypedDict,
    total=False,
):
    """
    专门为 RAG 构造的结构化查询信息。

    它和 Vision 原始描述不同。

    Vision：
        负责看到什么。

    RetrievalProfile：
        负责告诉 RAG 应该检索什么。
    """

    is_product: bool

    category: str

    product_name: str

    brand: str

    keywords: list[str]

    retrieval_query: str

    confidence: float

    reasoning: str


class AgentState(
    TypedDict,
    total=False,
):

    # ========================================================
    # Input
    # ========================================================

    image: Image.Image

    user_query: str

    # ========================================================
    # Vision
    # ========================================================

    product_desc: str

    # ========================================================
    # Query Transformation
    # ========================================================

    retrieval_profile: RetrievalProfile

    # ========================================================
    # RAG
    # ========================================================

    template_text: str

    rag_found: bool

    rag_mode: str

    rag_score: float

    rag_debug: str

    # ========================================================
    # Output
    # ========================================================

    final_answer: str

    # ========================================================
    # Error
    # ========================================================

    error: str