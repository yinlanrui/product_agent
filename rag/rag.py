from __future__ import annotations

from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from rag.attribute_retriever import (
    AttributeKnowledgeRetriever,
)

from system.compute_device import (
    get_embedding_device,
    print_device_report,
)

from system.config import (
    EMBED_MODEL_NAME,
    CONFIDENCE_HIGH_THRESHOLD,
    CONFIDENCE_MEDIUM_THRESHOLD,
    CONFIDENCE_LOW_THRESHOLD,
    CONFIDENCE_GENERIC_CAP,
)

from rag.taxonomy import TaxonomyResolver

from rag.knowledge_bootstrap import (
    ensure_knowledge_ready,
)


# ============================================================
# V0.4.0 Cleanup
# ============================================================
#
# Production knowledge architecture:
#
#   Vision / Query Understanding
#              ↓
#        TaxonomyResolver
#              ↓
#     Attribute Retrieval
#              ↓
#     Evidence-aware Gate
#              ↓
#        LLM Context
#
# Important:
# - No product_knowledge.json
# - No product_knowledge_v3 Chroma collection
# - No L3/L2/L1 hand-written copy seed knowledge
# - Taxonomy is routing/context, not a copy-knowledge database
# - Attribute knowledge is generated from canonical properties
# ============================================================


GLOBAL_GROUNDING_RULE = (
    "只允许把图片中直接可见、可读文字明确支持，"
    "或可信结构化数据明确提供的信息写成商品事实。"
    "不得因为某属性知识被检索到，就假定当前商品拥有该属性。"
    "品牌、型号、材质、精确尺寸、功率、容量、认证、性能、"
    "兼容协议等不能从外观可靠确认的信息必须省略或保守表达。"
)


class ProductKnowledgeStore:
    """
    Compatibility name retained because product_agent.py already imports
    `template_store` from rag.py.

    The class no longer stores hand-written Product Knowledge.
    It now orchestrates:
        Taxonomy routing + Canonical Attribute RAG.
    """

    def __init__(self):

        print(
            "\n[RAG] V0.4.1 Bootstrap "
            "初始化 Taxonomy + Canonical Attribute RAG..."
        )

        print_device_report(
            prefix="[RAG DEVICE]"
        )

        # One shared E5 instance:
        # taxonomy semantic routing + attribute retrieval.
        self.embeddings = (
            HuggingFaceEmbeddings(
                model_name=(
                    EMBED_MODEL_NAME
                ),
                model_kwargs={
                    "device":
                        get_embedding_device(),
                },
                encode_kwargs={
                    "normalize_embeddings":
                        True,
                },
            )
        )

        self.taxonomy = (
            TaxonomyResolver(
                embeddings=self.embeddings
            )
        )

        # Cloud/local zero-touch bootstrap:
        # if generated knowledge or Chroma is missing/stale,
        # rebuild automatically before opening Attribute Retrieval.
        ensure_knowledge_ready(
            embeddings=self.embeddings,
            verbose=True,
        )

        self.attribute_retriever = (
            AttributeKnowledgeRetriever(
                embeddings=self.embeddings
            )
        )

        # Attribute DB is the only runtime Chroma knowledge collection.
        self.db = (
            self.attribute_retriever.db
        )

        print(
            "[RAG] Attribute Knowledge Count:",
            self.count(),
        )

    # ========================================================
    # Public compatibility helpers
    # ========================================================

    def count(
        self,
    ) -> int:

        return (
            self.attribute_retriever
            .count()
        )

    @staticmethod
    def _build_query(
        retrieval_profile: dict,
    ) -> str:

        return (
            AttributeKnowledgeRetriever
            .build_attribute_query(
                retrieval_profile
            )
        )

    def _resolve_taxonomy(
        self,
        retrieval_profile: dict,
    ) -> dict[str, Any]:

        raw_category = str(
            retrieval_profile.get(
                "category",
                "",
            )
            or retrieval_profile.get(
                "raw_category",
                "",
            )
            or retrieval_profile.get(
                "product_name",
                "",
            )
        ).strip()

        context_text = (
            self._build_query(
                retrieval_profile
            )
        )

        return (
            self.taxonomy.resolve(
                raw_category=raw_category,
                context_text=context_text,
            )
        )

    # ========================================================
    # Attribute Gate
    # ========================================================

    @staticmethod
    def _select_attribute_candidates(
        attribute_result: dict,
        *,
        resolved: bool,
    ) -> list[dict]:

        selected = []

        for candidate in attribute_result.get(
            "candidates",
            [],
        ):

            final_score = float(
                candidate.get(
                    "final_score",
                    0.0,
                )
                or 0.0
            )

            family_hint = float(
                candidate.get(
                    "family_hint_score",
                    0.0,
                )
                or 0.0
            )

            dense_similarity = float(
                candidate.get(
                    "dense_similarity",
                    0.0,
                )
                or 0.0
            )

            # Keep the same conservative C4 gate.
            if final_score < 0.50:
                continue

            if (
                family_hint <= 0.0
                and dense_similarity < 0.58
            ):
                continue

            selected.append(
                candidate
            )

        # Known taxonomy: attribute knowledge is supplemental.
        # OPEN_UNKNOWN: allow slightly broader attribute guidance.
        max_k = (
            2
            if resolved
            else 3
        )

        return selected[
            :max_k
        ]

    # ========================================================
    # Confidence
    # ========================================================

    @staticmethod
    def _confidence_level(
        score: float,
    ) -> str:

        if (
            score
            >= CONFIDENCE_HIGH_THRESHOLD
        ):
            return "high"

        if (
            score
            >= CONFIDENCE_MEDIUM_THRESHOLD
        ):
            return "medium"

        if (
            score
            >= CONFIDENCE_LOW_THRESHOLD
        ):
            return "low"

        return "no_match"

    @staticmethod
    def _calculate_confidence(
        *,
        attribute_candidates: list[dict],
        taxonomy_resolution: dict,
        retrieval_profile: dict,
    ) -> tuple[
        float,
        str,
        str,
    ]:

        if not attribute_candidates:
            return (
                0.0,
                "no_match",
                "no_attribute_candidates",
            )

        top_attribute = float(
            attribute_candidates[0].get(
                "final_score",
                0.0,
            )
            or 0.0
        )

        resolved = bool(
            taxonomy_resolution.get(
                "resolved",
                False,
            )
        )

        taxonomy_score = float(
            taxonomy_resolution.get(
                "score",
                0.0,
            )
            or 0.0
        )

        if not resolved:
            taxonomy_score = 0.0

        profile_score = float(
            retrieval_profile.get(
                "confidence",
                0.0,
            )
            or 0.0
        )

        # Attribute relevance is now the primary knowledge-confidence signal.
        score = (
            0.65
            * top_attribute
            +
            0.20
            * taxonomy_score
            +
            0.15
            * profile_score
        )

        score = max(
            0.0,
            min(
                1.0,
                score,
            ),
        )

        reason = (
            f"attribute={top_attribute:.4f}; "
            f"taxonomy={taxonomy_score:.4f}; "
            f"profile={profile_score:.4f}"
        )

        # OPEN_UNKNOWN is still conservative:
        # attributes can support description, but do not prove identity.
        if (
            not resolved
            and score
            > CONFIDENCE_GENERIC_CAP
        ):
            score = (
                CONFIDENCE_GENERIC_CAP
            )

            reason += (
                "; cap=open_unknown"
            )

        level = (
            ProductKnowledgeStore
            ._confidence_level(
                score
            )
        )

        return (
            score,
            level,
            reason,
        )

    # ========================================================
    # Context Formatting
    # ========================================================

    @staticmethod
    def _format_attribute_context(
        candidates: list[dict],
    ) -> list[str]:

        parts = []

        for index, candidate in enumerate(
            candidates,
            start=1,
        ):

            parts.append(
                (
                    f"【属性知识 {index}】\n"
                    f"属性族："
                    f"{candidate.get('attribute_family', '')}\n"
                    f"属性："
                    f"{candidate.get('attribute_names', '')}\n"
                    f"Attribute Score："
                    f"{float(candidate.get('final_score', 0.0) or 0.0):.4f}\n"
                    "使用规则：此知识只是属性描述规范，"
                    "不是当前商品拥有该属性的证据。\n"
                    f"{candidate.get('content', '')}"
                )
            )

        return parts

    # ========================================================
    # Main Search
    # ========================================================

    def search(
        self,
        retrieval_profile: dict,
    ) -> dict:

        if not retrieval_profile.get(
            "is_product",
            False,
        ):
            return {
                "found": False,
                "mode": "not_product",
                "score": 0.0,
                "confidence_score": 0.0,
                "confidence_level": "no_match",
                "taxonomy_resolution": {},
                "knowledge_mode": "NONE",
                "attribute_candidates": [],
                "text": "",
                "debug": (
                    "Query Understanding 判定当前图片不是明确商品，"
                    "因此跳过 RAG。"
                ),
            }

        taxonomy_resolution = (
            self._resolve_taxonomy(
                retrieval_profile
            )
            or {}
        )

        resolved = bool(
            taxonomy_resolution.get(
                "resolved",
                False,
            )
        )

        resolution_mode = str(
            taxonomy_resolution.get(
                "resolution_mode",
                "",
            )
            or ""
        )

        if not resolved:
            mode = "open_unknown_attribute"
        elif resolution_mode.startswith(
            (
                "exact_",
                "alias_",
            )
        ):
            mode = "taxonomy_exact_attribute"
        else:
            mode = "taxonomy_semantic_attribute"

        attribute_result = (
            self.attribute_retriever
            .retrieve(
                retrieval_profile
            )
        )

        attribute_candidates = (
            self._select_attribute_candidates(
                attribute_result,
                resolved=resolved,
            )
        )

        (
            confidence_score,
            confidence_level,
            confidence_reason,
        ) = (
            self._calculate_confidence(
                attribute_candidates=(
                    attribute_candidates
                ),
                taxonomy_resolution=(
                    taxonomy_resolution
                ),
                retrieval_profile=(
                    retrieval_profile
                ),
            )
        )

        taxonomy_path = (
            " > ".join(
                taxonomy_resolution.get(
                    "path_names",
                    [],
                )
            )
            or "OPEN_UNKNOWN"
        )

        debug_parts = [
            "========== RAG V0.4.1 Bootstrap Debug ==========",
            f"mode={mode}",
            (
                "taxonomy_mode="
                f"{resolution_mode}"
            ),
            (
                "taxonomy_resolved="
                f"{resolved}"
            ),
            (
                "taxonomy_path="
                f"{taxonomy_path}"
            ),
            (
                "taxonomy_score="
                f"{float(taxonomy_resolution.get('score', 0.0) or 0.0):.4f}"
            ),
            "",
            "----- Attribute Retrieval -----",
            attribute_result.get(
                "debug",
                "",
            ),
            (
                "attribute_context_selected="
                f"{len(attribute_candidates)}"
            ),
            "",
            "----- Confidence -----",
            (
                "confidence_level="
                f"{confidence_level.upper()}"
            ),
            (
                "confidence_score="
                f"{confidence_score:.4f}"
            ),
            (
                "reason="
                f"{confidence_reason}"
            ),
            "",
            (
                "legacy_product_knowledge="
                "DISABLED"
            ),
        ]

        if (
            confidence_level
            == "no_match"
            or not attribute_candidates
        ):
            return {
                "found": False,
                "mode": mode,
                "score":
                    confidence_score,
                "confidence_score":
                    confidence_score,
                "confidence_level":
                    "no_match",
                "rrf_score": 0.0,
                "rerank_score": 0.0,
                "taxonomy_resolution":
                    taxonomy_resolution,
                "knowledge_mode":
                    "ATTRIBUTE_NONE",
                "attribute_candidates": [],
                "text": "",
                "debug":
                    "\n".join(
                        debug_parts
                    ),
            }

        context_parts = [
            (
                f"【RAG置信度："
                f"{confidence_level.upper()}】"
            ),
            (
                "【分类路径】"
                f"{taxonomy_path}"
            ),
            (
                "【事实约束】"
                + GLOBAL_GROUNDING_RULE
            ),
        ]

        context_parts.extend(
            self._format_attribute_context(
                attribute_candidates
            )
        )

        return {
            "found": True,
            "mode": mode,
            "score":
                confidence_score,
            "confidence_score":
                confidence_score,
            "confidence_level":
                confidence_level,

            # Public compatibility fields retained.
            "rrf_score":
                float(
                    attribute_candidates[0]
                    .get(
                        "rrf_score",
                        0.0,
                    )
                    or 0.0
                ),
            "rerank_score":
                float(
                    attribute_candidates[0]
                    .get(
                        "final_score",
                        0.0,
                    )
                    or 0.0
                ),

            "taxonomy_resolution":
                taxonomy_resolution,
            "knowledge_mode":
                (
                    "CANONICAL_ATTRIBUTE"
                    if resolved
                    else "OPEN_ATTRIBUTE"
                ),
            "attribute_candidates":
                attribute_candidates,
            "text":
                "\n\n".join(
                    context_parts
                ),
            "debug":
                "\n".join(
                    debug_parts
                ),
        }


# Existing product_agent.py imports this singleton.
template_store = (
    ProductKnowledgeStore()
)
