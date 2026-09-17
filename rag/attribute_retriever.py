from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jieba
from rank_bm25 import BM25Plus

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from system.compute_device import (
    get_embedding_device,
    print_device_report,
)

from system.config import (
    EMBED_MODEL_NAME,
    VECTOR_DB_PATH,
    RRF_K,
)


ATTRIBUTE_COLLECTION_NAME = "product_attribute_knowledge"

ATTRIBUTE_DENSE_TOP_K = 6
ATTRIBUTE_BM25_TOP_K = 6
ATTRIBUTE_FINAL_K = 6

ATTRIBUTE_FAMILY_HINTS = {
    "appearance": [
        "颜色",
        "配色",
        "图案",
        "形状",
        "外观",
        "表面",
        "造型",
        "纹理",
    ],
    "material_physical": [
        "材质",
        "材料",
        "尺寸",
        "大小",
        "宽",
        "高",
        "厚",
        "重量",
        "轻薄",
        "紧凑",
    ],
    "structure_components": [
        "结构",
        "组件",
        "配件",
        "部件",
        "手柄",
        "支架",
        "抽屉",
        "腔体",
        "盖子",
        "收纳",
    ],
    "control_interaction": [
        "旋钮",
        "按键",
        "触控",
        "遥控",
        "控制",
        "面板",
        "显示屏",
        "操作",
        "交互",
    ],
    "installation_placement": [
        "壁挂",
        "墙面",
        "桌面",
        "落地",
        "吊装",
        "安装",
        "摆放",
        "固定",
        "支架",
    ],
    "function_use": [
        "用途",
        "功能",
        "清洁",
        "烹饪",
        "显示",
        "照明",
        "收纳",
        "办公",
        "家用",
        "使用场景",
    ],
    "connectivity_compatibility": [
        "接口",
        "连接",
        "USB",
        "蓝牙",
        "无线",
        "插头",
        "端口",
        "兼容",
        "协议",
    ],
    "power_energy": [
        "供电",
        "电源",
        "充电",
        "电池",
        "插电",
        "功率",
        "能效",
        "续航",
    ],
    "identity_variant": [
        "品牌",
        "型号",
        "货号",
        "标识",
        "版本",
        "变体",
        "颜色款",
        "尺寸款",
        "材质款",
    ],
    "protection_environmental": [
        "IP等级",
        "防护等级",
        "防尘",
        "防水",
        "防护",
        "环境防护",
    ],
    "fmcg_measurement": [
        "净含量",
        "净重",
        "容量",
        "重量",
        "克",
        "千克",
        "毫升",
        "升",
    ],
    "fmcg_composition": [
        "配料",
        "成分",
        "过敏原",
        "转基因",
        "原料",
        "含有",
        "配料表",
    ],
    "fmcg_storage_use": [
        "冷藏",
        "冷冻",
        "储存",
        "保存",
        "食用场景",
        "早餐",
        "即食",
    ],
}


def normalize_text(
    text: str,
) -> str:
    return (
        " ".join(
            str(
                text
                or ""
            )
            .lower()
            .split()
        )
        .strip()
    )


def tokenize(
    text: str,
) -> list[str]:
    text = normalize_text(
        text
    )

    if not text:
        return []

    tokens = []

    for token in jieba.cut(
        text
    ):
        token = (
            token
            .strip()
            .lower()
        )

        if not token:
            continue

        if (
            len(token) == 1
            and not token.isdigit()
        ):
            continue

        tokens.append(
            token
        )

    return tokens


@dataclass
class AttributeCandidate:
    document_id: str
    document: Document

    rrf_score: float = 0.0
    dense_similarity: float = 0.0
    bm25_score: float = 0.0
    family_hint_score: float = 0.0
    final_score: float = 0.0
    rank: int | None = None

    def to_dict(
        self,
    ) -> dict[str, Any]:
        metadata = (
            self.document.metadata
            or {}
        )

        return {
            "rank":
                self.rank,

            "document_id":
                self.document_id,

            "title":
                metadata.get(
                    "title",
                    "",
                ),

            "attribute_family":
                metadata.get(
                    "attribute_family",
                    "",
                ),

            "attribute_names":
                metadata.get(
                    "attribute_names",
                    "",
                ),

            "retrieval_terms":
                metadata.get(
                    "retrieval_terms",
                    "",
                ),

            "rrf_score":
                self.rrf_score,

            "dense_similarity":
                self.dense_similarity,

            "bm25_score":
                self.bm25_score,

            "family_hint_score":
                self.family_hint_score,

            "final_score":
                self.final_score,

            "content":
                self.document.page_content,
        }


class AttributeKnowledgeRetriever:

    def __init__(
        self,
        embeddings:
            HuggingFaceEmbeddings
            | None = None,
    ):
        print(
            "\n[ATTR-RAG] 初始化 Attribute Retrieval..."
        )

        if embeddings is None:
            print_device_report(
                prefix="[ATTR-RAG DEVICE]"
            )

            embeddings = (
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
        else:
            print(
                "[ATTR-RAG] Reusing shared embedding model."
            )

        self.embeddings = embeddings

        self.db = Chroma(
            collection_name=(
                ATTRIBUTE_COLLECTION_NAME
            ),
            embedding_function=(
                self.embeddings
            ),
            persist_directory=str(
                VECTOR_DB_PATH
            ),
            collection_metadata={
                "hnsw:space":
                    "cosine",
            },
        )

        print(
            "[ATTR-RAG] Knowledge Count:",
            self.count(),
        )

    def count(
        self,
    ) -> int:
        return len(
            self.db.get().get(
                "ids",
                [],
            )
        )

    @staticmethod
    def build_attribute_query(
        retrieval_profile: dict,
        product_desc: str = "",
    ) -> str:
        parts = []

        if product_desc:
            parts.append(
                product_desc
            )

        for key in (
            "category",
            "product_name",
            "brand",
            "retrieval_query",
        ):
            value = (
                retrieval_profile
                .get(
                    key,
                    "",
                )
            )

            if value:
                parts.append(
                    str(value)
                )

        for keyword in (
            retrieval_profile
            .get(
                "keywords",
                [],
            )
        ):
            if keyword:
                parts.append(
                    str(keyword)
                )

        return " ".join(
            parts
        ).strip()

    @staticmethod
    def _detect_family_hints(
        query: str,
    ) -> dict[str, float]:
        normalized = (
            normalize_text(
                query
            )
        )

        scores = {}

        for (
            family,
            terms,
        ) in ATTRIBUTE_FAMILY_HINTS.items():

            matches = 0

            for term in terms:
                if (
                    normalize_text(term)
                    in normalized
                ):
                    matches += 1

            if matches > 0:
                scores[
                    family
                ] = min(
                    1.0,
                    matches / 3.0,
                )

        return scores

    def _get_all_documents(
        self,
    ) -> list[
        tuple[
            str,
            Document,
        ]
    ]:
        result = self.db.get(
            include=[
                "documents",
                "metadatas",
            ]
        )

        docs = []

        for (
            document_id,
            content,
            metadata,
        ) in zip(
            result.get(
                "ids",
                [],
            ),
            result.get(
                "documents",
                [],
            ),
            result.get(
                "metadatas",
                [],
            ),
        ):
            docs.append(
                (
                    document_id,
                    Document(
                        page_content=(
                            content
                            or ""
                        ),
                        metadata=(
                            metadata
                            or {}
                        ),
                    ),
                )
            )

        return docs

    def _dense_search(
        self,
        query: str,
    ):
        results = (
            self.db
            .similarity_search_with_score(
                query=(
                    "query: "
                    + query
                ),
                k=(
                    ATTRIBUTE_DENSE_TOP_K
                ),
            )
        )

        output = []

        for rank, (
            document,
            distance,
        ) in enumerate(
            results,
            start=1,
        ):
            doc_id = (
                document.metadata
                .get(
                    "document_id",
                    f"dense-{rank}",
                )
            )

            similarity = max(
                0.0,
                min(
                    1.0,
                    1.0
                    - float(
                        distance
                    ),
                ),
            )

            output.append(
                (
                    doc_id,
                    document,
                    similarity,
                    rank,
                )
            )

        return output

    def _bm25_search(
        self,
        query: str,
    ):
        source_docs = (
            self
            ._get_all_documents()
        )

        if not source_docs:
            return []

        query_tokens = (
            tokenize(
                query
            )
        )

        if not query_tokens:
            return []

        corpus = []

        for (
            _doc_id,
            document,
        ) in source_docs:
            metadata = (
                document.metadata
                or {}
            )

            text = (
                f"{metadata.get('title', '')} "
                f"{metadata.get('attribute_family', '')} "
                f"{metadata.get('attribute_names', '')} "
                f"{metadata.get('retrieval_terms', '')} "
                f"{document.page_content}"
            )

            corpus.append(
                tokenize(
                    text
                )
            )

        bm25 = BM25Plus(
            corpus
        )

        scores = (
            bm25
            .get_scores(
                query_tokens
            )
        )

        ranked = sorted(
            range(
                len(scores)
            ),
            key=lambda i:
                scores[i],
            reverse=True,
        )[
            :ATTRIBUTE_BM25_TOP_K
        ]

        output = []

        for rank, index in enumerate(
            ranked,
            start=1,
        ):
            score = float(
                scores[index]
            )

            if score <= 0:
                continue

            doc_id, document = (
                source_docs[index]
            )

            output.append(
                (
                    doc_id,
                    document,
                    score,
                    rank,
                )
            )

        return output

    def retrieve(
        self,
        retrieval_profile: dict,
        product_desc: str = "",
    ) -> dict[str, Any]:
        if not retrieval_profile.get(
            "is_product",
            False,
        ):
            return {
                "found":
                    False,
                "query":
                    "",
                "candidates":
                    [],
                "family_hints":
                    {},
                "debug":
                    "非商品，跳过 Attribute Retrieval。",
            }

        query = (
            self.build_attribute_query(
                retrieval_profile,
                product_desc,
            )
        )

        if not query:
            return {
                "found":
                    False,
                "query":
                    "",
                "candidates":
                    [],
                "family_hints":
                    {},
                "debug":
                    "Attribute query 为空。",
            }

        family_hints = (
            self
            ._detect_family_hints(
                query
            )
        )

        dense = (
            self
            ._dense_search(
                query
            )
        )

        bm25 = (
            self
            ._bm25_search(
                query
            )
        )

        merged = {}

        for (
            doc_id,
            document,
            similarity,
            rank,
        ) in dense:
            if doc_id not in merged:
                merged[
                    doc_id
                ] = AttributeCandidate(
                    document_id=(
                        doc_id
                    ),
                    document=(
                        document
                    ),
                )

            item = merged[
                doc_id
            ]

            item.rrf_score += (
                1.0
                / (
                    RRF_K
                    + rank
                )
            )

            item.dense_similarity = max(
                item.dense_similarity,
                similarity,
            )

        for (
            doc_id,
            document,
            score,
            rank,
        ) in bm25:
            if doc_id not in merged:
                merged[
                    doc_id
                ] = AttributeCandidate(
                    document_id=(
                        doc_id
                    ),
                    document=(
                        document
                    ),
                )

            item = merged[
                doc_id
            ]

            item.rrf_score += (
                1.0
                / (
                    RRF_K
                    + rank
                )
            )

            item.bm25_score = max(
                item.bm25_score,
                score,
            )

        if not merged:
            return {
                "found":
                    False,
                "query":
                    query,
                "candidates":
                    [],
                "family_hints":
                    family_hints,
                "debug":
                    "Attribute Dense/BM25 均无候选。",
            }

        max_rrf = max(
            item.rrf_score
            for item
            in merged.values()
        )

        for item in (
            merged.values()
        ):
            family = (
                item.document
                .metadata
                .get(
                    "attribute_family",
                    "",
                )
            )

            hint = (
                family_hints.get(
                    family,
                    0.0,
                )
            )

            item.family_hint_score = (
                hint
            )

            normalized_rrf = (
                item.rrf_score
                / max_rrf
                if max_rrf > 0
                else 0.0
            )

            # Simple explainable fusion:
            # 70% retrieval evidence
            # 30% explicit attribute-family hints.
            item.final_score = (
                normalized_rrf
                * 0.70
                +
                hint
                * 0.30
            )

        ranked = sorted(
            merged.values(),
            key=lambda item:
                item.final_score,
            reverse=True,
        )[
            :ATTRIBUTE_FINAL_K
        ]

        for rank, item in enumerate(
            ranked,
            start=1,
        ):
            item.rank = rank

        debug = [
            "========== V0.3.4-C3 Attribute Retrieval ==========",
            f"query={query}",
            (
                "family_hints="
                f"{family_hints}"
            ),
            "",
        ]

        for item in ranked:
            metadata = (
                item.document.metadata
                or {}
            )

            debug.append(
                (
                    f"#{item.rank} "
                    f"{metadata.get('title', '')} "
                    f"| family="
                    f"{metadata.get('attribute_family', '')} "
                    f"| final="
                    f"{item.final_score:.4f} "
                    f"| rrf="
                    f"{item.rrf_score:.6f} "
                    f"| dense="
                    f"{item.dense_similarity:.4f} "
                    f"| family_hint="
                    f"{item.family_hint_score:.4f}"
                )
            )

        return {
            "found":
                bool(
                    ranked
                ),

            "query":
                query,

            "family_hints":
                family_hints,

            "candidates": [
                item.to_dict()
                for item
                in ranked
            ],

            "debug":
                "\n".join(
                    debug
                ),
        }


def main() -> None:
    retriever = (
        AttributeKnowledgeRetriever()
    )

    tests = [
        (
            "桌面真空吸屑器",
            {
                "is_product": True,
                "category":
                    "桌面真空吸屑器",
                "product_name":
                    "桌面真空吸屑器",
                "brand": "",
                "keywords": [
                    "桌面",
                    "清洁",
                    "碎屑",
                    "按键",
                ],
                "retrieval_query":
                    "桌面小型清洁设备 吸取碎屑 按键操作",
            },
            "一个小型桌面设备，有按键，可用于清理桌面碎屑。",
        ),
        (
            "壁挂电视",
            {
                "is_product": True,
                "category":
                    "电视",
                "product_name":
                    "壁挂式电视",
                "brand": "",
                "keywords": [
                    "电视",
                    "壁挂",
                    "显示屏",
                    "墙面",
                ],
                "retrieval_query":
                    "壁挂式电视 墙面 显示屏",
            },
            "一台显示设备固定在墙面上。",
        ),
        (
            "未知透明容器设备",
            {
                "is_product": True,
                "category":
                    "未知设备",
                "product_name":
                    "透明窗口桌面设备",
                "brand": "",
                "keywords": [
                    "透明窗口",
                    "旋钮",
                    "手柄",
                    "桌面",
                ],
                "retrieval_query":
                    "桌面设备 透明窗口 旋钮 手柄",
            },
            "一个桌面设备，有透明窗口、旋钮和手柄。",
        ),
    ]

    for (
        name,
        profile,
        desc,
    ) in tests:
        print(
            "\n"
            + "=" * 78
        )
        print(
            "TEST:",
            name,
        )

        result = (
            retriever
            .retrieve(
                profile,
                desc,
            )
        )

        print(
            result["debug"]
        )


if __name__ == "__main__":
    main()
