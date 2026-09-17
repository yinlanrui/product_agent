from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

from langchain_huggingface import HuggingFaceEmbeddings

from system.compute_device import (
    get_embedding_device,
    print_device_report,
)

from system.config import (
    TAXONOMY_FILE,
    EMBED_MODEL_NAME,
)


DEFAULT_TAXONOMY_PATH = TAXONOMY_FILE
DEFAULT_EMBED_MODEL_NAME = EMBED_MODEL_NAME

SEMANTIC_ACCEPT_THRESHOLD = 0.67
SEMANTIC_LOW_THRESHOLD = 0.54
SEMANTIC_MARGIN_THRESHOLD = 0.035
TOP_CANDIDATES = 8


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = str(text).strip().lower()
    return re.sub(r"[\s_/\\\-—–·,，。；;：:、|]+", "", text)


def cosine_similarity(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


@dataclass
class Candidate:
    level: int
    category_id: str
    name: str
    path_ids: list[str]
    path_names: list[str]
    score: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Resolution:
    raw_category: str
    resolution_mode: str
    resolved: bool
    level: int | None
    category_id: str | None
    category_name: str | None
    path_ids: list[str]
    path_names: list[str]
    score: float
    candidates: list[dict[str, Any]]
    reasoning: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class TaxonomyResolver:
    """
    V0.3.4-A2
    Three-level open-world taxonomy resolver.

    Matching order:
      exact L3 -> exact L2 -> exact L1
      alias L3 -> alias L2 -> alias L1
      semantic L3/L2/L1
      open_unknown

    There is deliberately NO examples-based routing.
    """

    def __init__(
        self,
        taxonomy_path: str | Path | None = None,
        embed_model_name: str = DEFAULT_EMBED_MODEL_NAME,
        semantic_accept_threshold: float = SEMANTIC_ACCEPT_THRESHOLD,
        semantic_low_threshold: float = SEMANTIC_LOW_THRESHOLD,
        semantic_margin_threshold: float = SEMANTIC_MARGIN_THRESHOLD,
        embeddings: HuggingFaceEmbeddings | None = None,
    ):
        self.taxonomy_path = Path(taxonomy_path or DEFAULT_TAXONOMY_PATH)
        self.embed_model_name = embed_model_name
        self.semantic_accept_threshold = semantic_accept_threshold
        self.semantic_low_threshold = semantic_low_threshold
        self.semantic_margin_threshold = semantic_margin_threshold

        self.categories = self._load()
        self.nodes = self._flatten()
        self._build_indexes()

        # Optional dependency injection:
        # HierarchicalKnowledgeRetriever can pass its already-loaded
        # multilingual-e5-small instance here. This prevents two
        # identical embedding models from occupying GPU/CPU memory.
        self._embeddings = embeddings
        self._semantic_vectors = None

    def _load(self) -> list[dict[str, Any]]:
        if not self.taxonomy_path.exists():
            raise FileNotFoundError(f"Taxonomy file not found: {self.taxonomy_path}")
        data = json.loads(self.taxonomy_path.read_text(encoding="utf-8"))
        categories = data.get("categories", []) if isinstance(data, dict) else data
        if not categories:
            raise ValueError("taxonomy.json contains no categories")
        return categories

    def _flatten(self) -> list[dict[str, Any]]:
        nodes = []
        for l1 in self.categories:
            nodes.append({
                "level": 1,
                "node": l1,
                "path_ids": [l1["category_id"]],
                "path_names": [l1["name"]],
            })
            for l2 in l1.get("children", []):
                nodes.append({
                    "level": 2,
                    "node": l2,
                    "path_ids": [l1["category_id"], l2["category_id"]],
                    "path_names": [l1["name"], l2["name"]],
                })
                for l3 in l2.get("children", []):
                    nodes.append({
                        "level": 3,
                        "node": l3,
                        "path_ids": [l1["category_id"], l2["category_id"], l3["category_id"]],
                        "path_names": [l1["name"], l2["name"], l3["name"]],
                    })
        return nodes

    def _build_indexes(self) -> None:
        self.name_index: dict[str, list[dict[str, Any]]] = {}
        self.alias_index: dict[str, list[dict[str, Any]]] = {}

        for item in self.nodes:
            node = item["node"]
            name_key = normalize_text(node["name"])
            self.name_index.setdefault(name_key, []).append(item)

            for alias in node.get("aliases", []):
                key = normalize_text(alias)
                if key:
                    self.alias_index.setdefault(key, []).append(item)

    @staticmethod
    def _choose_most_specific(items: list[dict[str, Any]]) -> dict[str, Any]:
        return sorted(items, key=lambda x: x["level"], reverse=True)[0]

    def _deterministic(self, raw: str) -> Resolution | None:
        key = normalize_text(raw)
        if not key:
            return None

        if key in self.name_index:
            item = self._choose_most_specific(self.name_index[key])
            node = item["node"]
            return Resolution(
                raw_category=raw,
                resolution_mode=f"exact_l{item['level']}",
                resolved=True,
                level=item["level"],
                category_id=node["category_id"],
                category_name=node["name"],
                path_ids=item["path_ids"],
                path_names=item["path_names"],
                score=1.0,
                candidates=[],
                reasoning="Exact canonical taxonomy match."
            )

        if key in self.alias_index:
            item = self._choose_most_specific(self.alias_index[key])
            node = item["node"]
            return Resolution(
                raw_category=raw,
                resolution_mode=f"alias_l{item['level']}",
                resolved=True,
                level=item["level"],
                category_id=node["category_id"],
                category_name=node["name"],
                path_ids=item["path_ids"],
                path_names=item["path_names"],
                score=0.98,
                candidates=[],
                reasoning="Exact taxonomy alias match."
            )

        return None

    def _ensure_semantic_index(self) -> None:
        if self._semantic_vectors is not None:
            return

        if self._embeddings is None:
            print_device_report(
                prefix="[TAXONOMY DEVICE]"
            )

            self._embeddings = HuggingFaceEmbeddings(
                model_name=self.embed_model_name,
                model_kwargs={"device": get_embedding_device()},
                encode_kwargs={"normalize_embeddings": True},
            )
        else:
            print(
                "[TAXONOMY] Reusing shared embedding model."
            )

        texts = []
        for item in self.nodes:
            node = item["node"]
            aliases = " ".join(node.get("aliases", []))
            desc = node.get("semantic_description", "")
            path = " > ".join(item["path_names"])
            texts.append(f"passage: {path} {node['name']} {aliases} {desc}")

        self._semantic_vectors = self._embeddings.embed_documents(texts)

    def _semantic_candidates(self, raw: str, context_text: str = "") -> list[Candidate]:
        self._ensure_semantic_index()
        q = f"query: {raw} {context_text}".strip()
        qv = self._embeddings.embed_query(q)

        candidates = []
        for item, vec in zip(self.nodes, self._semantic_vectors):
            node = item["node"]
            score = cosine_similarity(qv, vec)

            # Prefer stable product category L3 when scores are close.
            level_bonus = {3: 0.018, 2: 0.008, 1: 0.0}[item["level"]]
            effective = min(1.0, score + level_bonus)

            candidates.append(Candidate(
                level=item["level"],
                category_id=node["category_id"],
                name=node["name"],
                path_ids=item["path_ids"],
                path_names=item["path_names"],
                score=effective,
            ))

        candidates.sort(key=lambda x: x.score, reverse=True)
        return candidates[:TOP_CANDIDATES]

    def resolve(self, raw_category: str, context_text: str = "") -> dict[str, Any]:
        raw = (raw_category or "").strip()

        if not raw:
            return Resolution(
                raw_category="",
                resolution_mode="open_unknown",
                resolved=False,
                level=None,
                category_id=None,
                category_name=None,
                path_ids=[],
                path_names=[],
                score=0.0,
                candidates=[],
                reasoning="Empty category; taxonomy is not forced."
            ).to_dict()

        deterministic = self._deterministic(raw)
        if deterministic:
            return deterministic.to_dict()

        candidates = self._semantic_candidates(raw, context_text)
        if not candidates:
            return Resolution(
                raw_category=raw,
                resolution_mode="open_unknown",
                resolved=False,
                level=None,
                category_id=None,
                category_name=None,
                path_ids=[],
                path_names=[],
                score=0.0,
                candidates=[],
                reasoning="No semantic candidates."
            ).to_dict()

        top1 = candidates[0]
        top2 = candidates[1].score if len(candidates) > 1 else 0.0
        margin = top1.score - top2

        if (
            top1.score >= self.semantic_accept_threshold
            and margin >= self.semantic_margin_threshold
        ):
            return Resolution(
                raw_category=raw,
                resolution_mode=f"semantic_l{top1.level}",
                resolved=True,
                level=top1.level,
                category_id=top1.category_id,
                category_name=top1.name,
                path_ids=top1.path_ids,
                path_names=top1.path_names,
                score=top1.score,
                candidates=[x.to_dict() for x in candidates],
                reasoning="Semantic match is sufficiently strong and separated from alternatives."
            ).to_dict()

        return Resolution(
            raw_category=raw,
            resolution_mode="open_unknown",
            resolved=False,
            level=None,
            category_id=None,
            category_name=None,
            path_ids=[],
            path_names=[],
            score=top1.score,
            candidates=[x.to_dict() for x in candidates],
            reasoning=(
                "Semantic candidates exist but confidence/margin is insufficient; "
                "keep the product open-set instead of forcing a category."
            )
        ).to_dict()

    def build_retrieval_scope(self, resolution: dict[str, Any]) -> dict[str, Any]:
        scopes = []
        path_ids = resolution.get("path_ids", [])
        path_names = resolution.get("path_names", [])

        # Most specific -> most general
        for idx in range(len(path_ids) - 1, -1, -1):
            level = idx + 1
            priority = {3: 1.0, 2: 0.72, 1: 0.48}.get(level, 0.3)
            scopes.append({
                "scope_type": "taxonomy",
                "scope_level": level,
                "scope_id": path_ids[idx],
                "scope_name": path_names[idx],
                "priority": priority,
            })

        scopes.append({
            "scope_type": "global",
            "scope_level": 0,
            "scope_id": "global",
            "scope_name": "全局商品规则",
            "priority": 0.25,
        })

        if not resolution.get("resolved"):
            knowledge_mode = "ATTRIBUTE_ONLY"
        elif resolution.get("level") == 3:
            knowledge_mode = "EXACT_OR_SEMANTIC_L3"
        else:
            knowledge_mode = "PARENT_KNOWN"

        return {
            "knowledge_mode": knowledge_mode,
            "taxonomy_resolution": resolution,
            "scopes": scopes,
        }


def main():
    resolver = TaxonomyResolver()
    samples = [
        "笔记本电脑",
        "机械键盘",
        "空气炸锅",
        "洗衣液",
        "桌面真空吸屑器",
        "磁悬浮桌面植物展示器",
    ]
    for sample in samples:
        print("\n" + "=" * 70)
        print("INPUT:", sample)
        result = resolver.resolve(sample)
        print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
