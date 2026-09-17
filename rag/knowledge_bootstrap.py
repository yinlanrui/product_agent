from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings

from system.compute_device import (
    get_embedding_device,
)

from system.config import (
    EMBED_MODEL_NAME,
    VECTOR_DB_PATH,
    CANONICAL_ATTRIBUTE_FILE,
    GENERATED_ATTRIBUTE_FILE,
    KNOWLEDGE_MANIFEST_FILE,
)


ATTRIBUTE_COLLECTION_NAME = "product_attribute_knowledge"

# Compatibility aliases keep the rest of this module simple while all
# actual filesystem locations are owned by system.config.
CANONICAL_FILE = CANONICAL_ATTRIBUTE_FILE
GENERATED_FILE = GENERATED_ATTRIBUTE_FILE
MANIFEST_FILE = KNOWLEDGE_MANIFEST_FILE


def _sha256(
    path: Path,
) -> str:

    digest = hashlib.sha256()

    with path.open("rb") as handle:

        for chunk in iter(
            lambda:
                handle.read(
                    1024 * 1024
                ),
            b"",
        ):
            digest.update(
                chunk
            )

    return digest.hexdigest()


def _load_json(
    path: Path,
) -> dict[str, Any]:

    return json.loads(
        path.read_text(
            encoding="utf-8"
        )
    )


# ============================================================
# Canonical Property -> Generated Attribute Knowledge
# ============================================================

def _format_sources(
    mappings: list[dict],
) -> list[dict]:

    output = []

    for item in mappings:

        mapped = {
            "source_id":
                item.get(
                    "source_id",
                    "",
                ),

            "source_property_id":
                item.get(
                    "source_property_id",
                    "",
                ),

            "source_property_name":
                item.get(
                    "source_property_name",
                    "",
                ),

            "mapping_type":
                item.get(
                    "mapping_type",
                    "",
                ),

            "mapping_confidence":
                item.get(
                    "mapping_confidence",
                    0.0,
                ),
        }

        if item.get(
            "source_version"
        ):
            mapped[
                "source_version"
            ] = item[
                "source_version"
            ]

        output.append(
            mapped
        )

    return output


def _build_document(
    prop: dict,
) -> dict:

    property_id = prop[
        "canonical_property_id"
    ]

    zh = prop[
        "canonical_name_zh"
    ]

    en = prop[
        "canonical_name_en"
    ]

    family = prop[
        "family_id"
    ]

    evidence = (
        prop.get(
            "evidence_policy",
            {},
        )
        or {}
    )

    assertable = "、".join(
        evidence.get(
            "assertable_from",
            [],
        )
    ) or "无"

    retrieval_only = "、".join(
        evidence.get(
            "retrieval_only_from",
            [],
        )
    ) or "无"

    unit_rule = (
        prop.get(
            "unit"
        )
        or "无固定单位"
    )

    content = (
        f"标准属性：{zh}（{en}）。"
        f"定义：{prop.get('definition', '')} "
        f"值类型：{prop.get('value_type', 'text')}；"
        f"单位规则：{unit_rule}。"
        f"允许作为最终文案事实的证据：{assertable}。"
        f"仅可用于检索、不能直接写成事实的弱证据：{retrieval_only}。"
        "生成文案时不得因为检索到此属性知识，"
        "就假定当前商品一定拥有该属性；"
        "只有当前图片、可读文字或可信结构化数据实际支持时，"
        "才可写入具体属性值。"
    )

    return {
        "document_id":
            "canonical-"
            + property_id.replace(
                ".",
                "-",
            ),

        "title":
            f"标准属性：{zh}",

        "document_type":
            "canonical_attribute_guide",

        "scope_type":
            "attribute_property",

        "attribute_family":
            family,

        "canonical_property_id":
            property_id,

        "attribute_names": [
            en,
            zh,
        ],

        "rule_origin":
            "generated_from_canonical_property_layer",

        "source_basis":
            _format_sources(
                prop.get(
                    "source_mappings",
                    [],
                )
            ),

        "retrieval_terms": [
            zh,
            en,
            property_id,
        ],

        "content":
            content,

        "generation_policy":
            {
                "allowed_evidence":
                    evidence.get(
                        "assertable_from",
                        [],
                    ),

                "retrieval_only_evidence":
                    evidence.get(
                        "retrieval_only_from",
                        [],
                    ),
            },

        "active":
            True,

        "version":
            "0.4.1",
    }


def generate_attribute_knowledge() -> int:

    if not CANONICAL_FILE.exists():

        raise FileNotFoundError(
            "Missing canonical attribute source: "
            f"{CANONICAL_FILE}"
        )

    data = _load_json(
        CANONICAL_FILE
    )

    properties = data.get(
        "properties",
        [],
    )

    if not properties:

        raise RuntimeError(
            "canonical_attribute_properties.json "
            "contains no properties."
        )

    documents = [
        _build_document(
            prop
        )
        for prop in properties
    ]

    payload = {
        "meta": {
            "name":
                "Generated Attribute Knowledge",

            "version":
                "0.4.1",

            "source":
                "canonical_attribute_properties.json",

            "note":
                (
                    "Generated file. "
                    "Do not hand-edit."
                ),
        },

        "documents":
            documents,
    }

    GENERATED_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return len(
        documents
    )


# ============================================================
# Generated Knowledge -> Chroma
# ============================================================

def _load_generated_documents() -> tuple[
    list[Document],
    list[str],
]:

    if not GENERATED_FILE.exists():

        raise FileNotFoundError(
            "Missing generated knowledge: "
            f"{GENERATED_FILE}"
        )

    data = _load_json(
        GENERATED_FILE
    )

    source_docs = data.get(
        "documents",
        [],
    )

    if not source_docs:

        raise RuntimeError(
            "Generated attribute knowledge is empty."
        )

    documents = []
    ids = []
    seen = set()

    for item in source_docs:

        document_id = str(
            item.get(
                "document_id",
                "",
            )
        ).strip()

        if not document_id:

            raise RuntimeError(
                "Generated knowledge contains "
                "an empty document_id."
            )

        if document_id in seen:

            raise RuntimeError(
                "Duplicate document_id: "
                f"{document_id}"
            )

        seen.add(
            document_id
        )

        if not item.get(
            "active",
            True,
        ):
            continue

        attribute_names = " ".join(
            str(value).strip()
            for value
            in item.get(
                "attribute_names",
                [],
            )
            if str(value).strip()
        )

        retrieval_terms = " ".join(
            str(value).strip()
            for value
            in item.get(
                "retrieval_terms",
                [],
            )
            if str(value).strip()
        )

        generation_policy = (
            item.get(
                "generation_policy",
                {},
            )
            or {}
        )

        allowed_evidence = " ".join(
            generation_policy.get(
                "allowed_evidence",
                [],
            )
        )

        retrieval_only = " ".join(
            generation_policy.get(
                "retrieval_only_evidence",
                [],
            )
        )

        canonical_property_id = str(
            item.get(
                "canonical_property_id",
                "",
            )
            or ""
        )

        page_content = (
            "passage: "
            f"标题：{item.get('title', '')}\n"
            f"属性族：{item.get('attribute_family', '')}\n"
            f"标准属性ID：{canonical_property_id}\n"
            f"属性：{attribute_names}\n"
            f"检索词：{retrieval_terms}\n"
            f"允许事实证据：{allowed_evidence}\n"
            f"仅检索证据：{retrieval_only}\n"
            f"正文：{item.get('content', '')}"
        )

        metadata = {
            "document_id":
                document_id,

            "title":
                item.get(
                    "title",
                    "",
                ),

            "document_type":
                item.get(
                    "document_type",
                    "",
                ),

            "scope_type":
                item.get(
                    "scope_type",
                    "attribute_property",
                ),

            "attribute_family":
                item.get(
                    "attribute_family",
                    "",
                ),

            "attribute_names":
                attribute_names,

            "retrieval_terms":
                retrieval_terms,

            "rule_origin":
                item.get(
                    "rule_origin",
                    "",
                ),

            "canonical_property_id":
                canonical_property_id,

            "version":
                str(
                    item.get(
                        "version",
                        "0.4.1",
                    )
                ),

            "active":
                True,
        }

        documents.append(
            Document(
                page_content=(
                    page_content
                ),
                metadata=metadata,
            )
        )

        ids.append(
            document_id
        )

    return (
        documents,
        ids,
    )


def _new_embeddings():

    return (
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


def rebuild_attribute_index(
    *,
    embeddings=None,
) -> int:

    documents, ids = (
        _load_generated_documents()
    )

    if embeddings is None:

        embeddings = (
            _new_embeddings()
        )

    db = Chroma(
        collection_name=(
            ATTRIBUTE_COLLECTION_NAME
        ),
        embedding_function=(
            embeddings
        ),
        persist_directory=str(
            VECTOR_DB_PATH
        ),
        collection_metadata={
            "hnsw:space":
                "cosine",
        },
    )

    try:

        db.delete_collection()

    except Exception:

        pass

    db = Chroma(
        collection_name=(
            ATTRIBUTE_COLLECTION_NAME
        ),
        embedding_function=(
            embeddings
        ),
        persist_directory=str(
            VECTOR_DB_PATH
        ),
        collection_metadata={
            "hnsw:space":
                "cosine",
        },
    )

    db.add_documents(
        documents=documents,
        ids=ids,
    )

    count = len(
        db.get().get(
            "ids",
            [],
        )
    )

    if count != len(
        documents
    ):

        raise RuntimeError(
            "Chroma rebuild count mismatch: "
            f"expected={len(documents)}, "
            f"actual={count}"
        )

    return count


# ============================================================
# Manifest / Staleness
# ============================================================

def _read_manifest() -> dict:

    if not MANIFEST_FILE.exists():

        return {}

    try:

        return _load_json(
            MANIFEST_FILE
        )

    except Exception:

        return {}


def _write_manifest(
    *,
    document_count: int,
) -> None:

    payload = {
        "version":
            "0.4.1",

        "canonical_sha256":
            _sha256(
                CANONICAL_FILE
            ),

        "generated_sha256":
            _sha256(
                GENERATED_FILE
            ),

        "document_count":
            int(
                document_count
            ),

        "collection_name":
            ATTRIBUTE_COLLECTION_NAME,
    }

    MANIFEST_FILE.write_text(
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )


def _attribute_collection_count() -> int:

    try:

        db = Chroma(
            collection_name=(
                ATTRIBUTE_COLLECTION_NAME
            ),
            persist_directory=str(
                VECTOR_DB_PATH
            ),
        )

        return len(
            db.get().get(
                "ids",
                [],
            )
        )

    except Exception:

        return 0


def knowledge_status() -> dict[str, Any]:

    manifest = (
        _read_manifest()
    )

    canonical_exists = (
        CANONICAL_FILE.exists()
    )

    generated_exists = (
        GENERATED_FILE.exists()
    )

    db_count = (
        _attribute_collection_count()
    )

    canonical_hash = (
        _sha256(
            CANONICAL_FILE
        )
        if canonical_exists
        else ""
    )

    generated_hash = (
        _sha256(
            GENERATED_FILE
        )
        if generated_exists
        else ""
    )

    expected_count = int(
        manifest.get(
            "document_count",
            0,
        )
        or 0
    )

    stale = (
        not canonical_exists
        or not generated_exists
        or not manifest
        or (
            canonical_hash
            != manifest.get(
                "canonical_sha256",
                "",
            )
        )
        or (
            generated_hash
            != manifest.get(
                "generated_sha256",
                "",
            )
        )
        or db_count <= 0
        or (
            expected_count > 0
            and db_count
            != expected_count
        )
    )

    return {
        "ready":
            not stale,

        "stale":
            stale,

        "canonical_exists":
            canonical_exists,

        "generated_exists":
            generated_exists,

        "db_count":
            db_count,

        "expected_count":
            expected_count,

        "manifest_exists":
            bool(
                manifest
            ),
    }


# ============================================================
# Public API
# ============================================================

def rebuild_knowledge(
    *,
    embeddings=None,
    verbose: bool = True,
) -> dict[str, Any]:

    if verbose:

        print(
            "\n[KNOWLEDGE] Rebuilding "
            "canonical attribute knowledge..."
        )

    generated_count = (
        generate_attribute_knowledge()
    )

    indexed_count = (
        rebuild_attribute_index(
            embeddings=embeddings
        )
    )

    _write_manifest(
        document_count=(
            indexed_count
        )
    )

    if verbose:

        print(
            "[KNOWLEDGE] Generated docs:",
            generated_count,
        )

        print(
            "[KNOWLEDGE] Indexed docs:",
            indexed_count,
        )

        print(
            "[KNOWLEDGE] READY"
        )

    return {
        "generated_count":
            generated_count,

        "indexed_count":
            indexed_count,

        "ready":
            True,
    }


def ensure_knowledge_ready(
    *,
    embeddings=None,
    verbose: bool = True,
) -> dict[str, Any]:

    status = (
        knowledge_status()
    )

    if status[
        "ready"
    ]:

        if verbose:

            print(
                "[KNOWLEDGE] Existing index "
                "is current."
            )

        return status

    if verbose:

        print(
            "[KNOWLEDGE] Missing or stale "
            "knowledge index detected."
        )

    rebuild_knowledge(
        embeddings=embeddings,
        verbose=verbose,
    )

    return (
        knowledge_status()
    )
