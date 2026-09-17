from langgraph.graph import (
    StateGraph,
    START,
    END,
)

from app.schemas import AgentState

from app.services import (
    vision_service,
    llm_service,
)

from rag.rag import (
    template_store,
)


# ============================================================
# Node 1 - Vision
# ============================================================

def recognize_node(
    state: AgentState,
):

    print(
        "\n[Agent] Vision"
    )

    image = state.get(
        "image"
    )

    if image is None:

        return {
            "error": "没有收到图片"
        }

    try:

        product_desc = (
            vision_service
            .recognize(
                image
            )
        )

        print(
            "\n[Vision Result]"
        )

        print(
            product_desc
        )

        return {
            "product_desc":
                product_desc
        }

    except Exception as exc:

        return {
            "error":
                f"Vision失败：{exc}"
        }


# ============================================================
# Node 2 - Query Transformation
# ============================================================

def query_understanding_node(
    state: AgentState,
):

    if state.get("error"):
        return {}

    print(
        "\n[Agent] Query Understanding"
    )

    product_desc = state.get(
        "product_desc",
        "",
    )

    try:

        profile = (
            llm_service
            .build_retrieval_profile(
                product_desc
            )
        )

        print(
            "\n[Retrieval Profile]"
        )

        print(profile)

        return {
            "retrieval_profile":
                profile
        }

    except Exception as exc:

        return {
            "error":
                f"Query Understanding失败："
                f"{exc}"
        }


# ============================================================
# Router
# ============================================================

def product_router(
    state: AgentState,
):

    if state.get("error"):

        return "finish"

    profile = state.get(
        "retrieval_profile",
        {},
    )

    if not profile.get(
        "is_product",
        False,
    ):

        return "not_product"

    return "retrieve"


# ============================================================
# Node 3 - RAG
# ============================================================

def retrieve_node(
    state: AgentState,
):

    if state.get("error"):
        return {}

    print(
        "\n[Agent] RAG Retrieval"
    )

    profile = state.get(
        "retrieval_profile",
        {},
    )

    try:

        result = (
            template_store.search(
                profile
            )
        )

        print(
            "\n[RAG Summary]"
        )

        print(
            "found=",
            result.get(
                "found",
                False,
            )
        )

        print(
            "mode=",
            result.get(
                "mode",
                "unknown",
            )
        )

        print(
            "confidence=",
            result.get(
                "confidence_level",
                "unknown",
            ),
            result.get(
                "confidence_score",
                result.get(
                    "score",
                    0.0,
                ),
            ),
        )

        taxonomy_resolution = (
            result.get(
                "taxonomy_resolution",
                {},
            )
            or {}
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

        print(
            "taxonomy_path=",
            taxonomy_path,
        )

        print(
            "knowledge_mode=",
            result.get(
                "knowledge_mode",
                "unknown",
            ),
        )

        print(
            "\n[RAG Debug]"
        )

        print(
            result.get(
                "debug",
                "",
            )
        )

        if result["found"]:

            context = (
                result["text"]
            )

        else:

            context = (
                "知识库没有找到满足相关性要求的"
                "商品知识。"
                "请严格依据图片真实信息生成，"
                "不要自行补充商品属性。"
            )

        return {
            "template_text":
                context,

            "rag_found":
                result["found"],

            "rag_mode":
                result["mode"],

            "rag_score":
                result["score"],

            "rag_debug":
                result["debug"],
        }

    except Exception as exc:

        return {
            "error":
                f"RAG检索失败：{exc}"
        }


# ============================================================
# Node 4 - Generation
# ============================================================

def compose_node(
    state: AgentState,
):

    if state.get("error"):
        return {}

    print(
        "\n[Agent] Generation"
    )

    try:

        answer = (
            llm_service
            .generate_product_copy(
                product_desc=(
                    state.get(
                        "product_desc",
                        "",
                    )
                ),

                template_text=(
                    state.get(
                        "template_text",
                        "",
                    )
                ),

                retrieval_profile=(
                    state.get(
                        "retrieval_profile",
                        {},
                    )
                ),
            )
        )

        print(
            "\n[Final Answer]"
        )

        print(answer)

        return {
            "final_answer":
                answer
        }

    except Exception as exc:

        return {
            "error":
                f"文案生成失败：{exc}"
        }


# ============================================================
# Non Product
# ============================================================

def not_product_node(
    state: AgentState,
):

    profile = state.get(
        "retrieval_profile",
        {},
    )

    desc = state.get(
        "product_desc",
        "",
    )

    reason = profile.get(
        "reasoning",
        "",
    )

    answer = (
        "当前图片未识别到足够明确的商品主体，"
        "因此没有进入商品知识库检索。\n\n"

        f"【视觉结果】\n{desc}\n\n"

        f"【判断原因】\n{reason}"
    )

    return {
        "final_answer":
            answer,

        "rag_found":
            False,

        "rag_mode":
            "not_product",

        "rag_score":
            0.0,

        "rag_debug":
            "非商品图片，不执行 RAG。",
    }


# ============================================================
# Graph
# ============================================================

def build_graph():

    workflow = (
        StateGraph(
            AgentState
        )
    )

    workflow.add_node(
        "recognize",
        recognize_node,
    )

    workflow.add_node(
        "query_understanding",
        query_understanding_node,
    )

    workflow.add_node(
        "retrieve",
        retrieve_node,
    )

    workflow.add_node(
        "compose",
        compose_node,
    )

    workflow.add_node(
        "not_product",
        not_product_node,
    )

    workflow.add_edge(
        START,
        "recognize",
    )

    workflow.add_edge(
        "recognize",
        "query_understanding",
    )

    workflow.add_conditional_edges(
        "query_understanding",

        product_router,

        {
            "retrieve":
                "retrieve",

            "not_product":
                "not_product",

            "finish":
                END,
        },
    )

    workflow.add_edge(
        "retrieve",
        "compose",
    )

    workflow.add_edge(
        "compose",
        END,
    )

    workflow.add_edge(
        "not_product",
        END,
    )

    return workflow.compile()


graph = build_graph()


# ============================================================
# API
# ============================================================

def run_agent(
    image,
):

    if image is None:

        return {
            "error":
                "请上传商品图片"
        }

    return graph.invoke(
        {
            "image": image
        }
    )