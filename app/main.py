import json

import gradio as gr

from system.config import (
    GRADIO_HOST,
    GRADIO_PORT,
    validate_settings,
)

from app.product_agent import (
    run_agent,
)


def gradio_run(
    image,
):

    if image is None:

        return (
            "请上传图片",
            "",
            "",
            "",
            "",
        )

    result = run_agent(
        image
    )

    if result.get(
        "error"
    ):

        return (
            result["error"],
            "",
            "",
            "",
            "",
        )

    product_desc = (
        result.get(
            "product_desc",
            "",
        )
    )

    profile = (
        result.get(
            "retrieval_profile",
            {},
        )
    )

    profile_text = json.dumps(
        profile,
        ensure_ascii=False,
        indent=2,
    )

    rag_context = (
        result.get(
            "template_text",
            "",
        )
    )

    rag_debug = (
        result.get(
            "rag_debug",
            "",
        )
    )

    answer = (
        result.get(
            "final_answer",
            "",
        )
    )

    return (
        product_desc,
        profile_text,
        rag_context,
        rag_debug,
        answer,
    )


def create_demo():

    with gr.Blocks(
        title=(
            "Product Agent RAG V0.3"
        )
    ) as demo:

        gr.Markdown(
            """
# AI 商品运营 Agent — RAG V0.3

当前重点：

**Vision → Query Transformation → Metadata Filter → Vector Retrieval → Hybrid Rerank → Threshold → Generation**
"""
        )

        with gr.Row():

            with gr.Column(
                scale=1
            ):

                image_input = (
                    gr.Image(
                        type="pil",
                        label="商品图片",
                        height=420,
                    )
                )

                run_button = (
                    gr.Button(
                        "运行 Agent",
                        variant="primary",
                    )
                )

            with gr.Column(
                scale=2
            ):

                vision_output = (
                    gr.Textbox(
                        label=(
                            "① Vision 原始识别"
                        ),
                        lines=6,
                    )
                )

                profile_output = (
                    gr.Textbox(
                        label=(
                            "② Retrieval Profile"
                        ),
                        lines=10,
                    )
                )

                rag_output = (
                    gr.Textbox(
                        label=(
                            "③ RAG 最终上下文"
                        ),
                        lines=12,
                    )
                )

                debug_output = (
                    gr.Textbox(
                        label=(
                            "④ RAG Debug"
                        ),
                        lines=10,
                    )
                )

                answer_output = (
                    gr.Textbox(
                        label=(
                            "⑤ 最终文案"
                        ),
                        lines=12,
                    )
                )

        run_button.click(

            fn=gradio_run,

            inputs=[
                image_input
            ],

            outputs=[
                vision_output,
                profile_output,
                rag_output,
                debug_output,
                answer_output,
            ],
        )

    return demo


if __name__ == "__main__":

    validate_settings()

    print(
        "\n启动 Product Agent RAG V0.3..."
    )

    demo = create_demo()

    demo.launch(
        server_name=GRADIO_HOST,
        server_port=GRADIO_PORT,
    )