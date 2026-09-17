from __future__ import annotations

import json
import re

import torch

from openai import OpenAI
from PIL import Image

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
)

from system.config import (
    DEEPSEEK_API_KEY,
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    VISION_MODEL_ID,
    VISION_MODEL_REVISION,
)


# ============================================================
# Vision Service
# ============================================================

class VisionService:

    def __init__(self):

        self.tokenizer = None
        self.model = None

        self.device = (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

    def load_model(self):

        if self.model is not None:
            return

        print(
            "\n[Vision] 正在加载 Moondream2..."
        )

        dtype = (
            torch.float16
            if torch.cuda.is_available()
            else torch.float32
        )

        self.model = (
            AutoModelForCausalLM
            .from_pretrained(
                VISION_MODEL_ID,
                revision=VISION_MODEL_REVISION,
                trust_remote_code=True,
                torch_dtype=dtype,
            )
        )

        self.tokenizer = (
            AutoTokenizer
            .from_pretrained(
                VISION_MODEL_ID,
                revision=VISION_MODEL_REVISION,
            )
        )

        if torch.cuda.is_available():

            self.model = self.model.cuda()

        self.model.eval()

        print(
            "[Vision] Moondream2 加载完成，"
            f"device={self.device}"
        )

    def recognize(
        self,
        image: Image.Image,
    ) -> str:

        self.load_model()

        if image is None:

            raise ValueError(
                "图片不能为空"
            )

        if not isinstance(
            image,
            Image.Image,
        ):

            raise TypeError(
                "Vision 输入必须是 PIL.Image.Image，"
                f"当前类型：{type(image)}"
            )

        image = image.convert("RGB")

        prompt = """
Please carefully analyze the main object in this image.

Describe:

1. What the main object is.
2. Whether it appears to be a commercial product.
3. Product category if identifiable.
4. Brand and product name if visible.
5. Main colors.
6. Packaging and appearance.
7. Visible text.
8. Visible features.
9. Possible usage scenarios.

Do not invent attributes that cannot be confirmed from the image.
""".strip()

        try:

            print(
                "[Vision] 正在编码商品图片..."
            )

            with torch.inference_mode():

                encoded_image = (
                    self.model.encode_image(
                        image
                    )
                )

                print(
                    "[Vision] 正在理解商品图片..."
                )

                result = (
                    self.model.answer_question(
                        encoded_image,
                        prompt,
                        self.tokenizer,
                    )
                )

        except Exception as exc:

            raise RuntimeError(
                f"Moondream2 推理失败：{exc}"
            ) from exc

        if result is None:

            raise RuntimeError(
                "Moondream2 没有返回内容"
            )

        result = str(result).strip()

        if not result:

            raise RuntimeError(
                "Moondream2 返回空结果"
            )

        return result


# ============================================================
# LLM Service
# ============================================================

class LLMService:

    def __init__(self):

        if not DEEPSEEK_API_KEY:

            raise RuntimeError(
                "DEEPSEEK_API_KEY 未配置"
            )

        self.client = OpenAI(
            api_key=DEEPSEEK_API_KEY,
            base_url=DEEPSEEK_BASE_URL,
            timeout=60.0,
            max_retries=2,
        )

    # --------------------------------------------------------
    # JSON Helper
    # --------------------------------------------------------

    @staticmethod
    def _parse_json(
        content: str,
    ) -> dict:

        content = content.strip()

        # 去除 ```json
        content = re.sub(
            r"^```(?:json)?",
            "",
            content,
            flags=re.I,
        )

        content = re.sub(
            r"```$",
            "",
            content,
        )

        content = content.strip()

        try:

            return json.loads(content)

        except json.JSONDecodeError:

            # 如果模型前后加了解释，
            # 尝试抽取第一个 JSON Object
            match = re.search(
                r"\{.*\}",
                content,
                flags=re.S,
            )

            if not match:

                raise RuntimeError(
                    "LLM 返回内容无法解析成 JSON：\n"
                    + content
                )

            return json.loads(
                match.group(0)
            )

    # --------------------------------------------------------
    # RAG Query Transformation
    # --------------------------------------------------------

    def build_retrieval_profile(
        self,
        vision_description: str,
    ) -> dict:

        """
        将 Vision 的自由文本描述，
        转换成适合知识库检索的查询。

        这是成熟 RAG 中非常重要的一层：
        Query Transformation。
        """

        system_prompt = """
你是电商 RAG 系统中的 Query Understanding 模块。

你的任务不是写商品文案，而是把视觉模型的原始图片描述，
转换成适合知识库检索的结构化查询。

注意：

视觉模型可能返回英文，也可能返回中文。
你的输出必须使用中文。

你需要判断图片是否存在明确商品主体。

例如：

“一个戴眼镜的人拿着手机的卡通头像”

不应该因为出现手机，
就判断商品类别为“手机”。

如果图片主体本身不是明确商品：

is_product=false

如果是商品，则提炼：

- 商品类别
- 商品名称
- 品牌
- 核心检索关键词
- 一句适合向量检索的中文查询

category 应尽可能使用通用商品类别：

例如：

T恤
马克杯
帆布包
双肩包
运动鞋
洗衣液
洗发水
手机
耳机
键盘
护肤品

不要输出很长的 category。

retrieval_query 应集中描述：

商品是什么 + 核心特征 + 用途

不要把营销语言写进去。

只输出 JSON：

{
    "is_product": true,
    "category": "",
    "product_name": "",
    "brand": "",
    "keywords": [],
    "retrieval_query": "",
    "confidence": 0.0,
    "reasoning": ""
}
""".strip()

        user_prompt = f"""
视觉模型原始描述：

{vision_description}

请生成知识库检索 Profile。
""".strip()

        response = (
            self.client
            .chat
            .completions
            .create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.1,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise RuntimeError(
                "Query Understanding 没有返回结果"
            )

        result = self._parse_json(
            content
        )

        # 默认值
        result.setdefault(
            "is_product",
            False,
        )

        result.setdefault(
            "category",
            "",
        )

        result.setdefault(
            "product_name",
            "",
        )

        result.setdefault(
            "brand",
            "",
        )

        result.setdefault(
            "keywords",
            [],
        )

        result.setdefault(
            "retrieval_query",
            vision_description,
        )

        result.setdefault(
            "confidence",
            0.0,
        )

        result.setdefault(
            "reasoning",
            "",
        )

        return result

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    def generate_product_copy(
        self,
        product_desc: str,
        template_text: str,
        retrieval_profile: dict | None = None,
    ) -> str:

        retrieval_profile = (
            retrieval_profile or {}
        )

        system_prompt = """
你是一名专业电商商品文案专家。

你需要结合：

1. 商品图片识别结果
2. 结构化商品信息
3. RAG 知识库返回的参考知识

生成商品文案。

规则：

1. 图片事实优先级最高。
2. 知识库只是参考，不能覆盖图片事实。
3. 不得因为模板里存在某个卖点，
   就把它强行套用到商品。
4. 不得编造品牌、材质、认证、容量、
   参数、功效。
5. 如果知识库明确说明是“通用模板”，
   只能学习表达结构。
6. 不直接复制模板原文。
7. 输出自然、简洁。

输出：

【商品标题】

【核心卖点】
- ...
- ...
- ...

【商品文案】
...
""".strip()

        profile_json = json.dumps(
            retrieval_profile,
            ensure_ascii=False,
            indent=2,
        )

        user_prompt = f"""
【视觉识别】

{product_desc}


【结构化商品信息】

{profile_json}


【RAG知识库】

{template_text}


请生成最终商品文案。
""".strip()

        response = (
            self.client
            .chat
            .completions
            .create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                ],
                temperature=0.6,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        if not content:

            raise RuntimeError(
                "DeepSeek 没有返回文案"
            )

        return content.strip()


# ============================================================
# Singletons
# ============================================================

vision_service = VisionService()

llm_service = LLMService()