"""多模态模型适配器.

支持图片+文本的混合输入，兼容 OpenAI Vision API 格式。
支持模型：Qwen2-VL、GPT-4V、GLM-4V 等。
"""
from __future__ import annotations

import base64
import os
import time
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter, GenerationResult


def encode_image_to_base64(image_path: str) -> str:
    """将图片编码为 base64 字符串."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def build_image_url(image_path: str, use_base64: bool = True) -> Dict[str, Any]:
    """构建 OpenAI Vision API 格式的 image_url.

    Args:
        image_path: 图片路径或 URL
        use_base64: 是否使用 base64 编码

    Returns:
        image_url 字典
    """
    if use_base64 and os.path.exists(image_path):
        b64 = encode_image_to_base64(image_path)
        ext = os.path.splitext(image_path)[1].lstrip(".").lower()
        mime_map = {"jpg": "jpeg", "jpeg": "jpeg", "png": "png", "gif": "gif", "webp": "webp"}
        mime = mime_map.get(ext, "jpeg")
        return {
            "url": f"data:image/{mime};base64,{b64}"
        }
    else:
        # 直接使用 URL
        return {"url": image_path}


def build_multimodal_message(
    text: str,
    image_paths: Optional[List[str]] = None,
    use_base64: bool = True,
) -> Dict[str, Any]:
    """构建多模态用户消息.

    Args:
        text: 文本内容
        image_paths: 图片路径列表
        use_base64: 是否使用 base64 编码

    Returns:
        OpenAI 格式的 user message
    """
    content: List[Dict[str, Any]] = []

    if image_paths:
        for img_path in image_paths:
            content.append({
                "type": "image_url",
                "image_url": build_image_url(img_path, use_base64),
            })

    content.append({"type": "text", "text": text})

    return {"role": "user", "content": content}


class MultimodalAdapter:
    """多模态模型适配器.

    封装在 OpenAIAdapter 之上，支持图片+文本混合输入。
    使用 OpenAI Vision API 格式（Qwen2-VL/GPT-4V 均兼容）。
    """

    def __init__(self, base_adapter: BaseModelAdapter) -> None:
        """初始化.

        Args:
            base_adapter: 底层文本模型适配器（需支持 chat 方法）
        """
        self._adapter = base_adapter
        self._config = getattr(base_adapter, '_config', None)

    @property
    def model_name(self) -> str:
        return self._config.model_name if self._config else "unknown"

    def chat_with_images(
        self,
        text: str,
        image_paths: Optional[List[str]] = None,
        max_tokens: int = 512,
        temperature: float = 0.0,
        use_base64: bool = True,
    ) -> GenerationResult:
        """多模态对话（图片+文本）.

        Args:
            text: 文本提问
            image_paths: 图片路径列表
            max_tokens: 最大输出 token 数
            temperature: 温度
            use_base64: 是否使用 base64 编码图片

        Returns:
            GenerationResult
        """
        message = build_multimodal_message(text, image_paths, use_base64)

        return self._adapter.chat(
            messages=[message],
            max_tokens=max_tokens,
            temperature=temperature,
        )

    def describe_image(
        self,
        image_path: str,
        prompt: str = "请详细描述这张图片的内容。",
        max_tokens: int = 512,
    ) -> GenerationResult:
        """图片描述.

        Args:
            image_path: 图片路径
            prompt: 提问文本
            max_tokens: 最大输出 token 数

        Returns:
            GenerationResult
        """
        return self.chat_with_images(
            text=prompt,
            image_paths=[image_path],
            max_tokens=max_tokens,
        )

    def vqa(
        self,
        image_path: str,
        question: str,
        max_tokens: int = 256,
    ) -> GenerationResult:
        """视觉问答（Visual Question Answering）.

        Args:
            image_path: 图片路径
            question: 问题
            max_tokens: 最大输出 token 数

        Returns:
            GenerationResult
        """
        return self.chat_with_images(
            text=question,
            image_paths=[image_path],
            max_tokens=max_tokens,
        )
