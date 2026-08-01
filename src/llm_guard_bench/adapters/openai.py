"""OpenAI API 适配器.

支持：
    - 原生 OpenAI API
    - 通义千问 DashScope（OpenAI 兼容模式）
    - Azure OpenAI
    - 其他兼容 OpenAI API 格式的服务

使用方式：
    config = OpenAIConfig.from_yaml("configs/models/qwen-turbo.yaml")
    adapter = OpenAIAdapter(config)
    result = adapter.generate("你好")
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

from llm_guard_bench.adapters.base import (
    BaseModelAdapter,
    GenerationResult,
    ModelConfigProtocol,
)

load_dotenv()


class OpenAIConfig:
    """OpenAI 模型配置."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.name = config_dict["name"]
        self.provider = config_dict["provider"]
        self.model_name = config_dict["model_name"]
        self.temperature = config_dict.get("temperature", 0.0)
        self.max_tokens = config_dict.get("max_tokens", 1024)
        self.timeout = config_dict.get("timeout", 60)

        self.api_key_env = config_dict.get("api_key_env")
        self.base_url_env = config_dict.get("base_url_env")

        self.retry_config = config_dict.get("retry", {})
        self.metadata = config_dict.get("metadata", {})

    def get_api_key(self) -> Optional[str]:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return os.getenv("OPENAI_API_KEY")

    def get_base_url(self) -> Optional[str]:
        if self.base_url_env:
            return os.getenv(self.base_url_env)
        return os.getenv("OPENAI_BASE_URL")

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "OpenAIConfig":
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))


@BaseModelAdapter.register("openai")
class OpenAIAdapter(BaseModelAdapter):
    """OpenAI API 适配器.

    通过 openai Python SDK 调用，支持 OpenAI 兼容的各种服务。
    """

    def __init__(self, config: OpenAIConfig) -> None:
        super().__init__(config)
        self._config = config
        self._client = None
        self._init_client()

    def _init_client(self) -> None:
        from openai import OpenAI

        api_key = self._config.get_api_key()
        base_url = self._config.get_base_url()

        if not api_key:
            raise ValueError("API key not found. Set OPENAI_API_KEY or api_key_env in config.")

        self._client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            timeout=self._config.timeout,
        )

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """生成响应.

        优先使用 Chat API（兼容性更好），如果失败则回退到 Completion API。
        """
        # 转发到 chat 接口（DashScope 等兼容 API 更稳定）
        messages = [{"role": "user", "content": prompt}]
        result = self.chat(messages, max_tokens=max_tokens, temperature=temperature, timeout=timeout, **kwargs)
        return result

    def chat(
        self,
        messages: List[Dict[str, str]],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        start_time = time.time()

        try:
            effective_max_tokens = max_tokens if max_tokens is not None else self._config.max_tokens
            effective_temperature = temperature if temperature is not None else self._config.temperature

            response = self._client.chat.completions.create(
                model=self._config.model_name,
                messages=messages,
                max_tokens=effective_max_tokens,
                temperature=effective_temperature,
                **kwargs,
            )

            latency_ms = self._record_latency(start_time)
            completion = response.choices[0]

            return GenerationResult(
                text=completion.message.content.strip() if completion.message.content else "",
                prompt_tokens=response.usage.prompt_tokens,
                completion_tokens=response.usage.completion_tokens,
                total_tokens=response.usage.total_tokens,
                latency_ms=latency_ms,
                finish_reason=completion.finish_reason,
                raw_response=response.model_dump(),
            )

        except Exception as e:
            latency_ms = self._record_latency(start_time)
            return GenerationResult(
                text="",
                latency_ms=latency_ms,
                error=str(e),
            )

    def get_model_info(self) -> Any:
        info = super().get_model_info()
        info.context_window = self._config.metadata.get("context_window")
        info.vendor = self._config.metadata.get("vendor")
        info.tags = self._config.metadata.get("tags", [])
        return info
