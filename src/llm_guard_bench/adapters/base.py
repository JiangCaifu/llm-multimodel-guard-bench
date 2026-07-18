"""统一模型适配器基类.

所有模型后端（OpenAI/HuggingFace/vLLM/多模态）都必须继承此类，
确保评测引擎可以统一调用。

使用方式：
    class MyModelAdapter(BaseModelAdapter):
        def generate(self, prompt: str, **kwargs) -> GenerationResult:
            # 实现模型调用逻辑
            pass

    adapter = MyModelAdapter(config)
    result = adapter.generate("你好", max_tokens=128)
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, Type, TypeVar, runtime_checkable


@runtime_checkable
class ModelConfigProtocol(Protocol):
    name: str
    provider: str
    model_name: str
    temperature: float
    max_tokens: int
    timeout: int

    def get_api_key(self) -> Optional[str]:
        ...

    def get_base_url(self) -> Optional[str]:
        ...


@dataclass
class GenerationResult:
    """模型生成结果.

    统一封装不同后端的返回格式，便于评测引擎处理。
    """
    text: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    latency_ms: float = 0.0
    finish_reason: Optional[str] = None
    raw_response: Optional[Dict[str, Any]] = None
    error: Optional[str] = None

    @property
    def success(self) -> bool:
        return self.error is None


@dataclass
class BatchGenerationResult:
    """批量生成结果."""
    results: List[GenerationResult]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_latency_ms: float = 0.0


@dataclass
class ModelInfo:
    """模型元信息."""
    name: str
    provider: str
    model_name: str
    context_window: Optional[int] = None
    vendor: Optional[str] = None
    tags: List[str] = field(default_factory=list)


AdapterType = TypeVar("AdapterType", bound="BaseModelAdapter")


class BaseModelAdapter:
    """模型适配器基类.

    定义统一接口：
        - generate: 生成响应（核心方法）
        - batch_generate: 批量生成
        - chat: 多轮对话
        - get_model_info: 获取模型信息
        - close: 释放资源
    """

    _registry: Dict[str, Type[BaseModelAdapter]] = {}

    def __init__(self, config: ModelConfigProtocol) -> None:
        self.config = config
        self._client = None

    @classmethod
    def register(cls, provider: str) -> Type[AdapterType]:
        """注册适配器到工厂.

        使用方式：
            @BaseModelAdapter.register("openai")
            class OpenAIAdapter(BaseModelAdapter):
                pass
        """
        def decorator(adapter_cls: Type[AdapterType]) -> Type[AdapterType]:
            cls._registry[provider] = adapter_cls
            return adapter_cls
        return decorator

    @classmethod
    def get_adapter(cls, provider: str) -> Optional[Type[BaseModelAdapter]]:
        return cls._registry.get(provider)

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        """生成响应.

        Args:
            prompt: 输入提示词
            max_tokens: 最大输出token数（覆盖配置）
            temperature: 温度（覆盖配置）
            timeout: 超时时间（覆盖配置）
            **kwargs: 其他参数

        Returns:
            GenerationResult: 生成结果
        """
        raise NotImplementedError("子类必须实现 generate 方法")

    def batch_generate(
        self,
        prompts: List[str],
        **kwargs: Any,
    ) -> BatchGenerationResult:
        """批量生成响应.

        默认实现为串行调用，子类可优化为并行。
        """
        results: List[GenerationResult] = []
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_latency_ms = 0.0

        for prompt in prompts:
            result = self.generate(prompt, **kwargs)
            results.append(result)
            total_prompt_tokens += result.prompt_tokens
            total_completion_tokens += result.completion_tokens
            total_latency_ms += result.latency_ms

        return BatchGenerationResult(
            results=results,
            total_prompt_tokens=total_prompt_tokens,
            total_completion_tokens=total_completion_tokens,
            total_latency_ms=total_latency_ms,
        )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> GenerationResult:
        """多轮对话.

        Args:
            messages: 消息列表，格式: [{"role": "user", "content": "..."}, ...]

        Returns:
            GenerationResult: 生成结果
        """
        raise NotImplementedError("子类必须实现 chat 方法")

    def get_model_info(self) -> ModelInfo:
        """获取模型元信息."""
        return ModelInfo(
            name=self.config.name,
            provider=self.config.provider,
            model_name=self.config.model_name,
        )

    def close(self) -> None:
        """释放资源."""
        pass

    def __enter__(self) -> BaseModelAdapter:
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        self.close()

    def _get_effective_params(
        self,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ) -> Tuple[int, float, int]:
        """获取有效参数（优先使用传入值，否则使用配置值）."""
        return (
            max_tokens or self.config.max_tokens,
            temperature if temperature is not None else self.config.temperature,
            timeout or self.config.timeout,
        )

    def _record_latency(self, start_time: float) -> float:
        """记录延迟（毫秒）."""
        return (time.time() - start_time) * 1000
