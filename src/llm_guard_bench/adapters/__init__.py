"""模型接入层 - 统一 Adapter 接口.

支持多种模型后端：
    - OpenAI API (GPT-4, GPT-3.5 等)
    - HuggingFace Transformers
    - vLLM 推理服务
    - 多模态模型扩展接口

使用方式：
    from llm_guard_bench.adapters import build_adapter, load_model_config

    config = load_model_config("qwen-turbo")
    adapter = build_adapter(config)
    response = adapter.generate(prompt="你好", max_tokens=128)
"""

from llm_guard_bench.adapters.base import (
    BaseModelAdapter,
    GenerationResult,
    ModelInfo,
)
from llm_guard_bench.adapters.factory import (
    ModelConfig,
    build_adapter,
    get_supported_providers,
    load_model_config,
)

__all__ = [
    "BaseModelAdapter",
    "GenerationResult",
    "ModelInfo",
    "ModelConfig",
    "build_adapter",
    "get_supported_providers",
    "load_model_config",
]
