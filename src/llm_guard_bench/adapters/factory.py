"""适配器工厂 - 根据配置构建模型适配器.

使用方式：
    config = load_model_config("configs/models/qwen-turbo.yaml")
    adapter = build_adapter(config)
    result = adapter.generate("你好")
"""
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Type

from dotenv import load_dotenv

from llm_guard_bench.adapters.base import BaseModelAdapter

# 导入适配器实现，触发 @register 装饰器注册
import llm_guard_bench.adapters.openai as _openai  # noqa: F401
import llm_guard_bench.adapters.huggingface as _huggingface  # noqa: F401

load_dotenv()


class ModelConfig:
    """统一模型配置类.

    从 YAML 文件加载配置，自动解析不同 provider 的配置格式。
    """

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

        self.device = config_dict.get("device", "auto")
        self.torch_dtype = config_dict.get("torch_dtype", "bfloat16")
        self.quantization = config_dict.get("quantization", None)

        self.vllm_config = config_dict.get("vllm", {})

    def get_api_key(self) -> Optional[str]:
        if self.api_key_env:
            return os.getenv(self.api_key_env)
        return None

    def get_base_url(self) -> Optional[str]:
        if self.base_url_env:
            return os.getenv(self.base_url_env)
        return None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "ModelConfig":
        """从 YAML 文件加载配置."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))


def load_model_config(name: str) -> ModelConfig:
    """加载模型配置.

    Args:
        name: 模型名称（对应 configs/models/<name>.yaml）

    Returns:
        ModelConfig: 模型配置
    """
    from llm_guard_bench.constants import CONFIGS_DIR

    config_path = CONFIGS_DIR / "models" / f"{name}.yaml"
    if not config_path.exists():
        raise ValueError(f"模型配置不存在: {config_path}")
    return ModelConfig.from_yaml(str(config_path))


def build_adapter(config: ModelConfig) -> BaseModelAdapter:
    """构建模型适配器.

    根据 provider 字段选择对应的适配器实现。

    Args:
        config: 模型配置

    Returns:
        BaseModelAdapter: 模型适配器实例
    """
    adapter_cls = BaseModelAdapter.get_adapter(config.provider)

    if adapter_cls is None:
        supported = ", ".join(BaseModelAdapter._registry.keys())
        raise ValueError(
            f"不支持的 provider: {config.provider}. "
            f"当前支持: {supported}"
        )

    return adapter_cls(config)


def get_supported_providers() -> list[str]:
    """获取支持的模型后端列表."""
    return list(BaseModelAdapter._registry.keys())
