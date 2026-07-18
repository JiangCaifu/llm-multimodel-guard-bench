"""模型接入层测试."""
from __future__ import annotations

from llm_guard_bench.adapters import (
    GenerationResult,
    ModelConfig,
    build_adapter,
    load_model_config,
)


def test_load_model_config() -> None:
    """测试加载模型配置."""
    config = load_model_config("qwen-turbo")
    assert config.name == "qwen-turbo"
    assert config.provider == "openai"
    assert config.model_name == "qwen-turbo"
    assert config.temperature == 0.0


def test_model_config_from_yaml() -> None:
    """测试从 YAML 加载配置."""
    from llm_guard_bench.constants import CONFIGS_DIR

    yaml_path = CONFIGS_DIR / "models" / "qwen-turbo.yaml"
    config = ModelConfig.from_yaml(str(yaml_path))
    assert config.name == "qwen-turbo"
    assert config.provider == "openai"


def test_build_adapter_openai() -> None:
    """测试构建 OpenAI 适配器."""
    config = load_model_config("qwen-turbo")
    adapter = build_adapter(config)
    
    assert adapter is not None
    assert adapter.get_model_info().name == "qwen-turbo"
    assert adapter.get_model_info().provider == "openai"


def test_adapter_registered_providers() -> None:
    """测试已注册的适配器类型."""
    from llm_guard_bench.adapters.base import BaseModelAdapter

    providers = list(BaseModelAdapter._registry.keys())
    assert "openai" in providers
    assert "huggingface" in providers
