"""评测引擎测试."""
from __future__ import annotations

from llm_guard_bench.engine import EvaluationConfig, DatasetConfig, load_dataset_config


def test_load_evaluation_config() -> None:
    """测试加载评测任务配置."""
    config = EvaluationConfig.from_yaml("configs/evaluation/capability.yaml")
    assert config.name == "capability_eval_v1"
    assert "qwen-turbo" in config.models
    assert "mmlu" in config.datasets
    assert not config.execution.parallel


def test_load_dataset_config() -> None:
    """测试加载数据集配置."""
    config = load_dataset_config("mmlu")
    assert config.name == "mmlu"
    assert config.source_type == "huggingface"
    assert config.source_path == "cais/mmlu"
    assert config.eval_method == "multiple_choice"


def test_dataset_config_from_yaml() -> None:
    """测试从 YAML 加载数据集配置."""
    from llm_guard_bench.constants import CONFIGS_DIR

    yaml_path = CONFIGS_DIR / "datasets" / "mmlu.yaml"
    config = DatasetConfig.from_yaml(str(yaml_path))
    assert config.name == "mmlu"
    assert config.judge_method == "exact_match"
