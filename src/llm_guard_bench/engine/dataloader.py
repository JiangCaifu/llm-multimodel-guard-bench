"""数据集加载器 - 支持多种数据源.

支持：
    - HuggingFace Datasets
    - 本地 JSON/CSV 文件
    - 自定义数据集

使用方式：
    loader = DatasetLoader("configs/datasets/mmlu.yaml")
    dataset = loader.load()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Protocol


@dataclass
class DatasetConfig:
    """数据集配置."""
    name: str
    version: str = "1.0.0"

    source_type: str = "huggingface"
    source_path: str = ""
    subset: Optional[str] = None
    split: str = "test"

    eval_method: str = "multiple_choice"
    sample_size: Optional[int] = None
    random_seed: int = 42
    n_shot: int = 5

    prompt_template: str = ""
    judge_method: str = "exact_match"
    answer_field: str = "answer"
    choices_field: str = "choices"

    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "DatasetConfig":
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        source = data.get("source", {})
        eval_config = data.get("eval", {})
        judge_config = data.get("judge", {})

        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            source_type=source.get("type", "huggingface"),
            source_path=source.get("path", ""),
            subset=source.get("subset"),
            split=source.get("split", "test"),
            eval_method=eval_config.get("method", "multiple_choice"),
            sample_size=eval_config.get("sample_size"),
            random_seed=eval_config.get("random_seed", 42),
            n_shot=eval_config.get("n_shot", 5),
            prompt_template=data.get("prompt_template", ""),
            judge_method=judge_config.get("method", "exact_match"),
            answer_field=judge_config.get("answer_field", "answer"),
            choices_field=judge_config.get("choices_field", "choices"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class DataSample:
    """单个数据样本."""
    id: str
    question: str
    choices: Optional[List[str]] = None
    answer: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class DatasetLoader:
    """数据集加载器."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(self) -> List[DataSample]:
        """加载数据集."""
        if self.config.source_type == "huggingface":
            return self._load_huggingface()
        elif self.config.source_type == "local":
            return self._load_local()
        elif self.config.source_type == "custom":
            return self._load_custom()
        else:
            raise ValueError(f"不支持的数据源类型: {self.config.source_type}")

    def _load_huggingface(self) -> List[DataSample]:
        from datasets import load_dataset

        dataset = load_dataset(
            self.config.source_path,
            self.config.subset,
            split=self.config.split,
        )

        samples: List[DataSample] = []
        for idx, item in enumerate(dataset):
            if self.config.sample_size and idx >= self.config.sample_size:
                break

            sample = DataSample(
                id=str(idx),
                question=item.get("question", ""),
                choices=item.get(self.config.choices_field),
                answer=str(item.get(self.config.answer_field)),
            )
            samples.append(sample)

        return samples

    def _load_local(self) -> List[DataSample]:
        path = self.config.source_path
        if path.endswith(".json"):
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif path.endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(path)
            data = df.to_dict("records")
        else:
            raise ValueError(f"不支持的文件格式: {path}")

        samples: List[DataSample] = []
        for idx, item in enumerate(data):
            if self.config.sample_size and idx >= self.config.sample_size:
                break

            sample = DataSample(
                id=str(idx),
                question=item.get("question", ""),
                choices=item.get(self.config.choices_field),
                answer=str(item.get(self.config.answer_field)),
            )
            samples.append(sample)

        return samples

    def _load_custom(self) -> List[DataSample]:
        raise NotImplementedError("自定义数据源加载器尚未实现")


def load_dataset_config(name: str) -> DatasetConfig:
    """加载数据集配置.

    Args:
        name: 数据集名称（对应 configs/datasets/<name>.yaml）

    Returns:
        DatasetConfig: 数据集配置
    """
    from llm_guard_bench.constants import CONFIGS_DIR

    config_path = CONFIGS_DIR / "datasets" / f"{name}.yaml"
    if not config_path.exists():
        raise ValueError(f"数据集配置不存在: {config_path}")
    return DatasetConfig.from_yaml(str(config_path))
