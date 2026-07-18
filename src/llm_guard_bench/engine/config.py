"""评测任务配置 - YAML 驱动.

支持从 YAML 文件加载完整的评测任务配置，包括：
    - 待评测模型列表
    - 待评测数据集列表
    - 执行参数（并行/串行、样本数限制等）
    - 报告生成配置

使用方式：
    config = EvaluationConfig.from_yaml("configs/evaluation/capability.yaml")
    runner = EvaluationRunner(config)
    runner.run()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_guard_bench.constants import CONFIGS_DIR, RESULTS_DIR


@dataclass
class ExecutionConfig:
    """执行配置."""
    parallel: bool = False
    max_samples_per_dataset: Optional[int] = None
    save_intermediate: bool = True
    output_dir: str = "./data/results"


@dataclass
class ReportConfig:
    """报告配置."""
    generate_html: bool = True
    generate_radar_chart: bool = True
    output_format: List[str] = field(default_factory=lambda: ["json", "csv"])


@dataclass
class EvaluationConfig:
    """评测任务配置."""
    name: str
    description: str = ""
    version: str = "0.1.0"

    models: List[str] = field(default_factory=list)
    datasets: List[str] = field(default_factory=list)

    execution: ExecutionConfig = field(default_factory=ExecutionConfig)
    report: ReportConfig = field(default_factory=ReportConfig)

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "EvaluationConfig":
        """从 YAML 文件加载配置."""
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)

        execution_data = data.get("execution", {})
        report_data = data.get("report", {})

        return cls(
            name=data["name"],
            description=data.get("description", ""),
            version=data.get("version", "0.1.0"),
            models=data.get("models", []),
            datasets=data.get("datasets", []),
            execution=ExecutionConfig(
                parallel=execution_data.get("parallel", False),
                max_samples_per_dataset=execution_data.get("max_samples_per_dataset"),
                save_intermediate=execution_data.get("save_intermediate", True),
                output_dir=execution_data.get("output_dir", "./data/results"),
            ),
            report=ReportConfig(
                generate_html=report_data.get("generate_html", True),
                generate_radar_chart=report_data.get("generate_radar_chart", True),
                output_format=report_data.get("output_format", ["json", "csv"]),
            ),
        )

    def get_output_dir(self) -> str:
        """获取输出目录（支持相对路径）."""
        if self.execution.output_dir.startswith("."):
            import os
            return os.path.abspath(self.execution.output_dir)
        return self.execution.output_dir
