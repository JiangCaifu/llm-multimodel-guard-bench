"""评测引擎 - YAML 配置驱动的评测集加载与执行调度.

核心组件：
    - EvaluationConfig: 评测任务配置 (YAML 加载)
    - EvaluationRunner: 评测执行器
    - DatasetLoader: 数据集加载器
    - EvalResult: 评测结果数据结构
"""

from llm_guard_bench.engine.config import EvaluationConfig, ExecutionConfig, ReportConfig
from llm_guard_bench.engine.dataloader import DataSample, DatasetConfig, DatasetLoader, load_dataset_config
from llm_guard_bench.engine.runner import DatasetEvalResult, EvaluationRunResult, EvaluationRunner, EvalResult

__all__ = [
    "EvaluationConfig",
    "ExecutionConfig",
    "ReportConfig",
    "DataSample",
    "DatasetConfig",
    "DatasetLoader",
    "load_dataset_config",
    "EvaluationRunner",
    "EvalResult",
    "DatasetEvalResult",
    "EvaluationRunResult",
]
