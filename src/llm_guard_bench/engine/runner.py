"""评测执行器 - 核心评测引擎.

负责：
    - 加载模型适配器
    - 加载数据集
    - 执行评测任务
    - 聚合结果
    - 生成报告

使用方式：
    config = EvaluationConfig.from_yaml("configs/evaluation/capability.yaml")
    runner = EvaluationRunner(config)
    result = runner.run()
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from llm_guard_bench.adapters import GenerationResult, build_adapter, load_model_config
from llm_guard_bench.engine.config import EvaluationConfig
from llm_guard_bench.engine.dataloader import DataSample, DatasetConfig, DatasetLoader, load_dataset_config


@dataclass
class EvalResult:
    """单条评测结果."""
    sample_id: str
    question: str
    model_answer: str
    correct_answer: str
    is_correct: bool
    prompt_tokens: int = 0
    completion_tokens: int = 0
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class DatasetEvalResult:
    """数据集评测结果."""
    dataset_name: str
    model_name: str
    total_samples: int
    correct_samples: int
    accuracy: float
    results: List[EvalResult] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRunResult:
    """评测运行结果."""
    task_name: str
    task_version: str
    model_results: Dict[str, DatasetEvalResult] = field(default_factory=dict)
    output_dir: str = ""
    start_time: str = ""
    end_time: str = ""


class EvaluationRunner:
    """评测执行器."""

    def __init__(self, config: EvaluationConfig) -> None:
        self.config = config
        self._results: List[DatasetEvalResult] = []

    def run(self) -> EvaluationRunResult:
        """执行评测任务."""
        from datetime import datetime

        start_time = datetime.now().isoformat()
        output_dir = self.config.get_output_dir()
        os.makedirs(output_dir, exist_ok=True)

        for model_name in self.config.models:
            self._evaluate_model(model_name, output_dir)

        end_time = datetime.now().isoformat()

        result = EvaluationRunResult(
            task_name=self.config.name,
            task_version=self.config.version,
            output_dir=output_dir,
            start_time=start_time,
            end_time=end_time,
        )

        self._save_results(result)
        return result

    def _evaluate_model(self, model_name: str, output_dir: str) -> None:
        from loguru import logger

        logger.info(f"开始评测模型: {model_name}")

        config = load_model_config(model_name)
        adapter = build_adapter(config)

        try:
            for dataset_name in self.config.datasets:
                dataset_result = self._evaluate_dataset(adapter, dataset_name)
                self._results.append(dataset_result)

                dataset_dir = os.path.join(output_dir, model_name)
                os.makedirs(dataset_dir, exist_ok=True)
                self._save_dataset_result(dataset_result, dataset_dir)

        finally:
            adapter.close()

    def _evaluate_dataset(
        self,
        adapter: Any,
        dataset_name: str,
    ) -> DatasetEvalResult:
        from loguru import logger

        logger.info(f"  加载数据集: {dataset_name}")

        dataset_config = load_dataset_config(dataset_name)
        loader = DatasetLoader(dataset_config)
        samples = loader.load()

        if self.config.execution.max_samples_per_dataset:
            samples = samples[: self.config.execution.max_samples_per_dataset]

        logger.info(f"  评测样本数: {len(samples)}")

        results: List[EvalResult] = []
        correct_count = 0

        for sample in samples:
            eval_result = self._evaluate_sample(adapter, dataset_config, sample)
            results.append(eval_result)
            if eval_result.is_correct:
                correct_count += 1

        accuracy = correct_count / len(results) if results else 0.0

        logger.info(f"  准确率: {accuracy:.2%} ({correct_count}/{len(results)})")

        return DatasetEvalResult(
            dataset_name=dataset_name,
            model_name=adapter.get_model_info().name,
            total_samples=len(results),
            correct_samples=correct_count,
            accuracy=accuracy,
            results=results,
        )

    def _evaluate_sample(
        self,
        adapter: Any,
        dataset_config: DatasetConfig,
        sample: DataSample,
    ) -> EvalResult:
        prompt = self._build_prompt(dataset_config, sample)

        result = adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=256,
            temperature=0.0,
        )

        is_correct = self._judge_answer(dataset_config, result.text, sample.answer)

        return EvalResult(
            sample_id=sample.id,
            question=sample.question,
            model_answer=result.text,
            correct_answer=sample.answer or "",
            is_correct=is_correct,
            prompt_tokens=result.prompt_tokens,
            completion_tokens=result.completion_tokens,
            latency_ms=result.latency_ms,
            error=result.error,
        )

    def _build_prompt(
        self,
        dataset_config: DatasetConfig,
        sample: DataSample,
    ) -> str:
        if not dataset_config.prompt_template:
            return sample.question

        template_vars = {
            "question": sample.question,
            "choices": sample.choices or [],
            "answer": sample.answer or "",
            "few_shot_examples": "",
            "subject": dataset_config.metadata.get("domain", ""),
        }

        from jinja2 import Template

        template = Template(dataset_config.prompt_template)
        return template.render(template_vars)

    def _judge_answer(
        self,
        dataset_config: DatasetConfig,
        model_answer: str,
        correct_answer: str,
    ) -> bool:
        method = dataset_config.judge_method

        if method == "exact_match":
            return model_answer.strip().lower() == correct_answer.strip().lower()

        elif method == "contains":
            return correct_answer.strip().lower() in model_answer.strip().lower()

        elif method == "multiple_choice":
            model_ans = model_answer.strip().upper()
            correct_ans = correct_answer.strip().upper()
            return model_ans == correct_ans or correct_ans in model_ans

        else:
            raise ValueError(f"不支持的评判方法: {method}")

    def _save_results(self, result: EvaluationRunResult) -> None:
        summary = {
            "task_name": result.task_name,
            "task_version": result.task_version,
            "start_time": result.start_time,
            "end_time": result.end_time,
            "models": {},
        }

        for dataset_result in self._results:
            if dataset_result.model_name not in summary["models"]:
                summary["models"][dataset_result.model_name] = {}
            summary["models"][dataset_result.model_name][dataset_result.dataset_name] = {
                "total_samples": dataset_result.total_samples,
                "correct_samples": dataset_result.correct_samples,
                "accuracy": dataset_result.accuracy,
            }

        summary_path = os.path.join(result.output_dir, "summary.json")
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    def _save_dataset_result(self, result: DatasetEvalResult, output_dir: str) -> None:
        result_data = {
            "dataset_name": result.dataset_name,
            "model_name": result.model_name,
            "total_samples": result.total_samples,
            "correct_samples": result.correct_samples,
            "accuracy": result.accuracy,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question,
                    "model_answer": r.model_answer,
                    "correct_answer": r.correct_answer,
                    "is_correct": r.is_correct,
                    "latency_ms": r.latency_ms,
                    "error": r.error,
                }
                for r in result.results
            ],
        }

        result_path = os.path.join(output_dir, f"{result.dataset_name}.json")
        with open(result_path, "w", encoding="utf-8") as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False)
