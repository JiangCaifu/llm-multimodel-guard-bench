"""深度分析模块 - 分领域得分、校准度分析、模型对比.

对应 Phase 1 Day 11-12：
    - 分领域得分分布（按子领域/难度/数据集）
    - 校准度分析（模型置信度与实际准确率的偏差）
    - 多模型对比（横向对比不同模型表现）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DomainBreakdown:
    """分领域得分."""
    domain: str
    total: int
    correct: int
    accuracy: float
    samples: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class DifficultyBreakdown:
    """难度分布得分."""
    difficulty: str
    total: int
    correct: int
    accuracy: float


@dataclass
class CalibrationPoint:
    """校准度数据点."""
    bin_range: str  # e.g., "80-90%"
    predicted_accuracy: float
    actual_accuracy: float
    sample_count: int
    calibration_error: float  # |predicted - actual|


@dataclass
class ModelComparison:
    """模型对比结果."""
    dataset_name: str
    models: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    winner: str = ""


@dataclass
class AnalysisReport:
    """完整分析报告."""
    task_name: str
    model_name: str
    overall_accuracy: float
    domain_breakdowns: List[DomainBreakdown] = field(default_factory=list)
    difficulty_breakdowns: List[DifficultyBreakdown] = field(default_factory=list)
    calibration_points: List[CalibrationPoint] = field(default_factory=list)
    avg_latency_ms: float = 0.0
    total_tokens: int = 0
    error_rate: float = 0.0


class DeepAnalyzer:
    """深度分析器."""

    def __init__(self) -> None:
        self._reports: List[AnalysisReport] = []

    def analyze_result_file(self, result_path: str) -> AnalysisReport:
        """从评测结果 JSON 文件生成分析报告."""
        with open(result_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return self.analyze_result_data(data)

    def analyze_result_data(self, data: Dict[str, Any]) -> AnalysisReport:
        """从评测结果字典生成分析报告."""
        results = data.get("results", [])
        model_name = data.get("model_name", "unknown")
        dataset_name = data.get("dataset_name", "unknown")

        # 总体指标
        total = len(results)
        correct = sum(1 for r in results if r.get("is_correct"))
        overall_accuracy = correct / total if total > 0 else 0.0

        # 平均延迟
        latencies = [r.get("latency_ms", 0) for r in results if r.get("latency_ms")]
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0

        # Token 统计
        total_tokens = sum(
            r.get("prompt_tokens", 0) + r.get("completion_tokens", 0) for r in results
        )

        # 错误率
        error_count = sum(1 for r in results if r.get("error"))
        error_rate = error_count / total if total > 0 else 0.0

        # 分领域得分
        domain_breakdowns = self._compute_domain_breakdowns(results)

        # 难度分布得分
        difficulty_breakdowns = self._compute_difficulty_breakdowns(results)

        # 校准度分析
        calibration_points = self._compute_calibration(results)

        report = AnalysisReport(
            task_name=dataset_name,
            model_name=model_name,
            overall_accuracy=overall_accuracy,
            domain_breakdowns=domain_breakdowns,
            difficulty_breakdowns=difficulty_breakdowns,
            calibration_points=calibration_points,
            avg_latency_ms=avg_latency,
            total_tokens=total_tokens,
            error_rate=error_rate,
        )

        self._reports.append(report)
        return report

    def _compute_domain_breakdowns(self, results: List[Dict]) -> List[DomainBreakdown]:
        """按子领域(来自样本 metadata)分组统计."""
        domain_map: Dict[str, List[Dict]] = {}

        for r in results:
            # 从 sample_id 推断领域: fin_001 → 金融, med_001 → 医疗
            sample_id = r.get("sample_id", "")
            if sample_id.startswith("fin_"):
                domain = "金融"
            elif sample_id.startswith("med_"):
                domain = "医疗"
            else:
                domain = "通用"

            if domain not in domain_map:
                domain_map[domain] = []
            domain_map[domain].append(r)

        breakdowns = []
        for domain, samples in sorted(domain_map.items()):
            total = len(samples)
            correct = sum(1 for s in samples if s.get("is_correct"))
            breakdowns.append(
                DomainBreakdown(
                    domain=domain,
                    total=total,
                    correct=correct,
                    accuracy=correct / total if total > 0 else 0.0,
                    samples=samples,
                )
            )

        return breakdowns

    def _compute_difficulty_breakdowns(self, results: List[Dict]) -> List[DifficultyBreakdown]:
        """按难度分组统计."""
        # 从结果 JSON 的原始数据中提取难度信息
        difficulty_map: Dict[str, List[Dict]] = {"easy": [], "medium": [], "hard": []}

        for r in results:
            # 尝试从 metadata 获取难度，否则按 sample_id 推断
            difficulty = r.get("difficulty", "medium")
            if difficulty not in difficulty_map:
                difficulty_map[difficulty] = []
            difficulty_map[difficulty].append(r)

        breakdowns = []
        for diff in ["easy", "medium", "hard"]:
            samples = difficulty_map.get(diff, [])
            if not samples:
                continue
            total = len(samples)
            correct = sum(1 for s in samples if s.get("is_correct"))
            breakdowns.append(
                DifficultyBreakdown(
                    difficulty=diff,
                    total=total,
                    correct=correct,
                    accuracy=correct / total if total > 0 else 0.0,
                )
            )

        return breakdowns

    def _compute_calibration(self, results: List[Dict]) -> List[CalibrationPoint]:
        """校准度分析 - 基于模型回答中正确选项出现的相对位置推断置信度.

        简化方法：根据模型回答中是否包含正确答案关键词来分 bin。
        """
        if not results:
            return []

        bins = [
            ("0-20%", 0.0, 0.2),
            ("20-40%", 0.2, 0.4),
            ("40-60%", 0.4, 0.6),
            ("60-80%", 0.6, 0.8),
            ("80-100%", 0.8, 1.0),
        ]

        calibration_points = []
        for bin_name, low, high in bins:
            # 简化: 用回答长度/模式作为"置信度"代理
            # 这里用等分方式分配样本
            mid = (low + high) / 2
            bin_samples = []
            for r in results:
                answer = r.get("model_answer", "")
                correct = r.get("correct_answer", "")
                if not answer or not correct:
                    continue
                # 简单置信度: 回答越简洁置信度越高
                confidence = min(1.0, max(0.0, 1.0 - len(answer) / 200.0))
                if low <= confidence < high:
                    bin_samples.append(r)

            if not bin_samples:
                continue

            actual_acc = sum(1 for s in bin_samples if s.get("is_correct")) / len(bin_samples)
            calibration_points.append(
                CalibrationPoint(
                    bin_range=bin_name,
                    predicted_accuracy=mid,
                    actual_accuracy=actual_acc,
                    sample_count=len(bin_samples),
                    calibration_error=abs(mid - actual_acc),
                )
            )

        return calibration_points

    def compare_models(
        self,
        result_paths: Dict[str, str],
    ) -> List[ModelComparison]:
        """多模型对比.

        Args:
            result_paths: {model_name: result_json_path}

        Returns:
            每个数据集的对比结果
        """
        model_results: Dict[str, Dict[str, Any]] = {}

        for model_name, path in result_paths.items():
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            dataset_name = data.get("dataset_name", "unknown")
            if model_name not in model_results:
                model_results[model_name] = {}
            model_results[model_name][dataset_name] = {
                "accuracy": data.get("accuracy", 0.0),
                "total_samples": data.get("total_samples", 0),
                "correct_samples": data.get("correct_samples", 0),
            }

        # 按数据集分组
        dataset_names = set()
        for model_data in model_results.values():
            dataset_names.update(model_data.keys())

        comparisons = []
        for ds_name in sorted(dataset_names):
            comparison = ModelComparison(dataset_name=ds_name)
            best_model = ""
            best_acc = -1.0

            for model_name, model_data in model_results.items():
                if ds_name in model_data:
                    ds_result = model_data[ds_name]
                    comparison.models[model_name] = ds_result
                    if ds_result["accuracy"] > best_acc:
                        best_acc = ds_result["accuracy"]
                        best_model = model_name

            comparison.winner = best_model
            comparisons.append(comparison)

        return comparisons

    def export_report(self, report: AnalysisReport, output_dir: str) -> str:
        """导出分析报告为 JSON."""
        os.makedirs(output_dir, exist_ok=True)

        report_data = {
            "task_name": report.task_name,
            "model_name": report.model_name,
            "overall_accuracy": report.overall_accuracy,
            "avg_latency_ms": round(report.avg_latency_ms, 1),
            "total_tokens": report.total_tokens,
            "error_rate": report.error_rate,
            "domain_breakdowns": [
                {
                    "domain": b.domain,
                    "total": b.total,
                    "correct": b.correct,
                    "accuracy": round(b.accuracy, 4),
                }
                for b in report.domain_breakdowns
            ],
            "difficulty_breakdowns": [
                {
                    "difficulty": b.difficulty,
                    "total": b.total,
                    "correct": b.correct,
                    "accuracy": round(b.accuracy, 4),
                }
                for b in report.difficulty_breakdowns
            ],
            "calibration": [
                {
                    "bin_range": p.bin_range,
                    "predicted": round(p.predicted_accuracy, 2),
                    "actual": round(p.actual_accuracy, 4),
                    "count": p.sample_count,
                    "error": round(p.calibration_error, 4),
                }
                for p in report.calibration_points
            ],
        }

        output_path = os.path.join(
            output_dir, f"analysis_{report.model_name}_{report.task_name}.json"
        )
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

        return output_path

    def print_report(self, report: AnalysisReport) -> None:
        """打印分析报告到终端."""
        from rich.console import Console
        from rich.table import Table

        console = Console()

        console.print(f"\n[bold cyan]{'='*60}[/bold cyan]")
        console.print(f"[bold]分析报告: {report.task_name} | 模型: {report.model_name}[/bold]")
        console.print(f"[bold cyan]{'='*60}[/bold cyan]")

        # 总体指标
        console.print(f"\n[bold]总体指标[/bold]")
        console.print(f"  准确率: [green]{report.overall_accuracy:.1%}[/green]")
        console.print(f"  平均延迟: {report.avg_latency_ms:.0f}ms")
        console.print(f"  总Token: {report.total_tokens}")
        console.print(f"  错误率: {report.error_rate:.1%}")

        # 分领域
        if report.domain_breakdowns:
            table = Table(title="分领域得分")
            table.add_column("领域", style="cyan")
            table.add_column("总数", justify="right")
            table.add_column("正确", justify="right")
            table.add_column("准确率", justify="right", style="green")

            for b in report.domain_breakdowns:
                table.add_row(b.domain, str(b.total), str(b.correct), f"{b.accuracy:.1%}")

            console.print(table)

        # 难度分布
        if report.difficulty_breakdowns:
            table = Table(title="难度分布")
            table.add_column("难度", style="cyan")
            table.add_column("总数", justify="right")
            table.add_column("正确", justify="right")
            table.add_column("准确率", justify="right", style="green")

            for b in report.difficulty_breakdowns:
                table.add_row(b.difficulty, str(b.total), str(b.correct), f"{b.accuracy:.1%}")

            console.print(table)

        # 校准度
        if report.calibration_points:
            table = Table(title="校准度分析")
            table.add_column("置信区间", style="cyan")
            table.add_column("预测准确率", justify="right")
            table.add_column("实际准确率", justify="right")
            table.add_column("样本数", justify="right")
            table.add_column("校准误差", justify="right", style="yellow")

            for p in report.calibration_points:
                table.add_row(
                    p.bin_range,
                    f"{p.predicted_accuracy:.0%}",
                    f"{p.actual_accuracy:.1%}",
                    str(p.sample_count),
                    f"{p.calibration_error:.2f}",
                )

            console.print(table)
