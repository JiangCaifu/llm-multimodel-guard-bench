"""对齐分析运行器 - 统一编排标注+对齐分析流程.

支持三种模式：
    1. demo: 用演示数据验证对齐指标计算是否正确
    2. annotate: 交互式标注模式
    3. analyze: 加载已标注数据，计算对齐指标
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter
from .annotator import Annotator, AnnotationDataset, AnnotationSample
from .metrics import AlignmentReport, compute_alignment, print_report


class AlignmentRunner:
    """对齐分析运行器."""

    def __init__(self, judge_adapter: Optional[BaseModelAdapter] = None) -> None:
        self._judge = judge_adapter
        self._annotator = Annotator()

    def run_demo(self, n: int = 10) -> AlignmentReport:
        """演示模式：用模拟数据验证对齐指标计算."""
        print("=" * 60)
        print("Phase 6 - Judge对齐分析 (演示模式)")
        print("=" * 60)

        dataset = Annotator.demo_dataset(n)
        print(f"\n生成演示数据: {len(dataset.samples)}条")
        Annotator.print_stats(dataset)

        # 计算对齐
        judge_labels = [s.judge_label for s in dataset.samples]
        human_labels = [s.human_label for s in dataset.samples]

        report = compute_alignment(
            judge_labels=judge_labels,
            human_labels=human_labels,
            samples=dataset.samples,
        )
        print_report(report)
        return report

    def run_annotate(
        self,
        inputs: List[str],
        outputs: List[str],
        judge_labels: Optional[List[str]] = None,
        output_path: str = "./data/annotations/annotation.json",
    ) -> AnnotationDataset:
        """交互式标注模式."""
        print("=" * 60)
        print("Phase 6 - 人工标注模式")
        print("=" * 60)

        dataset = self._annotator.build_samples(inputs, outputs, judge_labels)
        dataset = self._annotator.annotate_interactive(dataset)

        # 保存
        self._annotator.save(dataset, output_path)
        print(f"\n标注数据已保存: {output_path}")
        Annotator.print_stats(dataset)

        return dataset

    def run_analyze(
        self,
        annotation_path: str,
        output_dir: str = "./data/results/alignment",
    ) -> AlignmentReport:
        """分析模式：加载已标注数据，计算对齐指标."""
        print("=" * 60)
        print("Phase 6 - Judge对齐分析")
        print("=" * 60)

        # 加载标注数据
        dataset = Annotator.load(annotation_path)
        print(f"\n加载标注数据: {len(dataset.samples)}条")
        Annotator.print_stats(dataset)

        # 计算对齐
        judge_labels = [s.judge_label for s in dataset.samples]
        human_labels = [s.human_label for s in dataset.samples]

        report = compute_alignment(
            judge_labels=judge_labels,
            human_labels=human_labels,
            samples=dataset.samples,
            labels=dataset.labels,
        )
        print_report(report)

        # 保存报告
        os.makedirs(output_dir, exist_ok=True)
        report_path = os.path.join(output_dir, "alignment_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n对齐报告已保存: {report_path}")

        return report

    def run_from_eval_results(
        self,
        eval_results_path: str,
        output_dir: str = "./data/results/alignment",
    ) -> AlignmentReport:
        """从已有评测结果生成待标注数据.

        读取评测结果（如UX/安全评测输出），提取样本构建标注任务。
        需要用户手动标注后再分析。
        """
        print("=" * 60)
        print("Phase 6 - 从评测结果生成标注任务")
        print("=" * 60)

        with open(eval_results_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 兼容多种评测结果格式
        samples_data = data.get("samples", [])
        if not samples_data:
            print("未找到samples字段，无法生成标注任务")
            return AlignmentReport()

        inputs = [s.get("user_input", "") for s in samples_data]
        outputs = [s.get("model_output", "") for s in samples_data]

        # 尝试提取judge标签
        judge_labels = None
        if samples_data and "judge_label" in samples_data[0]:
            judge_labels = [s.get("judge_label", "") for s in samples_data]
        elif samples_data and "risk_level" in samples_data[0]:
            # 安全评测格式
            judge_labels = [str(s.get("risk_level", "")) for s in samples_data]
        elif samples_data and "overall_score" in samples_data[0]:
            # UX评测格式：分数>=8为correct，<6为incorrect，其余为partial
            judge_labels = []
            for s in samples_data:
                score = s.get("overall_score", 0)
                if score >= 8:
                    judge_labels.append("correct")
                elif score < 6:
                    judge_labels.append("incorrect")
                else:
                    judge_labels.append("partial")

        dataset = self._annotator.build_samples(inputs, outputs, judge_labels)

        # 保存待标注数据
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "pending_annotation.json")
        self._annotator.save(dataset, path)
        print(f"\n待标注数据已生成: {path}")
        print(f"样本数: {len(dataset.samples)}")
        print(f"\n下一步: 编辑该文件填写human_label字段，或使用CLI交互式标注")
        print(f"  python -m llm_guard_bench.cli alignment --task annotate --input {path}")

        return dataset
