"""应用层体验测试 - 统一入口.

编排 G1/G2/G3 的评测流程。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter
from ..adapters.factory import build_adapter, load_model_config
from .ab_testing import ABTestConfig, ABTestRunner, ABTestReport
from .optimization_loop import OptimizationLoop, OptimizationReport
from .ux_eval import UXEvaluator, UXReport


@dataclass
class ExperienceReport:
    """应用层体验测试总报告."""

    model_name: str = ""
    ux_report: Optional[UXReport] = None
    ab_report: Optional[ABTestReport] = None
    opt_report: Optional[OptimizationReport] = None

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {"model_name": self.model_name}
        if self.ux_report:
            d["ux_report"] = self.ux_report.to_dict()
        if self.ab_report:
            d["ab_report"] = self.ab_report.to_dict()
        if self.opt_report:
            d["opt_report"] = self.opt_report.to_dict()
        return d


class ExperienceRunner:
    """应用层体验测试运行器."""

    def __init__(
        self,
        model_adapter: BaseModelAdapter,
        judge_adapter: Optional[BaseModelAdapter] = None,
    ) -> None:
        self._model = model_adapter
        self._judge = judge_adapter or model_adapter

    def run_ux(
        self,
        inputs: Optional[List[str]] = None,
        samples: Optional[List[Dict[str, str]]] = None,
    ) -> UXReport:
        """运行UX评测."""
        evaluator = UXEvaluator(self._model, self._judge)
        report = evaluator.evaluate(inputs=inputs, samples=samples)
        UXEvaluator.print_report(report)
        return report

    def run_ab(
        self,
        model_b: BaseModelAdapter,
        prompts: List[str],
        config: Optional[ABTestConfig] = None,
    ) -> ABTestReport:
        """运行A/B测试."""
        config = config or ABTestConfig(
            model_a_name=self._model.get_model_info().name,
            model_b_name=model_b.get_model_info().name,
        )
        runner = ABTestRunner(self._model, model_b, self._judge, config)
        report = runner.run(prompts)
        ABTestRunner.print_report(report)
        return report

    def run_optimization(
        self,
        scores: Dict[str, float],
        issues: List[str],
        after_scores: Optional[Dict[str, float]] = None,
    ) -> OptimizationReport:
        """运行优化闭环."""
        loop = OptimizationLoop(self._judge)
        report = loop.run_loop(
            model_name=self._model.get_model_info().name,
            before_scores=scores,
            issues=issues,
            after_scores=after_scores,
        )
        OptimizationLoop.print_report(report)
        return report

    def run_full(
        self,
        inputs: Optional[List[str]] = None,
    ) -> ExperienceReport:
        """运行全部体验测试."""
        report = ExperienceReport(model_name=self._model.get_model_info().name)

        # G2: UX评测
        print("=" * 60)
        print("G2: 用户体验评测")
        print("=" * 60)
        report.ux_report = self.run_ux(inputs=inputs)

        # G3: 优化闭环（基于UX结果）
        print("\n" + "=" * 60)
        print("G3: 优化闭环")
        print("=" * 60)
        report.opt_report = self.run_optimization(
            scores=report.ux_report.dimension_scores,
            issues=[issue.split(" ")[0] for issue in report.ux_report.top_issues],
        )

        return report

    def save_report(self, report: ExperienceReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"experience_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path
