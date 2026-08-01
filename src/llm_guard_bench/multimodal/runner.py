"""多模态评测统一入口.

编排 VQA / 幻觉检测 / CLIP Score 评测流程。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .adapter import MultimodalAdapter
from .clip_score import CLIPScoreEvaluator
from .hallucination import HallucinationDetector, HallucinationSample
from .vqa import VQAEvaluator, VQASample


@dataclass
class MultimodalReport:
    """多模态评测综合报告."""

    model_name: str
    vqa_accuracy: float = 0.0
    hallucination_rate: float = 0.0
    avg_clip_score: float = 0.0
    vqa_report: Optional[Dict[str, Any]] = None
    hallucination_report: Optional[Dict[str, Any]] = None
    clip_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "vqa_accuracy": f"{self.vqa_accuracy:.1%}",
            "hallucination_rate": f"{self.hallucination_rate:.1%}",
            "avg_clip_score": f"{self.avg_clip_score:.3f}",
            "vqa_report": self.vqa_report,
            "hallucination_report": self.hallucination_report,
            "clip_report": self.clip_report,
        }


class MultimodalRunner:
    """多模态评测执行器."""

    def __init__(
        self,
        adapter: MultimodalAdapter,
        judge_adapter: Optional[MultimodalAdapter] = None,
    ) -> None:
        self._adapter = adapter
        self._judge = judge_adapter or adapter
        self._vqa_evaluator = VQAEvaluator(adapter)
        self._hallucination_detector = HallucinationDetector(adapter, self._judge)
        self._clip_evaluator = CLIPScoreEvaluator(adapter)

    def run_vqa(self, samples: List[VQASample]) -> Dict[str, Any]:
        """运行 VQA 评测."""
        report = self._vqa_evaluator.evaluate_batch(samples)
        return report.to_dict()

    def run_hallucination(
        self,
        samples: List[HallucinationSample],
    ) -> Dict[str, Any]:
        """运行幻觉检测."""
        report = self._hallucination_detector.detect_batch(samples)
        return report.to_dict()

    def run_clip_score(
        self,
        samples: List[tuple[str, str, str]],
    ) -> Dict[str, Any]:
        """运行 CLIP Score 评测."""
        report = self._clip_evaluator.evaluate_batch(samples)
        return report.to_dict()

    def run_full(
        self,
        vqa_samples: Optional[List[VQASample]] = None,
        hallucination_samples: Optional[List[HallucinationSample]] = None,
        clip_samples: Optional[List[tuple[str, str, str]]] = None,
    ) -> MultimodalReport:
        """运行完整多模态评测."""
        report = MultimodalReport(model_name=self._adapter.model_name)

        if vqa_samples:
            vqa_report = self.run_vqa(vqa_samples)
            report.vqa_report = vqa_report
            acc_str = vqa_report.get("accuracy", "0.0%")
            report.vqa_accuracy = float(acc_str.rstrip("%")) / 100 if "%" in acc_str else 0.0

        if hallucination_samples:
            h_report = self.run_hallucination(hallucination_samples)
            report.hallucination_report = h_report
            rate_str = h_report.get("overall_hallucination_rate", "0.0%")
            report.hallucination_rate = float(rate_str.rstrip("%")) / 100 if "%" in rate_str else 0.0

        if clip_samples:
            c_report = self.run_clip_score(clip_samples)
            report.clip_report = c_report
            report.avg_clip_score = c_report.get("avg_clip_score", 0.0)

        return report

    def save_report(self, report: MultimodalReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"multimodal_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: MultimodalReport) -> None:
        """打印报告."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(Panel(
                f"VQA 准确率: {report.vqa_accuracy:.1%}\n"
                f"幻觉率: [red]{report.hallucination_rate:.1%}[/red]\n"
                f"CLIP Score: {report.avg_clip_score:.3f}",
                title=f"多模态评测 - {report.model_name}",
                expand=False,
            ))
        except ImportError:
            print(f"\n多模态评测报告 - {report.model_name}")
            print(f"  VQA准确率: {report.vqa_accuracy:.1%}")
            print(f"  幻觉率: {report.hallucination_rate:.1%}")
            print(f"  CLIP Score: {report.avg_clip_score:.3f}")
