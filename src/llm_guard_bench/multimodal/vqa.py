"""图文理解评测.

评测内容：
    - TextVQA: 图片中的文字识别和理解
    - MMBench: 多模态综合理解
    - 自建中文场景 VQA
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .adapter import MultimodalAdapter


@dataclass
class VQASample:
    """视觉问答评测样本."""

    sample_id: str
    image_path: str
    question: str
    answer: str  # 标准答案
    category: str = "general"  # 类别：ocr / scene / knowledge
    difficulty: str = "medium"  # easy / medium / hard


@dataclass
class VQAResult:
    """VQA 评测结果."""

    sample_id: str
    question: str
    expected_answer: str
    model_answer: str
    is_correct: bool = False
    similarity: float = 0.0  # 与标准答案的相似度
    category: str = ""
    difficulty: str = ""


@dataclass
class VQAReport:
    """VQA 评测报告."""

    model_name: str
    total_samples: int = 0
    correct_count: int = 0
    accuracy: float = 0.0
    results: List[VQAResult] = field(default_factory=list)
    category_stats: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "correct_count": self.correct_count,
            "accuracy": f"{self.accuracy:.1%}",
            "category_stats": self.category_stats,
            "results": [
                {
                    "sample_id": r.sample_id,
                    "question": r.question[:80],
                    "expected": r.expected_answer[:50],
                    "model_answer": r.model_answer[:50],
                    "is_correct": r.is_correct,
                    "similarity": round(r.similarity, 3),
                    "category": r.category,
                }
                for r in self.results
            ],
        }


# ========== 内置中文场景 VQA 数据集 ==========

BUILTIN_VQA_SAMPLES = [
    VQASample(
        sample_id="vqa_ocr_001",
        image_path="",  # 运行时需要提供图片
        question="图片中的文字内容是什么？",
        answer="菜单",
        category="ocr",
        difficulty="easy",
    ),
    VQASample(
        sample_id="vqa_scene_001",
        image_path="",
        question="这张图片是在什么场景下拍摄的？",
        answer="户外",
        category="scene",
        difficulty="easy",
    ),
    VQASample(
        sample_id="vqa_knowledge_001",
        image_path="",
        question="图片中展示的是什么类型的图表？",
        answer="柱状图",
        category="knowledge",
        difficulty="medium",
    ),
]


class VQAEvaluator:
    """VQA 评测器."""

    def __init__(self, adapter: MultimodalAdapter) -> None:
        self._adapter = adapter

    def evaluate_single(self, sample: VQASample) -> VQAResult:
        """评测单个 VQA 样本."""
        result = self._adapter.vqa(
            image_path=sample.image_path,
            question=sample.question,
            max_tokens=256,
        )

        model_answer = result.text if result.success else f"[ERROR: {result.error}]"

        # 简单匹配：检查标准答案是否在模型回复中
        is_correct = sample.answer.lower() in model_answer.lower()

        # 计算相似度（简单的词重叠率）
        expected_words = set(sample.answer.lower().split())
        model_words = set(model_answer.lower().split())
        if expected_words:
            overlap = len(expected_words & model_words)
            similarity = overlap / len(expected_words)
        else:
            similarity = 0.0

        return VQAResult(
            sample_id=sample.sample_id,
            question=sample.question,
            expected_answer=sample.answer,
            model_answer=model_answer,
            is_correct=is_correct,
            similarity=similarity,
            category=sample.category,
            difficulty=sample.difficulty,
        )

    def evaluate_batch(self, samples: List[VQASample]) -> VQAReport:
        """批量评测."""
        report = VQAReport(model_name=self._adapter.model_name)

        for sample in samples:
            vqa_result = self.evaluate_single(sample)
            report.results.append(vqa_result)

        report.total_samples = len(report.results)
        report.correct_count = sum(1 for r in report.results if r.is_correct)
        report.accuracy = report.correct_count / report.total_samples if report.total_samples > 0 else 0.0

        # 按类别统计
        cat_stats: Dict[str, Dict[str, Any]] = {}
        for r in report.results:
            if r.category not in cat_stats:
                cat_stats[r.category] = {"total": 0, "correct": 0}
            cat_stats[r.category]["total"] += 1
            if r.is_correct:
                cat_stats[r.category]["correct"] += 1

        for cat in cat_stats:
            total = cat_stats[cat]["total"]
            correct = cat_stats[cat]["correct"]
            cat_stats[cat]["accuracy"] = correct / total if total > 0 else 0.0

        report.category_stats = cat_stats
        return report

    def load_samples_from_json(self, json_path: str) -> List[VQASample]:
        """从 JSON 文件加载评测样本."""
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        samples = []
        for item in data:
            samples.append(VQASample(
                sample_id=item.get("id", ""),
                image_path=item.get("image_path", ""),
                question=item.get("question", ""),
                answer=item.get("answer", ""),
                category=item.get("category", "general"),
                difficulty=item.get("difficulty", "medium"),
            ))
        return samples

    def save_report(self, report: VQAReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"vqa_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: VQAReport) -> None:
        """打印报告."""
        print(f"\nVQA 评测报告 - {report.model_name}")
        print(f"  总样本数: {report.total_samples}")
        print(f"  正确数: {report.correct_count}")
        print(f"  准确率: {report.accuracy:.1%}")

        if report.category_stats:
            print("\n  按类别统计:")
            for cat, stats in report.category_stats.items():
                print(f"    {cat}: {stats['correct']}/{stats['total']} ({stats['accuracy']:.1%})")
