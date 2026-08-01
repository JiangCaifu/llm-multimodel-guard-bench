"""F2: AI缺陷识别与分类.

输入：模型回复 + 期望输出
输出：缺陷报告（是否Bug、分类、严重度、复现步骤）

流程：
    1. 输入测试输出（模型回复 vs 期望结果）
    2. LLM判断是否是缺陷
    3. 自动分类 + 生成缺陷报告
    4. 与人工判断对齐（Precision/Recall）
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


class BugCategory(str, Enum):
    """缺陷分类."""

    FUNCTIONAL = "functional"        # 功能缺陷
    PERFORMANCE = "performance"      # 性能缺陷
    SECURITY = "security"            # 安全缺陷
    UX = "ux"                        # 体验缺陷
    ACCURACY = "accuracy"            # 准确性缺陷（幻觉/错误信息）
    COMPLETENESS = "completeness"    # 完整性缺陷（遗漏）
    FORMATTING = "formatting"        # 格式缺陷
    OTHER = "other"                  # 其他


class BugSeverity(str, Enum):
    """严重程度."""

    P0 = "P0"  # 阻断：功能完全不可用/安全漏洞
    P1 = "P1"  # 严重：核心功能异常
    P2 = "P2"  # 一般：非核心功能异常
    P3 = "P3"  # 轻微：体验/格式问题


@dataclass
class BugReport:
    """单条缺陷报告."""

    bug_id: str
    is_bug: bool
    category: BugCategory = BugCategory.OTHER
    severity: BugSeverity = BugSeverity.P3
    description: str = ""
    reproduce_steps: List[str] = field(default_factory=list)
    actual_output: str = ""
    expected_output: str = ""
    confidence: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "bug_id": self.bug_id,
            "is_bug": self.is_bug,
            "category": self.category.value,
            "severity": self.severity.value,
            "description": self.description,
            "reproduce_steps": self.reproduce_steps,
            "actual_output": self.actual_output,
            "expected_output": self.expected_output,
            "confidence": self.confidence,
        }


@dataclass
class BugDetectionReport:
    """缺陷检测总报告."""

    total_samples: int = 0
    bug_count: int = 0
    no_bug_count: int = 0
    by_category: Dict[str, int] = field(default_factory=dict)
    by_severity: Dict[str, int] = field(default_factory=dict)
    results: List[BugReport] = field(default_factory=list)
    detection_time_ms: float = 0.0

    @property
    def bug_rate(self) -> float:
        return self.bug_count / self.total_samples if self.total_samples else 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "bug_count": self.bug_count,
            "no_bug_count": self.no_bug_count,
            "bug_rate": self.bug_rate,
            "by_category": self.by_category,
            "by_severity": self.by_severity,
            "results": [r.to_dict() for r in self.results],
            "detection_time_ms": self.detection_time_ms,
        }


# 缺陷识别 Prompt
_BUG_DETECT_PROMPT = """你是一位资深QA工程师，请判断以下模型输出是否存在缺陷。

用户输入：{user_input}
期望输出：{expected}
模型实际输出：{actual}

请判断模型输出是否存在缺陷，并以JSON格式返回：
```json
{{
  "is_bug": true/false,
  "category": "functional|performance|security|ux|accuracy|completeness|formatting|other",
  "severity": "P0|P1|P2|P3",
  "description": "缺陷描述（如无缺陷则写'无'）",
  "reproduce_steps": ["步骤1", "步骤2"],
  "confidence": 0.0-1.0
}}
```

判断标准：
- is_bug: 模型输出与期望存在实质差异，或包含明显错误
- category: functional=功能错误, accuracy=准确性(幻觉/错误信息), completeness=遗漏,
            security=安全问题, ux=体验差, performance=性能问题, formatting=格式问题
- severity: P0=阻断(完全不可用/安全漏洞), P1=严重(核心功能异常),
            P2=一般(非核心异常), P3=轻微(体验/格式)
- confidence: 判断置信度

只输出JSON，不要其他内容。
"""


class BugDetector:
    """AI缺陷识别器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter

    def _parse_result(self, raw: str) -> BugReport:
        """解析LLM输出的缺陷报告."""
        # 优先提取 ```json ... ``` 代码块
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw)
        json_str = code_block.group(1) if code_block else raw

        obj_start = json_str.find('{')
        obj_end = json_str.rfind('}')
        if obj_start == -1 or obj_end == -1:
            return BugReport(bug_id="unknown", is_bug=False, description="解析失败")

        try:
            data = json.loads(json_str[obj_start:obj_end + 1])
        except json.JSONDecodeError:
            return BugReport(bug_id="unknown", is_bug=False, description="JSON解析失败")

        try:
            category = BugCategory(data.get("category", "other"))
        except ValueError:
            category = BugCategory.OTHER

        try:
            severity = BugSeverity(data.get("severity", "P3"))
        except ValueError:
            severity = BugSeverity.P3

        return BugReport(
            bug_id="",
            is_bug=data.get("is_bug", False),
            category=category,
            severity=severity,
            description=data.get("description", ""),
            reproduce_steps=data.get("reproduce_steps", []),
            confidence=float(data.get("confidence", 0.5)),
        )

    def detect_one(
        self, user_input: str, expected: str, actual: str
    ) -> BugReport:
        """检测单条输出的缺陷."""
        prompt = _BUG_DETECT_PROMPT.format(
            user_input=user_input,
            expected=expected,
            actual=actual,
        )
        result = self._adapter.generate(prompt, max_tokens=1024)
        report = self._parse_result(result.text)
        report.actual_output = actual[:200]
        report.expected_output = expected[:200]
        return report

    def detect_batch(
        self,
        samples: List[Dict[str, str]],
    ) -> BugDetectionReport:
        """批量检测缺陷.

        Args:
            samples: 每个元素包含 user_input, expected, actual
        """
        import time

        start = time.time()
        report = BugDetectionReport(total_samples=len(samples))

        for i, sample in enumerate(samples):
            bug = self.detect_one(
                user_input=sample.get("user_input", ""),
                expected=sample.get("expected", ""),
                actual=sample.get("actual", ""),
            )
            bug.bug_id = f"bug_{i+1:03d}"
            report.results.append(bug)

            if bug.is_bug:
                report.bug_count += 1
                cat = bug.category.value
                sev = bug.severity.value
                report.by_category[cat] = report.by_category.get(cat, 0) + 1
                report.by_severity[sev] = report.by_severity.get(sev, 0) + 1
            else:
                report.no_bug_count += 1

        report.detection_time_ms = (time.time() - start) * 1000
        return report

    def evaluate_alignment(
        self,
        ai_results: List[BugReport],
        human_labels: List[bool],
    ) -> Dict[str, float]:
        """评估AI检测结果与人工标注的对齐度.

        Returns:
            precision, recall, f1, accuracy, cohens_kappa
        """
        if len(ai_results) != len(human_labels):
            raise ValueError("AI结果与人工标注数量不一致")

        tp = fp = tn = fn = 0
        for bug, human in zip(ai_results, human_labels):
            ai_bug = bug.is_bug
            if ai_bug and human:
                tp += 1
            elif ai_bug and not human:
                fp += 1
            elif not ai_bug and human:
                fn += 1
            else:
                tn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        accuracy = (tp + tn) / len(human_labels) if human_labels else 0.0

        # Cohen's Kappa
        total = len(human_labels)
        po = accuracy
        pe = ((tp + fp) * (tp + fn) + (tn + fn) * (tn + fp)) / (total * total) if total else 0.0
        kappa = (po - pe) / (1 - pe) if (1 - pe) else 0.0

        return {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "cohens_kappa": kappa,
            "tp": tp,
            "fp": fp,
            "tn": tn,
            "fn": fn,
        }

    def save_report(self, report: BugDetectionReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, "bug_detection.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: BugDetectionReport) -> None:
        """打印报告."""
        print(f"\n缺陷检测报告")
        print(f"  总样本数: {report.total_samples}")
        print(f"  缺陷数: {report.bug_count}")
        print(f"  正常数: {report.no_bug_count}")
        print(f"  缺陷率: {report.bug_rate:.1%}")
        print(f"  耗时: {report.detection_time_ms:.0f}ms")

        if report.by_category:
            print(f"\n  按分类:")
            for cat, count in sorted(report.by_category.items()):
                print(f"    {cat}: {count}")

        if report.by_severity:
            print(f"\n  按严重度:")
            for sev, count in sorted(report.by_severity.items()):
                print(f"    {sev}: {count}")

        # 打印P0/P1详情
        critical = [r for r in report.results if r.is_bug and r.severity in (BugSeverity.P0, BugSeverity.P1)]
        if critical:
            print(f"\n  严重缺陷详情:")
            for bug in critical:
                print(f"    [{bug.severity.value}][{bug.category.value}] {bug.description[:80]}")
                if bug.reproduce_steps:
                    for step in bug.reproduce_steps[:3]:
                        print(f"      - {step}")
