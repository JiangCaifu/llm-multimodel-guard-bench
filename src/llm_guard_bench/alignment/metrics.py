"""对齐指标计算.

对比 LLM-as-Judge 判定与人工判定，计算一致性指标：

    - Accuracy: 整体准确率
    - Precision / Recall / F1: 针对每个类别
    - Cohen's Kappa: 排除随机一致性的kappa系数
        < 0.2   极差
        0.2-0.4 一般
        0.4-0.6 中等
        0.6-0.8 良好
        > 0.8   优秀
    - 混淆矩阵: 看Judge在哪类样本上容易判错

参考:
    Cohen's Kappa: https://en.wikipedia.org/wiki/Cohen%27s_kappa
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ClassMetrics:
    """单类别指标."""

    label: str
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    support: int = 0  # 该类别人工标注样本数

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "support": self.support,
        }


@dataclass
class AlignmentReport:
    """对齐分析报告."""

    total_samples: int = 0
    annotated_samples: int = 0
    accuracy: float = 0.0
    cohen_kappa: float = 0.0
    kappa_quality: str = ""           # 极差/一般/中等/良好/优秀
    per_class: Dict[str, ClassMetrics] = field(default_factory=dict)
    macro_f1: float = 0.0
    weighted_f1: float = 0.0
    confusion_matrix: List[List[int]] = field(default_factory=list)
    labels: List[str] = field(default_factory=list)
    disagreements: List[Dict[str, Any]] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "annotated_samples": self.annotated_samples,
            "accuracy": round(self.accuracy, 4),
            "cohen_kappa": round(self.cohen_kappa, 4),
            "kappa_quality": self.kappa_quality,
            "per_class": {k: v.to_dict() for k, v in self.per_class.items()},
            "macro_f1": round(self.macro_f1, 4),
            "weighted_f1": round(self.weighted_f1, 4),
            "confusion_matrix": self.confusion_matrix,
            "labels": self.labels,
            "disagreements": self.disagreements,
            "summary": self.summary,
        }


def compute_accuracy(judge_labels: List[str], human_labels: List[str]) -> float:
    """计算准确率."""
    if not judge_labels:
        return 0.0
    correct = sum(1 for j, h in zip(judge_labels, human_labels) if j == h)
    return correct / len(judge_labels)


def compute_confusion_matrix(
    judge_labels: List[str],
    human_labels: List[str],
    labels: Optional[List[str]] = None,
) -> Tuple[List[List[int]], List[str]]:
    """计算混淆矩阵.

    返回 (matrix, labels)，matrix[i][j] 表示真实标签为labels[i]、Judge判为labels[j]的数量。
    """
    if labels is None:
        labels = sorted(set(judge_labels) | set(human_labels))

    label_to_idx = {l: i for i, l in enumerate(labels)}
    n = len(labels)
    matrix = [[0] * n for _ in range(n)]

    for j, h in zip(judge_labels, human_labels):
        # 行=human(真实), 列=judge(预测)
        if j in label_to_idx and h in label_to_idx:
            matrix[label_to_idx[h]][label_to_idx[j]] += 1

    return matrix, labels


def compute_per_class_metrics(
    judge_labels: List[str],
    human_labels: List[str],
    labels: Optional[List[str]] = None,
) -> Dict[str, ClassMetrics]:
    """计算每个类别的 P/R/F1.

    约定：human_labels是真实标签，judge_labels是预测标签。
    """
    if labels is None:
        labels = sorted(set(judge_labels) | set(human_labels))

    metrics: Dict[str, ClassMetrics] = {}
    for label in labels:
        tp = sum(1 for j, h in zip(judge_labels, human_labels) if j == label and h == label)
        fp = sum(1 for j, h in zip(judge_labels, human_labels) if j == label and h != label)
        fn = sum(1 for j, h in zip(judge_labels, human_labels) if j != label and h == label)
        support = sum(1 for h in human_labels if h == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        metrics[label] = ClassMetrics(
            label=label,
            precision=precision,
            recall=recall,
            f1=f1,
            support=support,
        )

    return metrics


def compute_cohen_kappa(
    judge_labels: List[str],
    human_labels: List[str],
    labels: Optional[List[str]] = None,
) -> float:
    """计算 Cohen's Kappa 系数.

    κ = (p_o - p_e) / (1 - p_e)
    其中 p_o 是观察一致率，p_e 是随机一致概率。
    """
    if len(judge_labels) != len(human_labels) or len(judge_labels) == 0:
        return 0.0

    if labels is None:
        labels = sorted(set(judge_labels) | set(human_labels))

    n = len(judge_labels)
    matrix, _ = compute_confusion_matrix(judge_labels, human_labels, labels)

    # p_o: 观察一致率（对角线之和 / 总数）
    p_o = sum(matrix[i][i] for i in range(len(labels))) / n

    # p_e: 随机一致概率
    # p_e = sum(human_label_i比例 * judge_label_i比例)
    human_counts = [0] * len(labels)
    judge_counts = [0] * len(labels)
    label_to_idx = {l: i for i, l in enumerate(labels)}
    for h, j in zip(human_labels, judge_labels):
        if h in label_to_idx:
            human_counts[label_to_idx[h]] += 1
        if j in label_to_idx:
            judge_counts[label_to_idx[j]] += 1

    p_e = sum((human_counts[i] / n) * (judge_counts[i] / n) for i in range(len(labels)))

    if p_e == 1.0:
        return 1.0  # 完全一致

    return (p_o - p_e) / (1 - p_e)


def kappa_quality(kappa: float) -> str:
    """Kappa值的质量评级."""
    if kappa < 0.2:
        return "极差"
    elif kappa < 0.4:
        return "一般"
    elif kappa < 0.6:
        return "中等"
    elif kappa < 0.8:
        return "良好"
    else:
        return "优秀"


def find_disagreements(
    samples,  # List[AnnotationSample]
    max_count: int = 20,
) -> List[Dict[str, Any]]:
    """找出Judge和人工判定不一致的样本."""
    disagreements = []
    for s in samples:
        if not s.human_label or not s.judge_label:
            continue
        if s.human_label != s.judge_label:
            disagreements.append({
                "sample_id": s.sample_id,
                "user_input": s.user_input[:200],
                "model_output": s.model_output[:300],
                "judge_label": s.judge_label,
                "human_label": s.human_label,
                "notes": s.notes,
            })
        if len(disagreements) >= max_count:
            break
    return disagreements


def compute_alignment(
    judge_labels: List[str],
    human_labels: List[str],
    samples=None,
    labels: Optional[List[str]] = None,
) -> AlignmentReport:
    """计算完整的对齐报告.

    Args:
        judge_labels: Judge判定标签列表
        human_labels: 人工标注标签列表
        samples: 原始样本（用于找不一致样本）
        labels: 标签集合
    """
    report = AlignmentReport()
    report.total_samples = len(judge_labels)
    report.annotated_samples = sum(1 for h in human_labels if h)

    # 只对有人工标注的样本计算
    paired_judge = []
    paired_human = []
    for j, h in zip(judge_labels, human_labels):
        if h:  # 有人工标注
            paired_judge.append(j)
            paired_human.append(h)

    if not paired_judge:
        report.summary = "无有效标注数据"
        return report

    # 确定标签集
    if labels is None:
        labels = sorted(set(paired_judge) | set(paired_human))
    report.labels = labels

    # 1. Accuracy
    report.accuracy = compute_accuracy(paired_judge, paired_human)

    # 2. Cohen's Kappa
    report.cohen_kappa = compute_cohen_kappa(paired_judge, paired_human, labels)
    report.kappa_quality = kappa_quality(report.cohen_kappa)

    # 3. 每个类别的 P/R/F1
    report.per_class = compute_per_class_metrics(paired_judge, paired_human, labels)

    # 4. Macro / Weighted F1
    f1_values = [m.f1 for m in report.per_class.values()]
    report.macro_f1 = sum(f1_values) / len(f1_values) if f1_values else 0.0

    total_support = sum(m.support for m in report.per_class.values())
    if total_support > 0:
        report.weighted_f1 = sum(m.f1 * m.support for m in report.per_class.values()) / total_support
    else:
        report.weighted_f1 = 0.0

    # 5. 混淆矩阵
    matrix, labels = compute_confusion_matrix(paired_judge, paired_human, labels)
    report.confusion_matrix = matrix
    report.labels = labels

    # 6. 不一致样本
    if samples is not None:
        report.disagreements = find_disagreements(samples)

    # 7. 总结
    report.summary = _generate_summary(report)

    return report


def _generate_summary(report: AlignmentReport) -> str:
    """生成总结."""
    lines = []
    lines.append(f"对齐分析报告")
    lines.append(f"样本数: {report.annotated_samples}")
    lines.append(f"Accuracy: {report.accuracy:.1%}")
    lines.append(f"Cohen's Kappa: {report.cohen_kappa:.3f} ({report.kappa_quality})")

    if report.cohen_kappa < 0.4:
        lines.append("⚠ Judge可靠性不足，建议优化评判Prompt或增加规则")
    elif report.cohen_kappa < 0.6:
        lines.append("⚠ Judge可靠性中等，部分场景需谨慎使用")
    else:
        lines.append("✓ Judge可靠性良好，可用于自动化评测")

    if report.disagreements:
        lines.append(f"不一致样本: {len(report.disagreements)}个，建议人工复核")

    return "\n".join(lines)


def print_report(report: AlignmentReport) -> None:
    """打印报告."""
    print(f"\n{'='*60}")
    print(f"Judge对齐分析报告")
    print(f"{'='*60}")
    print(f"样本数: {report.annotated_samples}")
    print(f"Accuracy: {report.accuracy:.1%}")
    print(f"Cohen's Kappa: {report.cohen_kappa:.3f} ({report.kappa_quality})")
    print(f"Macro F1: {report.macro_f1:.3f}")
    print(f"Weighted F1: {report.weighted_f1:.3f}")

    # 各类别指标
    print(f"\n各类别指标:")
    print(f"  {'标签':<12} {'P':<10} {'R':<10} {'F1':<10} {'support':<10}")
    for label in report.labels:
        m = report.per_class.get(label)
        if m:
            print(f"  {label:<12} {m.precision:<10.3f} {m.recall:<10.3f} {m.f1:<10.3f} {m.support:<10}")

    # 混淆矩阵
    print(f"\n混淆矩阵 (行=真实, 列=Judge):")
    header = "  " + " ".join(f"{l[:8]:>10}" for l in report.labels)
    print(header)
    for i, label in enumerate(report.labels):
        row = "  " + " ".join(f"{report.confusion_matrix[i][j]:>10}" for j in range(len(report.labels)))
        print(f"{label[:8]:<10}{row[2:]}")

    # 不一致样本
    if report.disagreements:
        print(f"\n不一致样本 ({len(report.disagreements)}个):")
        for d in report.disagreements[:5]:
            print(f"  [{d['sample_id']}] Judge={d['judge_label']} → 人工={d['human_label']}")
            print(f"    输入: {d['user_input'][:80]}")
        if len(report.disagreements) > 5:
            print(f"  ... 还有 {len(report.disagreements) - 5} 个")

    print(f"\n{report.summary}")
