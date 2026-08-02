"""人工标注工具.

支持两种使用方式：
    1. CLI交互式标注（逐条展示，键盘输入标签）
    2. 加载/保存标注数据（JSON格式）

标注 schema：
    {
      "sample_id": "s001",
      "user_input": "...",
      "model_output": "...",
      "judge_label": "correct",      # Judge给的标签
      "human_label": "correct",      # 人工标注的标签
      "notes": "..."                 # 可选备注
    }
"""
from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field, asdict
from typing import Any, Dict, List, Optional


# 标准标签集（可扩展）
DEFAULT_LABELS = ["correct", "incorrect", "partial", "unclear"]


@dataclass
class AnnotationSample:
    """标注样本."""

    sample_id: str
    user_input: str
    model_output: str
    judge_label: str = ""       # Judge判定
    human_label: str = ""       # 人工标注
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class AnnotationDataset:
    """标注数据集."""

    samples: List[AnnotationSample] = field(default_factory=list)
    labels: List[str] = field(default_factory=lambda: DEFAULT_LABELS.copy())
    task_type: str = "general"  # general / safety / ux
    annotator: str = "default"
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_type": self.task_type,
            "annotator": self.annotator,
            "labels": self.labels,
            "notes": self.notes,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AnnotationDataset":
        samples = [AnnotationSample(**s) for s in data.get("samples", [])]
        return cls(
            samples=samples,
            labels=data.get("labels", DEFAULT_LABELS.copy()),
            task_type=data.get("task_type", "general"),
            annotator=data.get("annotator", "default"),
            notes=data.get("notes", ""),
        )


class Annotator:
    """人工标注工具."""

    def __init__(self, labels: Optional[List[str]] = None) -> None:
        self._labels = labels or DEFAULT_LABELS.copy()

    @property
    def labels(self) -> List[str]:
        return self._labels

    def build_samples(
        self,
        inputs: List[str],
        outputs: List[str],
        judge_labels: Optional[List[str]] = None,
        shuffle: bool = False,
    ) -> AnnotationDataset:
        """从模型输出构建待标注样本.

        Args:
            inputs: 用户输入列表
            outputs: 模型输出列表
            judge_labels: Judge给的标签（如有）
            shuffle: 是否打乱顺序（避免标注者按顺序记忆）
        """
        if len(inputs) != len(outputs):
            raise ValueError(f"inputs({len(inputs)})和outputs({len(outputs)})长度不一致")

        samples = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            samples.append(AnnotationSample(
                sample_id=f"s{i+1:03d}",
                user_input=inp,
                model_output=out,
                judge_label=judge_labels[i] if judge_labels else "",
            ))

        if shuffle:
            random.shuffle(samples)

        return AnnotationDataset(samples=samples, labels=self._labels)

    def annotate_interactive(self, dataset: AnnotationDataset) -> AnnotationDataset:
        """交互式标注（CLI）.

        每条样本展示输入/输出/Judge标签，让标注者输入人工标签。
        """
        print(f"\n{'='*60}")
        print(f"人工标注工具")
        print(f"{'='*60}")
        print(f"标签集: {self._labels}")
        print(f"待标注样本: {len(dataset.samples)}条")
        print(f"操作: 输入标签名 / 数字快捷键 / s=跳过 / q=保存退出 / h=帮助")
        print(f"{'='*60}\n")

        for i, sample in enumerate(dataset.samples):
            if sample.human_label:  # 跳过已标注
                continue

            print(f"\n[{i+1}/{len(dataset.samples)}] 样本 {sample.sample_id}")
            print(f"用户输入: {sample.user_input[:200]}")
            print(f"模型输出: {sample.model_output[:500]}")
            if sample.judge_label:
                print(f"Judge判定: {sample.judge_label}")
            print(f"可选标签: {self._labels}")

            while True:
                choice = input("你的标注> ").strip().lower()
                if choice == "q":
                    print("保存并退出...")
                    return dataset
                if choice == "s":
                    print("跳过")
                    break
                if choice == "h":
                    print(f"标签: {self._labels} | s=跳过 | q=保存退出")
                    continue
                # 数字快捷键
                if choice.isdigit():
                    idx = int(choice) - 1
                    if 0 <= idx < len(self._labels):
                        sample.human_label = self._labels[idx]
                        print(f"已标注: {sample.human_label}")
                        break
                    else:
                        print(f"数字超出范围，请输入1-{len(self._labels)}")
                        continue
                # 标签名
                if choice in self._labels:
                    sample.human_label = choice
                    print(f"已标注: {sample.human_label}")
                    break
                print(f"无效输入，可选: {self._labels}")

        print(f"\n标注完成！")
        return dataset

    def save(self, dataset: AnnotationDataset, path: str) -> str:
        """保存标注数据."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def load(path: str) -> AnnotationDataset:
        """加载标注数据."""
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return AnnotationDataset.from_dict(data)

    @staticmethod
    def demo_dataset(n: int = 10) -> AnnotationDataset:
        """生成演示用标注数据（模拟人工标注，用于验证对齐指标计算）.

        构造真实的多类别分布：
            - 人工标签: 7 correct + 3 incorrect (10条)
            - Judge标签: 8正确 + 2错误
                - 1条 correct 误判为 incorrect (假阳性)
                - 1条 incorrect 误判为 correct (假阴性)
        预期 Kappa ≈ 0.62 (良好)，能验证指标计算逻辑
        """
        all_inputs = [
            "什么是机器学习？", "帮我写一封请假邮件", "Python和Java哪个好？",
            "如何提高写作能力？", "解释一下什么是API", "推荐一本好书",
            "怎么处理工作压力？", "什么是云计算？", "如何学习算法？", "什么是数据库？",
        ]
        inputs = all_inputs[:n]
        outputs = [f"这是关于{inp}的回答..." for inp in inputs]

        # 人工标签（"真实"标签）：前7条correct，后3条incorrect
        human_labels = ["correct"] * 7 + ["incorrect"] * 3
        human_labels = human_labels[:n]

        # Judge标签：8正确 + 2错误
        # s1-s6: correct 判对
        # s7:   correct 误判为 incorrect (假阳性)
        # s8-s9: incorrect 判对
        # s10:  incorrect 误判为 correct (假阴性)
        judge_labels = ["correct"] * 6 + ["incorrect", "incorrect", "incorrect", "correct"]
        judge_labels = judge_labels[:n]

        samples = []
        for i, (inp, out) in enumerate(zip(inputs, outputs)):
            samples.append(AnnotationSample(
                sample_id=f"s{i+1:03d}",
                user_input=inp,
                model_output=out,
                judge_label=judge_labels[i],
                human_label=human_labels[i],
            ))

        return AnnotationDataset(samples=samples, labels=DEFAULT_LABELS.copy())

    @staticmethod
    def print_stats(dataset: AnnotationDataset) -> None:
        """打印标注统计."""
        total = len(dataset.samples)
        annotated = sum(1 for s in dataset.samples if s.human_label)
        print(f"\n{'='*40}")
        print(f"标注统计")
        print(f"{'='*40}")
        print(f"总样本: {total}")
        print(f"已标注: {annotated}")
        print(f"未标注: {total - annotated}")
        if annotated > 0:
            print(f"\n标签分布:")
            for label in dataset.labels:
                count = sum(1 for s in dataset.samples if s.human_label == label)
                pct = count / annotated * 100
                bar = "█" * int(pct / 5)
                print(f"  {label:<12} {count:>3} ({pct:.1f}%) {bar}")
