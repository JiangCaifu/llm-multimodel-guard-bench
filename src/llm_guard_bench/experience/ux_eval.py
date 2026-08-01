"""G2: 用户体验评测.

从真实用户视角评测模型回复质量，关注：
    - 响应质量：准确性、逻辑性、表达清晰度
    - 相关性：是否切题、是否回答了用户问题
    - 信息量：信息是否充分、是否有价值
    - 可读性：格式、结构、语言流畅度
    - 友好度：语气、礼貌、是否有帮助

不同于能力评测（答对没），UX评测关注"答得好不好"。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


@dataclass
class UXSample:
    """单条UX评测样本."""

    sample_id: str
    user_input: str
    model_output: str = ""
    # 各维度评分 (0-10)
    scores: Dict[str, float] = field(default_factory=dict)
    # 总体评分
    overall_score: float = 0.0
    # 问题标签
    issues: List[str] = field(default_factory=list)
    # 优化建议
    suggestion: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "user_input": self.user_input,
            "model_output": self.model_output,
            "scores": self.scores,
            "overall_score": self.overall_score,
            "issues": self.issues,
            "suggestion": self.suggestion,
        }


@dataclass
class UXReport:
    """UX评测报告."""

    model_name: str = ""
    total_samples: int = 0
    # 各维度均分
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    # 总体均分
    avg_overall: float = 0.0
    # 评分分布
    score_distribution: Dict[str, int] = field(default_factory=dict)
    # 问题统计
    issue_stats: Dict[str, int] = field(default_factory=dict)
    # 样本详情
    samples: List[UXSample] = field(default_factory=list)
    # 优化建议汇总
    top_issues: List[str] = field(default_factory=list)
    improvement_suggestions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "dimension_scores": self.dimension_scores,
            "avg_overall": self.avg_overall,
            "score_distribution": self.score_distribution,
            "issue_stats": self.issue_stats,
            "top_issues": self.top_issues,
            "improvement_suggestions": self.improvement_suggestions,
            "samples": [s.to_dict() for s in self.samples],
        }


# UX评测维度
UX_DIMENSIONS = [
    "quality",        # 响应质量
    "relevance",      # 相关性
    "completeness",   # 信息量/完整性
    "readability",    # 可读性
    "friendliness",   # 友好度
]

# UX评测 Prompt
_UX_EVAL_PROMPT = """请从用户体验角度评测以下模型回复。

用户输入：{user_input}
模型回复：{model_output}

请按以下维度评分（0-10分，保留一位小数）：
- quality: 响应质量（内容是否准确、逻辑是否清晰）
- relevance: 相关性（是否切题、是否直接回答了问题）
- completeness: 信息量（信息是否充分、是否有价值）
- readability: 可读性（格式、结构、语言是否流畅易读）
- friendliness: 友好度（语气是否友好、是否有帮助）

同时标注存在的问题（如有）：
- off_topic: 跑题
- too_short: 回复过短
- too_long: 回复过长
- unclear: 表达不清
- inaccurate: 内容不准确
- unhelpful: 没有帮助
- bad_format: 格式问题
- rude: 语气不佳

请给出一条优化建议。

只输出JSON：
```json
{{
  "scores": {{
    "quality": 8.0,
    "relevance": 9.0,
    "completeness": 7.0,
    "readability": 8.5,
    "friendliness": 7.5
  }},
  "issues": ["too_short"],
  "suggestion": "建议增加具体示例说明"
}}
```
"""

# 演示用测试输入
DEMO_UX_INPUTS = [
    "什么是机器学习？",
    "帮我写一封请假邮件",
    "Python和Java哪个好？",
    "如何提高写作能力？",
    "解释一下什么是API",
    "推荐一本好书",
    "怎么处理工作压力？",
    "什么是云计算？",
]


class UXEvaluator:
    """用户体验评测器."""

    def __init__(self, model_adapter: BaseModelAdapter, judge_adapter: BaseModelAdapter) -> None:
        self._model = model_adapter
        self._judge = judge_adapter

    def _evaluate_one(self, user_input: str, model_output: str) -> UXSample:
        """评测单条."""
        prompt = _UX_EVAL_PROMPT.format(user_input=user_input, model_output=model_output)
        result = self._judge.generate(prompt, max_tokens=512)

        # 解析JSON
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', result.text)
        json_str = code_block.group(1) if code_block else result.text

        obj_start = json_str.find('{')
        obj_end = json_str.rfind('}')
        if obj_start == -1 or obj_end == -1:
            return UXSample(
                sample_id="",
                user_input=user_input,
                model_output=model_output,
                scores={d: 5.0 for d in UX_DIMENSIONS},
                overall_score=5.0,
            )

        try:
            data = json.loads(json_str[obj_start:obj_end + 1])
        except json.JSONDecodeError:
            return UXSample(
                sample_id="",
                user_input=user_input,
                model_output=model_output,
                scores={d: 5.0 for d in UX_DIMENSIONS},
                overall_score=5.0,
            )

        scores = {k: float(v) for k, v in data.get("scores", {}).items()}
        # 补全缺失维度
        for d in UX_DIMENSIONS:
            if d not in scores:
                scores[d] = 5.0

        overall = sum(scores.values()) / len(scores)

        return UXSample(
            sample_id="",
            user_input=user_input,
            model_output=model_output,
            scores=scores,
            overall_score=overall,
            issues=data.get("issues", []),
            suggestion=data.get("suggestion", ""),
        )

    def evaluate(
        self,
        inputs: Optional[List[str]] = None,
        samples: Optional[List[Dict[str, str]]] = None,
    ) -> UXReport:
        """执行UX评测.

        Args:
            inputs: 用户输入列表（模型实时生成回复）
            samples: 已有样本 [{"user_input": ..., "model_output": ...}, ...]
        """
        report = UXReport(model_name=self._model.get_model_info().name)

        # 确定评测数据
        eval_samples = []
        if samples:
            for i, s in enumerate(samples):
                eval_samples.append((f"ux_{i+1:03d}", s["user_input"], s.get("model_output", "")))
        else:
            inputs = inputs or DEMO_UX_INPUTS
            for i, inp in enumerate(inputs):
                eval_samples.append((f"ux_{i+1:03d}", inp, ""))

        for sample_id, user_input, existing_output in eval_samples:
            # 生成模型回复（如果没有提供）
            model_output = existing_output
            if not model_output:
                result = self._model.generate(user_input, max_tokens=1024)
                model_output = result.text

            # 评测
            sample = self._evaluate_one(user_input, model_output)
            sample.sample_id = sample_id
            report.samples.append(sample)

        # 统计
        report.total_samples = len(report.samples)

        # 各维度均分
        for dim in UX_DIMENSIONS:
            values = [s.scores.get(dim, 0) for s in report.samples]
            report.dimension_scores[dim] = sum(values) / len(values) if values else 0.0

        # 总体均分
        report.avg_overall = sum(s.overall_score for s in report.samples) / report.total_samples if report.total_samples else 0.0

        # 评分分布
        for s in report.samples:
            bucket = self._score_bucket(s.overall_score)
            report.score_distribution[bucket] = report.score_distribution.get(bucket, 0) + 1

        # 问题统计
        for s in report.samples:
            for issue in s.issues:
                report.issue_stats[issue] = report.issue_stats.get(issue, 0) + 1

        # Top问题
        sorted_issues = sorted(report.issue_stats.items(), key=lambda x: x[1], reverse=True)
        report.top_issues = [f"{issue} ({count}次)" for issue, count in sorted_issues[:5]]

        # 优化建议
        report.improvement_suggestions = self._generate_suggestions(report)

        return report

    @staticmethod
    def _score_bucket(score: float) -> str:
        if score >= 8.0:
            return "优秀(8-10)"
        elif score >= 6.0:
            return "良好(6-8)"
        elif score >= 4.0:
            return "及格(4-6)"
        else:
            return "差(0-4)"

    @staticmethod
    def _generate_suggestions(report: UXReport) -> List[str]:
        """根据评测结果生成优化建议."""
        suggestions = []

        # 找最弱的维度
        if report.dimension_scores:
            weakest = min(report.dimension_scores.items(), key=lambda x: x[1])
            dim_names = {
                "quality": "响应质量",
                "relevance": "相关性",
                "completeness": "信息完整性",
                "readability": "可读性",
                "friendliness": "友好度",
            }
            suggestions.append(f"最弱维度: {dim_names.get(weakest[0], weakest[0])} (均分{weakest[1]:.1f})，建议优先改进")

        # 根据问题类型给建议
        issue_advice = {
            "off_topic": "模型存在跑题问题，建议优化提示词或增加约束",
            "too_short": "回复过短，建议增加输出长度要求",
            "too_long": "回复过长，建议增加精简要求",
            "unclear": "表达不清，建议优化输出格式和结构",
            "inaccurate": "内容不准确，建议增加事实核查",
            "unhelpful": "帮助性不足，建议增加实用建议",
            "bad_format": "格式问题，建议统一输出格式",
            "rude": "语气不佳，建议调整语气",
        }
        for issue, _ in sorted(report.issue_stats.items(), key=lambda x: x[1], reverse=True)[:3]:
            if issue in issue_advice:
                suggestions.append(issue_advice[issue])

        # 收集样本级建议
        sample_suggestions = [s.suggestion for s in report.samples if s.suggestion]
        if sample_suggestions:
            suggestions.append(f"样本级建议: {sample_suggestions[0]}")

        return suggestions

    def save_report(self, report: UXReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"ux_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: UXReport) -> None:
        """打印报告."""
        print(f"\n{'='*60}")
        print(f"用户体验评测报告 - {report.model_name}")
        print(f"{'='*60}")
        print(f"样本数: {report.total_samples}")
        print(f"总体均分: {report.avg_overall:.2f}/10")

        print(f"\n各维度评分:")
        dim_names = {
            "quality": "响应质量",
            "relevance": "相关性",
            "completeness": "信息完整性",
            "readability": "可读性",
            "friendliness": "友好度",
        }
        for dim in UX_DIMENSIONS:
            score = report.dimension_scores.get(dim, 0)
            bar = "█" * int(score) + "░" * (10 - int(score))
            print(f"  {dim_names.get(dim, dim):<12} {score:.1f} {bar}")

        print(f"\n评分分布:")
        for bucket, count in sorted(report.score_distribution.items()):
            print(f"  {bucket}: {count}个")

        if report.top_issues:
            print(f"\n主要问题:")
            for issue in report.top_issues:
                print(f"  - {issue}")

        if report.improvement_suggestions:
            print(f"\n优化建议:")
            for i, sug in enumerate(report.improvement_suggestions, 1):
                print(f"  {i}. {sug}")
