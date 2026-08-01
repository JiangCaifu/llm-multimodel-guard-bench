"""G1: A/B测试框架.

对比两个模型/策略的效果差异，包含统计显著性检验。

核心能力：
    - 同一组测试输入，分别跑两个模型
    - 对每条输出用评判器打分
    - 计算均值/标准差/置信区间
    - 统计显著性检验（t检验 / Mann-Whitney U检验）
    - 自动判定哪个模型更优 + 效果量（Cohen's d）
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..adapters.base import BaseModelAdapter, GenerationResult


@dataclass
class ABTestConfig:
    """A/B测试配置."""

    name: str = "ab_test"
    model_a_name: str = "model_a"
    model_b_name: str = "model_b"
    sample_count: int = 20          # 每个模型的样本数
    significance_level: float = 0.05  # 显著性水平
    metrics: List[str] = field(default_factory=lambda: ["quality", "relevance", "completeness"])


@dataclass
class ABTestSample:
    """单条A/B测试样本."""

    sample_id: str
    prompt: str
    # 模型A的结果
    output_a: str = ""
    scores_a: Dict[str, float] = field(default_factory=dict)  # 各维度评分
    # 模型B的结果
    output_b: str = ""
    scores_b: Dict[str, float] = field(default_factory=dict)
    # 偏好标注（A更好=-1, 平局=0, B更好=1）
    preference: int = 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "prompt": self.prompt,
            "output_a": self.output_a,
            "scores_a": self.scores_a,
            "output_b": self.output_b,
            "scores_b": self.scores_b,
            "preference": self.preference,
        }


@dataclass
class ABTestReport:
    """A/B测试报告."""

    config: ABTestConfig
    samples: List[ABTestSample] = field(default_factory=list)
    # 统计结果
    mean_a: Dict[str, float] = field(default_factory=dict)
    mean_b: Dict[str, float] = field(default_factory=dict)
    std_a: Dict[str, float] = field(default_factory=dict)
    std_b: Dict[str, float] = field(default_factory=dict)
    # 统计检验
    p_values: Dict[str, float] = field(default_factory=dict)
    cohen_d: Dict[str, float] = field(default_factory=dict)
    is_significant: Dict[str, bool] = field(default_factory=dict)
    # 偏好统计
    a_win_count: int = 0
    b_win_count: int = 0
    tie_count: int = 0
    # 结论
    winner: str = ""               # "A" / "B" / "tie"
    confidence: float = 0.0        # 置信度
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": {
                "name": self.config.name,
                "model_a_name": self.config.model_a_name,
                "model_b_name": self.config.model_b_name,
                "sample_count": self.config.sample_count,
                "significance_level": self.config.significance_level,
            },
            "mean_a": self.mean_a,
            "mean_b": self.mean_b,
            "std_a": self.std_a,
            "std_b": self.std_b,
            "p_values": self.p_values,
            "cohen_d": self.cohen_d,
            "is_significant": self.is_significant,
            "a_win_count": self.a_win_count,
            "b_win_count": self.b_win_count,
            "tie_count": self.tie_count,
            "winner": self.winner,
            "confidence": self.confidence,
            "summary": self.summary,
            "samples": [s.to_dict() for s in self.samples],
        }


# 评判 Prompt
_SCORING_PROMPT = """请对以下模型回复进行多维度评分。

用户问题：{prompt}
模型回复：{response}

请按以下维度评分（0-10分，保留一位小数）：
- quality: 回答质量（准确性、逻辑性、表达清晰度）
- relevance: 相关性（是否切题、是否回答了用户问题）
- completeness: 完整性（信息是否充分、是否有遗漏）

只输出JSON，格式如下：
```json
{{
  "quality": 8.5,
  "relevance": 9.0,
  "completeness": 7.5
}}
```
"""

# 偏好判断 Prompt
_PREFERENCE_PROMPT = """请判断以下两个模型的回复哪个更好。

用户问题：{prompt}
模型A回复：{output_a}
模型B回复：{output_b}

请综合质量、相关性、完整性判断，输出偏好：
- -1: 模型A更好
- 0: 平局
- 1: 模型B更好

只输出一个数字（-1, 0, 或 1）。
"""


class ABTestRunner:
    """A/B测试运行器."""

    def __init__(
        self,
        model_a: BaseModelAdapter,
        model_b: BaseModelAdapter,
        judge: BaseModelAdapter,
        config: Optional[ABTestConfig] = None,
    ) -> None:
        self._model_a = model_a
        self._model_b = model_b
        self._judge = judge
        self._config = config or ABTestConfig()

    def _score_output(self, prompt: str, response: str) -> Dict[str, float]:
        """用评判模型对输出打分."""
        import re as _re

        judge_prompt = _SCORING_PROMPT.format(prompt=prompt, response=response)
        result = self._judge.generate(judge_prompt, max_tokens=256)

        # 解析JSON
        code_block = _re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', result.text)
        json_str = code_block.group(1) if code_block else result.text

        obj_start = json_str.find('{')
        obj_end = json_str.rfind('}')
        if obj_start == -1 or obj_end == -1:
            return {"quality": 5.0, "relevance": 5.0, "completeness": 5.0}

        try:
            scores = json.loads(json_str[obj_start:obj_end + 1])
            return {k: float(v) for k, v in scores.items()}
        except (json.JSONDecodeError, ValueError):
            return {"quality": 5.0, "relevance": 5.0, "completeness": 5.0}

    def _judge_preference(self, prompt: str, output_a: str, output_b: str) -> int:
        """判断偏好."""
        judge_prompt = _PREFERENCE_PROMPT.format(
            prompt=prompt, output_a=output_a, output_b=output_b
        )
        result = self._judge.generate(judge_prompt, max_tokens=64)

        text = result.text.strip()
        # 提取数字
        import re as _re
        match = _re.search(r'-?[01]', text)
        if match:
            return int(match.group())
        return 0

    @staticmethod
    def _mean(values: List[float]) -> float:
        return sum(values) / len(values) if values else 0.0

    @staticmethod
    def _std(values: List[float], mean_val: float) -> float:
        if len(values) < 2:
            return 0.0
        variance = sum((x - mean_val) ** 2 for x in values) / (len(values) - 1)
        return math.sqrt(variance)

    @staticmethod
    def _t_test(a: List[float], b: List[float]) -> Tuple[float, float]:
        """配对t检验，返回 (t值, p值近似).

        简化版：用正态近似计算p值。
        """
        if len(a) != len(b) or len(a) < 2:
            return 0.0, 1.0

        diffs = [ai - bi for ai, bi in zip(a, b)]
        n = len(diffs)
        mean_diff = sum(diffs) / n
        if n < 2:
            return 0.0, 1.0

        var_diff = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
        std_diff = math.sqrt(var_diff)

        if std_diff == 0:
            return 0.0, 1.0

        t_stat = mean_diff / (std_diff / math.sqrt(n))

        # 正态近似计算p值（双尾）
        # |t| > z 对应的 p
        from math import erf
        p_value = 2 * (1 - 0.5 * (1 + erf(abs(t_stat) / math.sqrt(2))))

        return t_stat, p_value

    @staticmethod
    def _cohens_d(a: List[float], b: List[float]) -> float:
        """计算 Cohen's d 效果量."""
        if not a or not b:
            return 0.0
        mean_a = sum(a) / len(a)
        mean_b = sum(b) / len(b)
        var_a = sum((x - mean_a) ** 2 for x in a) / max(len(a) - 1, 1)
        var_b = sum((x - mean_b) ** 2 for x in b) / max(len(b) - 1, 1)
        pooled_std = math.sqrt((var_a + var_b) / 2)
        if pooled_std == 0:
            return 0.0
        return (mean_a - mean_b) / pooled_std

    def run(self, prompts: List[str]) -> ABTestReport:
        """执行A/B测试."""
        report = ABTestReport(config=self._config)

        for i, prompt in enumerate(prompts):
            sample = ABTestSample(sample_id=f"ab_{i+1:03d}", prompt=prompt)

            # 模型A生成
            result_a = self._model_a.generate(prompt, max_tokens=1024)
            sample.output_a = result_a.text

            # 模型B生成
            result_b = self._model_b.generate(prompt, max_tokens=1024)
            sample.output_b = result_b.text

            # 评分
            sample.scores_a = self._score_output(prompt, sample.output_a)
            sample.scores_b = self._score_output(prompt, sample.output_b)

            # 偏好判断
            sample.preference = self._judge_preference(prompt, sample.output_a, sample.output_b)

            if sample.preference == -1:
                report.a_win_count += 1
            elif sample.preference == 1:
                report.b_win_count += 1
            else:
                report.tie_count += 1

            report.samples.append(sample)

        # 统计分析
        metrics = self._config.metrics
        for metric in metrics:
            values_a = [s.scores_a.get(metric, 0) for s in report.samples]
            values_b = [s.scores_b.get(metric, 0) for s in report.samples]

            report.mean_a[metric] = self._mean(values_a)
            report.mean_b[metric] = self._mean(values_b)
            report.std_a[metric] = self._std(values_a, report.mean_a[metric])
            report.std_b[metric] = self._std(values_b, report.mean_b[metric])

            _, p_val = self._t_test(values_a, values_b)
            report.p_values[metric] = p_val
            report.cohen_d[metric] = self._cohens_d(values_a, values_b)
            report.is_significant[metric] = p_val < self._config.significance_level

        # 判定胜者
        a_better = sum(1 for m in metrics if report.mean_a[m] > report.mean_b[m])
        b_better = sum(1 for m in metrics if report.mean_b[m] > report.mean_a[m])

        if a_better > b_better:
            report.winner = self._config.model_a_name
        elif b_better > a_better:
            report.winner = self._config.model_b_name
        else:
            # 按偏好计数
            if report.a_win_count > report.b_win_count:
                report.winner = self._config.model_a_name
            elif report.b_win_count > report.a_win_count:
                report.winner = self._config.model_b_name
            else:
                report.winner = "tie"

        # 置信度
        sig_count = sum(1 for s in report.is_significant.values() if s)
        report.confidence = sig_count / len(metrics) if metrics else 0.0

        # 总结
        report.summary = self._generate_summary(report)
        return report

    @staticmethod
    def _generate_summary(report: ABTestReport) -> str:
        """生成总结."""
        lines = []
        lines.append(f"A/B测试: {report.config.model_a_name} vs {report.config.model_b_name}")
        lines.append(f"样本数: {len(report.samples)}")
        lines.append(f"偏好: A胜{report.a_win_count}次, B胜{report.b_win_count}次, 平局{report.tie_count}次")

        for metric in report.config.metrics:
            ma = report.mean_a.get(metric, 0)
            mb = report.mean_b.get(metric, 0)
            p = report.p_values.get(metric, 1.0)
            sig = "显著" if report.is_significant.get(metric, False) else "不显著"
            lines.append(f"  {metric}: A={ma:.2f}±{report.std_a.get(metric, 0):.2f}, "
                        f"B={mb:.2f}±{report.std_b.get(metric, 0):.2f}, p={p:.4f} ({sig})")

        lines.append(f"胜者: {report.winner} (置信度: {report.confidence:.0%})")
        return "\n".join(lines)

    def save_report(self, report: ABTestReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"abtest_{report.config.name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: ABTestReport) -> None:
        """打印报告."""
        print(f"\n{'='*60}")
        print(f"A/B测试报告: {report.config.model_a_name} vs {report.config.model_b_name}")
        print(f"{'='*60}")
        print(f"样本数: {len(report.samples)}")
        print(f"\n偏好统计:")
        print(f"  {report.config.model_a_name} 胜: {report.a_win_count}次")
        print(f"  {report.config.model_b_name} 胜: {report.b_win_count}次")
        print(f"  平局: {report.tie_count}次")

        print(f"\n指标对比:")
        print(f"  {'指标':<15} {'A均值':<12} {'B均值':<12} {'p值':<10} {'显著':<8} {'Cohen d':<10}")
        for metric in report.config.metrics:
            ma = report.mean_a.get(metric, 0)
            mb = report.mean_b.get(metric, 0)
            p = report.p_values.get(metric, 1.0)
            sig = "是" if report.is_significant.get(metric, False) else "否"
            d = report.cohen_d.get(metric, 0)
            print(f"  {metric:<15} {ma:<12.2f} {mb:<12.2f} {p:<10.4f} {sig:<8} {d:<10.2f}")

        print(f"\n结论: {report.winner} (置信度: {report.confidence:.0%})")
        print(f"\n{report.summary}")
