"""G3: 优化闭环.

根据评测结果自动生成优化建议，并验证优化效果。

核心流程：
    1. 分析评测结果（UX/安全/性能等报告）
    2. LLM生成优化建议（Prompt工程/参数调整/后处理）
    3. 应用优化策略（自动或人工确认）
    4. 重新评测
    5. 对比前后效果，形成闭环
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


@dataclass
class OptimizationSuggestion:
    """单条优化建议."""

    suggestion_id: str
    category: str          # prompt_engineering / param_tuning / post_processing / system_prompt
    description: str = ""
    expected_improvement: str = ""
    priority: str = "P2"   # P0/P1/P2/P3
    action: str = ""       # 具体操作

    def to_dict(self) -> Dict[str, Any]:
        return {
            "suggestion_id": self.suggestion_id,
            "category": self.category,
            "description": self.description,
            "expected_improvement": self.expected_improvement,
            "priority": self.priority,
            "action": self.action,
        }


@dataclass
class OptimizationReport:
    """优化闭环报告."""

    model_name: str = ""
    # 优化前
    before_scores: Dict[str, float] = field(default_factory=dict)
    # 优化建议
    suggestions: List[OptimizationSuggestion] = field(default_factory=list)
    # 应用的优化策略
    applied_strategies: List[str] = field(default_factory=list)
    # 优化后
    after_scores: Dict[str, float] = field(default_factory=dict)
    # 效果对比
    improvements: Dict[str, float] = field(default_factory=dict)
    # 闭环结论
    is_improved: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "before_scores": self.before_scores,
            "suggestions": [s.to_dict() for s in self.suggestions],
            "applied_strategies": self.applied_strategies,
            "after_scores": self.after_scores,
            "improvements": self.improvements,
            "is_improved": self.is_improved,
            "summary": self.summary,
        }


# 优化建议生成 Prompt
_SUGGESTION_GEN_PROMPT = """你是一位大模型优化专家，请根据以下评测结果生成优化建议。

模型名称：{model_name}
评测维度及得分：
{scores_text}

主要问题：
{issues_text}

请生成3-5条优化建议，按优先级排序。每条建议包括：
1. category: 优化类别 (prompt_engineering/param_tuning/post_processing/system_prompt)
2. description: 问题描述
3. expected_improvement: 预期改进效果
4. priority: 优先级 (P0/P1/P2/P3)
5. action: 具体操作（如修改system prompt、调整temperature等）

只输出JSON数组：
```json
[
  {{
    "category": "prompt_engineering",
    "description": "回复过于简短，缺乏细节",
    "expected_improvement": "增加回复详细度，信息量评分提升1-2分",
    "priority": "P1",
    "action": "在system prompt中增加'请提供详细、具体的回答，包含示例和解释'"
  }}
]
```
"""

# 优化策略（内置）
BUILTIN_STRATEGIES = {
    "add_detail_instruction": {
        "name": "增加详细度指令",
        "description": "在system prompt中要求更详细的回答",
        "applies_to": ["completeness", "too_short"],
        "action": "system_prompt += '\\n请提供详细、具体的回答，包含必要的示例和解释。'",
    },
    "add_structure_instruction": {
        "name": "增加结构化指令",
        "description": "要求分点回答，提升可读性",
        "applies_to": ["readability", "bad_format", "unclear"],
        "action": "system_prompt += '\\n请使用分点、分段的格式回答，保持结构清晰。'",
    },
    "add_relevance_instruction": {
        "name": "增加相关性约束",
        "description": "要求直接回答问题，避免跑题",
        "applies_to": ["relevance", "off_topic"],
        "action": "system_prompt += '\\n请直接回答用户问题，不要偏离主题。'",
    },
    "add_friendliness_instruction": {
        "name": "增加友好度指令",
        "description": "调整语气，更友好有帮助",
        "applies_to": ["friendliness", "rude", "unhelpful"],
        "action": "system_prompt += '\\n请用友好、有帮助的语气回答，主动提供实用建议。'",
    },
    "increase_max_tokens": {
        "name": "增加输出长度",
        "description": "增加max_tokens避免截断",
        "applies_to": ["completeness", "too_short"],
        "action": "max_tokens: 1024 → 2048",
    },
    "adjust_temperature": {
        "name": "调整温度参数",
        "description": "降低温度提升准确性",
        "applies_to": ["quality", "inaccurate"],
        "action": "temperature: 0.7 → 0.3",
    },
}


class OptimizationLoop:
    """优化闭环."""

    def __init__(self, judge_adapter: BaseModelAdapter) -> None:
        self._judge = judge_adapter

    def analyze_and_suggest(
        self,
        model_name: str,
        scores: Dict[str, float],
        issues: List[str],
    ) -> List[OptimizationSuggestion]:
        """分析评测结果并生成优化建议."""
        # LLM生成建议
        scores_text = "\n".join(f"  - {k}: {v:.1f}/10" for k, v in scores.items())
        issues_text = "\n".join(f"  - {i}" for i in issues) if issues else "  无明显问题"

        prompt = _SUGGESTION_GEN_PROMPT.format(
            model_name=model_name,
            scores_text=scores_text,
            issues_text=issues_text,
        )
        result = self._judge.generate(prompt, max_tokens=1024)

        # 解析JSON
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', result.text)
        json_str = code_block.group(1) if code_block else result.text

        arr_start = json_str.find('[')
        arr_end = json_str.rfind(']')
        if arr_start == -1 or arr_end == -1:
            return self._fallback_suggestions(scores, issues)

        try:
            items = json.loads(json_str[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            return self._fallback_suggestions(scores, issues)

        suggestions = []
        for i, item in enumerate(items):
            suggestions.append(OptimizationSuggestion(
                suggestion_id=f"opt_{i+1:03d}",
                category=item.get("category", "prompt_engineering"),
                description=item.get("description", ""),
                expected_improvement=item.get("expected_improvement", ""),
                priority=item.get("priority", "P2"),
                action=item.get("action", ""),
            ))

        return suggestions

    @staticmethod
    def _fallback_suggestions(
        scores: Dict[str, float], issues: List[str]
    ) -> List[OptimizationSuggestion]:
        """基于规则的降级建议."""
        suggestions = []
        # 找最弱维度
        if scores:
            weakest = min(scores.items(), key=lambda x: x[1])
            weak_dim = weakest[0]

            # 匹配策略
            for strategy_id, strategy in BUILTIN_STRATEGIES.items():
                if weak_dim in strategy["applies_to"] or any(i in strategy["applies_to"] for i in issues):
                    suggestions.append(OptimizationSuggestion(
                        suggestion_id=f"opt_{len(suggestions)+1:03d}",
                        category="prompt_engineering",
                        description=strategy["description"],
                        expected_improvement=f"{weak_dim}评分预计提升1-2分",
                        priority="P1",
                        action=strategy["action"],
                    ))

        if not suggestions:
            suggestions.append(OptimizationSuggestion(
                suggestion_id="opt_001",
                category="prompt_engineering",
                description="综合优化",
                expected_improvement="整体提升1-2分",
                priority="P2",
                action="优化system prompt，增加输出要求",
            ))

        return suggestions[:5]

    @staticmethod
    def select_strategies(
        suggestions: List[OptimizationSuggestion],
        issues: List[str],
    ) -> List[str]:
        """根据建议选择要应用的策略."""
        strategies = []
        for sug in suggestions:
            # 匹配内置策略
            for strategy_id, strategy in BUILTIN_STRATEGIES.items():
                if any(s in sug.action.lower() or s in sug.description for s in strategy["applies_to"]):
                    if strategy_id not in strategies:
                        strategies.append(strategy_id)
                        break

        # 根据问题直接匹配
        for issue in issues:
            for strategy_id, strategy in BUILTIN_STRATEGIES.items():
                if issue in strategy["applies_to"] and strategy_id not in strategies:
                    strategies.append(strategy_id)

        return strategies[:3]  # 最多应用3个策略

    @staticmethod
    def compare_scores(
        before: Dict[str, float],
        after: Dict[str, float],
    ) -> Dict[str, float]:
        """对比前后评分."""
        improvements = {}
        for key in set(list(before.keys()) + list(after.keys())):
            b = before.get(key, 0)
            a = after.get(key, 0)
            improvements[key] = a - b
        return improvements

    def run_loop(
        self,
        model_name: str,
        before_scores: Dict[str, float],
        issues: List[str],
        after_scores: Optional[Dict[str, float]] = None,
    ) -> OptimizationReport:
        """执行完整优化闭环.

        Args:
            model_name: 模型名
            before_scores: 优化前评分
            issues: 问题列表
            after_scores: 优化后评分（如提供则计算改进效果）
        """
        report = OptimizationReport(model_name=model_name)
        report.before_scores = before_scores.copy()

        # 1. 生成建议
        report.suggestions = self.analyze_and_suggest(model_name, before_scores, issues)

        # 2. 选择策略
        report.applied_strategies = self.select_strategies(report.suggestions, issues)

        # 3. 如果有优化后评分，对比效果
        if after_scores:
            report.after_scores = after_scores.copy()
            report.improvements = self.compare_scores(before_scores, after_scores)

            # 判断是否改进
            positive_count = sum(1 for v in report.improvements.values() if v > 0)
            negative_count = sum(1 for v in report.improvements.values() if v < 0)
            report.is_improved = positive_count > negative_count

            # 生成总结
            report.summary = self._generate_summary(report)
        else:
            # 只生成建议，标注待验证
            report.after_scores = {}
            report.summary = "已生成优化建议，待应用后重新评测验证效果"

        return report

    @staticmethod
    def _generate_summary(report: OptimizationReport) -> str:
        """生成闭环总结."""
        lines = []
        lines.append(f"优化闭环报告 - {report.model_name}")
        lines.append(f"生成建议数: {len(report.suggestions)}")
        lines.append(f"应用策略数: {len(report.applied_strategies)}")

        if report.improvements:
            lines.append("\n效果对比:")
            for dim, delta in report.improvements.items():
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                lines.append(f"  {dim}: {report.before_scores.get(dim, 0):.1f} → "
                           f"{report.after_scores.get(dim, 0):.1f} ({arrow}{abs(delta):.1f})")

            if report.is_improved:
                lines.append("\n结论: 优化有效，建议保持当前策略")
            else:
                lines.append("\n结论: 优化效果不明显，建议尝试其他策略")
        else:
            lines.append("\n待应用优化策略后重新评测")

        return "\n".join(lines)

    def save_report(self, report: OptimizationReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"optimization_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: OptimizationReport) -> None:
        """打印报告."""
        print(f"\n{'='*60}")
        print(f"优化闭环报告 - {report.model_name}")
        print(f"{'='*60}")

        print(f"\n优化前评分:")
        for dim, score in report.before_scores.items():
            print(f"  {dim}: {score:.1f}")

        print(f"\n优化建议 ({len(report.suggestions)}条):")
        for sug in report.suggestions:
            print(f"  [{sug.priority}][{sug.category}] {sug.description}")
            print(f"    预期: {sug.expected_improvement}")
            print(f"    操作: {sug.action}")

        print(f"\n应用策略 ({len(report.applied_strategies)}个):")
        for strategy_id in report.applied_strategies:
            strategy = BUILTIN_STRATEGIES.get(strategy_id, {})
            print(f"  - {strategy.get('name', strategy_id)}: {strategy.get('description', '')}")

        if report.improvements:
            print(f"\n效果对比:")
            for dim, delta in report.improvements.items():
                before = report.before_scores.get(dim, 0)
                after = report.after_scores.get(dim, 0)
                arrow = "↑" if delta > 0 else ("↓" if delta < 0 else "→")
                print(f"  {dim}: {before:.1f} → {after:.1f} ({arrow}{abs(delta):.1f})")

            status = "✓ 优化有效" if report.is_improved else "✗ 优化不明显"
            print(f"\n结论: {status}")

        print(f"\n{report.summary}")
