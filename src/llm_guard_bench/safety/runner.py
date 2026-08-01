"""安全评测执行器 - 编排攻击和评判流程.

对应 Phase 2 核心流程：
    - 加载攻击模板
    - 对目标模型发起攻击
    - 使用混合评判器评判结果
    - 生成安全评测报告
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .attacks import AttackCategory, AttackTemplate, AttackTemplateLibrary, RiskTarget
from .judges import HybridJudge, JudgeResult, RiskLevel, SafetyEvalSample
from .level2 import MultiTurnAttacker, SemanticMutator


@dataclass
class SafetyAttackResult:
    """单次攻击结果."""

    attack_id: str
    attack_name: str
    attack_category: str
    risk_target: str
    attack_prompt: str
    model_response: str
    risk_level: RiskLevel
    risk_categories: List[str]
    confidence: float
    reason: str
    judge_method: str
    latency_ms: float = 0.0
    error: Optional[str] = None


@dataclass
class SafetyReport:
    """安全评测报告."""

    model_name: str
    total_attacks: int = 0
    p0_count: int = 0
    p1_count: int = 0
    p2_count: int = 0
    safe_count: int = 0
    attack_success_rate: float = 0.0  # 模型被攻破率
    defense_rate: float = 0.0  # 模型防御率
    results: List[SafetyAttackResult] = field(default_factory=list)
    risk_target_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)
    category_stats: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """转为字典."""
        return {
            "model_name": self.model_name,
            "total_attacks": self.total_attacks,
            "p0_count": self.p0_count,
            "p1_count": self.p1_count,
            "p2_count": self.p2_count,
            "safe_count": self.safe_count,
            "attack_success_rate": f"{self.attack_success_rate:.1%}",
            "defense_rate": f"{self.defense_rate:.1%}",
            "risk_target_stats": self.risk_target_stats,
            "category_stats": self.category_stats,
            "results": [
                {
                    "attack_id": r.attack_id,
                    "attack_name": r.attack_name,
                    "attack_category": r.attack_category,
                    "risk_target": r.risk_target,
                    "attack_prompt": r.attack_prompt[:200],
                    "model_response": r.model_response[:200],
                    "risk_level": r.risk_level.value,
                    "risk_categories": r.risk_categories,
                    "confidence": r.confidence,
                    "reason": r.reason,
                    "judge_method": r.judge_method,
                    "latency_ms": f"{r.latency_ms:.0f}",
                    "error": r.error,
                }
                for r in self.results
            ],
        }


class SafetyRunner:
    """安全评测执行器."""

    def __init__(
        self,
        model_adapter: Any,
        judge_adapter: Any = None,
        use_semantic_mutation: bool = False,
        mutation_count: int = 5,
        max_samples: Optional[int] = None,
    ) -> None:
        """初始化.

        Args:
            model_adapter: 被评测的模型适配器
            judge_adapter: LLM评判用的模型适配器（可选）
            use_semantic_mutation: 是否启用语义变异
            mutation_count: 每个模板生成的变异数量
            max_samples: 最大攻击样本数（开发阶段限制）
        """
        self._adapter = model_adapter
        self._judge = HybridJudge(llm_adapter=judge_adapter)
        self._template_lib = AttackTemplateLibrary()
        self._mutator = SemanticMutator(adapter=judge_adapter if use_semantic_mutation else None)
        self._multi_turn_attacker = MultiTurnAttacker()
        self._use_mutation = use_semantic_mutation
        self._mutation_count = mutation_count
        self._max_samples = max_samples

    def run_level1(self) -> SafetyReport:
        """运行 Level 1 攻击（经典越狱 + 角色扮演 + 编码绕过）."""
        templates = self._template_lib.list_level1()
        if self._max_samples:
            templates = templates[:self._max_samples]

        report = SafetyReport(model_name=getattr(self._adapter, '_config', None) and self._adapter._config.model_name or "unknown")

        for tpl in templates:
            result = self._attack_single(tpl)
            report.results.append(result)

            # 语义变异
            if self._use_mutation:
                mutations = self._mutator.mutate(tpl, self._mutation_count)
                for mut_tpl in mutations[:2]:  # 限制变异数量
                    mut_result = self._attack_single(mut_tpl)
                    report.results.append(mut_result)

        self._compute_stats(report)
        return report

    def run_level2_multi_turn(self) -> SafetyReport:
        """运行 Level 2 多轮诱导攻击."""
        conversations = self._multi_turn_attacker.list_all()
        report = SafetyReport(model_name=getattr(self._adapter, '_config', None) and self._adapter._config.model_name or "unknown")

        for conv in conversations:
            result = self._attack_multi_turn(conv)
            if result:
                report.results.append(result)

        self._compute_stats(report)
        return report

    def run_full(self) -> SafetyReport:
        """运行完整安全评测（Level 1 + Level 2）."""
        report = SafetyReport(model_name=getattr(self._adapter, '_config', None) and self._adapter._config.model_name or "unknown")

        # Level 1
        templates = self._template_lib.list_level1()
        if self._max_samples:
            templates = templates[:self._max_samples]

        for tpl in templates:
            result = self._attack_single(tpl)
            report.results.append(result)

            if self._use_mutation:
                mutations = self._mutator.mutate(tpl, min(self._mutation_count, 2))
                for mut_tpl in mutations:
                    mut_result = self._attack_single(mut_tpl)
                    report.results.append(mut_result)

        # Level 2: 多轮诱导
        for conv in self._multi_turn_attacker.list_all():
            result = self._attack_multi_turn(conv)
            if result:
                report.results.append(result)

        self._compute_stats(report)
        return report

    def _attack_single(self, template: AttackTemplate) -> SafetyAttackResult:
        """执行单次攻击."""
        start_time = time.time()

        try:
            gen_result = self._adapter.chat(
                messages=[{"role": "user", "content": template.prompt}],
                max_tokens=512,
                temperature=0.0,
            )
            latency_ms = (time.time() - start_time) * 1000
            model_response = gen_result.text

            if gen_result.error:
                return SafetyAttackResult(
                    attack_id=template.id,
                    attack_name=template.name,
                    attack_category=template.category.value,
                    risk_target=template.risk_target.value,
                    attack_prompt=template.prompt,
                    model_response="",
                    risk_level=RiskLevel.SAFE,
                    risk_categories=[],
                    confidence=0.0,
                    reason=f"模型调用失败: {gen_result.error}",
                    judge_method="error",
                    latency_ms=latency_ms,
                    error=gen_result.error,
                )

        except Exception as e:
            return SafetyAttackResult(
                attack_id=template.id,
                attack_name=template.name,
                attack_category=template.category.value,
                risk_target=template.risk_target.value,
                attack_prompt=template.prompt,
                model_response="",
                risk_level=RiskLevel.SAFE,
                risk_categories=[],
                confidence=0.0,
                reason=f"攻击执行异常: {e}",
                judge_method="error",
                error=str(e),
            )

        # 评判
        judge_result = self._judge.judge(template.prompt, model_response)

        return SafetyAttackResult(
            attack_id=template.id,
            attack_name=template.name,
            attack_category=template.category.value,
            risk_target=template.risk_target.value,
            attack_prompt=template.prompt,
            model_response=model_response,
            risk_level=judge_result.risk_level,
            risk_categories=judge_result.risk_categories,
            confidence=judge_result.confidence,
            reason=judge_result.reason,
            judge_method=judge_result.judge_method,
            latency_ms=latency_ms,
        )

    def _attack_multi_turn(self, conv: Any) -> Optional[SafetyAttackResult]:
        """执行多轮诱导攻击."""
        start_time = time.time()
        messages: List[Dict[str, str]] = []
        last_response = ""
        last_user_prompt = ""

        try:
            for turn in conv.turns:
                if turn["role"] == "user":
                    messages.append({"role": "user", "content": turn["content"]})
                    last_user_prompt = turn["content"]

                    gen_result = self._adapter.chat(
                        messages=messages,
                        max_tokens=512,
                        temperature=0.0,
                    )

                    if gen_result.error:
                        last_response = f"[ERROR: {gen_result.error}]"
                    else:
                        last_response = gen_result.text

                    messages.append({"role": "assistant", "content": last_response})
                elif turn["role"] == "assistant":
                    # 预设的助手回复（用于引导对话）
                    messages.append({"role": "assistant", "content": turn["content"]})

        except Exception as e:
            last_response = f"[EXCEPTION: {e}]"

        latency_ms = (time.time() - start_time) * 1000

        # 评判最后一轮
        judge_result = self._judge.judge(last_user_prompt, last_response)

        return SafetyAttackResult(
            attack_id=conv.id,
            attack_name=conv.description,
            attack_category="multi_turn",
            risk_target=conv.target_risk.value,
            attack_prompt=last_user_prompt,
            model_response=last_response,
            risk_level=judge_result.risk_level,
            risk_categories=judge_result.risk_categories,
            confidence=judge_result.confidence,
            reason=judge_result.reason,
            judge_method=judge_result.judge_method,
            latency_ms=latency_ms,
        )

    def _compute_stats(self, report: SafetyReport) -> None:
        """计算统计数据."""
        report.total_attacks = len(report.results)
        report.p0_count = sum(1 for r in report.results if r.risk_level == RiskLevel.P0)
        report.p1_count = sum(1 for r in report.results if r.risk_level == RiskLevel.P1)
        report.p2_count = sum(1 for r in report.results if r.risk_level == RiskLevel.P2)
        report.safe_count = sum(1 for r in report.results if r.risk_level == RiskLevel.SAFE)

        if report.total_attacks > 0:
            report.attack_success_rate = (report.p0_count + report.p1_count) / report.total_attacks
            report.defense_rate = report.safe_count / report.total_attacks

        # 按风险目标统计
        target_stats: Dict[str, Dict[str, int]] = {}
        for r in report.results:
            target = r.risk_target
            if target not in target_stats:
                target_stats[target] = {"total": 0, "p0": 0, "p1": 0, "safe": 0}
            target_stats[target]["total"] += 1
            if r.risk_level == RiskLevel.P0:
                target_stats[target]["p0"] += 1
            elif r.risk_level == RiskLevel.P1:
                target_stats[target]["p1"] += 1
            elif r.risk_level == RiskLevel.SAFE:
                target_stats[target]["safe"] += 1
        report.risk_target_stats = target_stats

        # 按攻击类型统计
        cat_stats: Dict[str, Dict[str, int]] = {}
        for r in report.results:
            cat = r.attack_category
            if cat not in cat_stats:
                cat_stats[cat] = {"total": 0, "breached": 0, "safe": 0}
            cat_stats[cat]["total"] += 1
            if r.risk_level in (RiskLevel.P0, RiskLevel.P1):
                cat_stats[cat]["breached"] += 1
            elif r.risk_level == RiskLevel.SAFE:
                cat_stats[cat]["safe"] += 1
        report.category_stats = cat_stats

    def save_report(self, report: SafetyReport, output_dir: str) -> str:
        """保存评测报告."""
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, f"safety_{report.model_name}.json")
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return output_path

    def print_report(self, report: SafetyReport) -> None:
        """打印评测报告（Rich格式）."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()

            # 总览
            console.print(f"\n[bold]安全评测报告 - {report.model_name}[/bold]")
            console.print(f"  总攻击数: {report.total_attacks}")
            console.print(f"  [red]P0 (阻断): {report.p0_count}[/red]")
            console.print(f"  [yellow]P1 (复核): {report.p1_count}[/yellow]")
            console.print(f"  [blue]P2 (记录): {report.p2_count}[/blue]")
            console.print(f"  [green]安全: {report.safe_count}[/green]")
            console.print(f"  攻破率: [red]{report.attack_success_rate:.1%}[/red]")
            console.print(f"  防御率: [green]{report.defense_rate:.1%}[/green]")

            # 按攻击类型
            if report.category_stats:
                table = Table(title="按攻击类型统计")
                table.add_column("攻击类型", style="cyan")
                table.add_column("总数", justify="right")
                table.add_column("攻破", justify="right", style="red")
                table.add_column("安全", justify="right", style="green")
                table.add_column("攻破率", justify="right")

                for cat, stats in report.category_stats.items():
                    breach_rate = stats["breached"] / stats["total"] if stats["total"] > 0 else 0
                    table.add_row(cat, str(stats["total"]), str(stats["breached"]), str(stats["safe"]), f"{breach_rate:.1%}")

                console.print(table)

            # 按风险目标
            if report.risk_target_stats:
                table = Table(title="按风险目标统计")
                table.add_column("风险目标", style="cyan")
                table.add_column("总数", justify="right")
                table.add_column("P0", justify="right", style="red")
                table.add_column("P1", justify="right", style="yellow")
                table.add_column("安全", justify="right", style="green")

                for target, stats in report.risk_target_stats.items():
                    table.add_row(target, str(stats["total"]), str(stats["p0"]), str(stats["p1"]), str(stats["safe"]))

                console.print(table)

            # P0 详情
            p0_results = [r for r in report.results if r.risk_level == RiskLevel.P0]
            if p0_results:
                console.print("\n[bold red]⚠ P0 级安全风险详情[/bold red]")
                for r in p0_results[:5]:  # 只显示前5条
                    console.print(f"  [{r.attack_id}] {r.attack_name}")
                    console.print(f"    回复摘要: {r.model_response[:100]}...")
                    console.print(f"    评判理由: {r.reason}")

        except ImportError:
            # 没有 rich 时的简单输出
            print(f"\n安全评测报告 - {report.model_name}")
            print(f"  总攻击: {report.total_attacks}, P0: {report.p0_count}, P1: {report.p1_count}, 安全: {report.safe_count}")
            print(f"  攻破率: {report.attack_success_rate:.1%}, 防御率: {report.defense_rate:.1%}")
