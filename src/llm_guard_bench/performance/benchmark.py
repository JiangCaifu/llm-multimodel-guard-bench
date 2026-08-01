"""性能基准测试核心模块.

压测场景矩阵：
    并发度 × 输入长度 × 输出长度 = 3 × 3 × 3 = 27 个场景

关键指标：
    - TTFT: 首 Token 延迟（请求发出到第一个 token 返回）
    - TPOT: Token 间延迟（相邻 token 的时间间隔）
    - 吞吐量: Tokens/s
    - 请求成功率: 非超时/非错误比例
    - E2E 延迟: 端到端完整响应时间
"""
from __future__ import annotations

import asyncio
import json
import os
import statistics
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

from ..adapters.base import BaseModelAdapter, GenerationResult


class InputLength(str, Enum):
    """输入长度级别."""

    SHORT = "128"
    MEDIUM = "512"
    LONG = "2048"


class OutputLength(str, Enum):
    """输出长度级别."""

    SHORT = "64"
    MEDIUM = "256"
    LONG = "1024"


# 预设的测试 prompt 模板

INPUT_TEMPLATES = {
    InputLength.SHORT: "请简要介绍一下人工智能的定义和主要分支。",
    InputLength.MEDIUM: (
        "请详细分析人工智能的三大主要分支：机器学习、自然语言处理和计算机视觉。"
        "对每个分支，说明其核心原理、典型应用场景和当前面临的主要挑战。"
        "同时讨论这三个分支之间如何相互交叉和促进。"
    ),
    InputLength.LONG: (
        "请全面深入地分析人工智能领域的发展现状和未来趋势。"
        "首先，回顾人工智能从1956年达特茅斯会议至今的发展历程，包括几次寒冬和复兴。"
        "然后，详细分析深度学习的三大主流架构——CNN、RNN和Transformer的原理、优缺点和适用场景。"
        "接着，讨论大语言模型（LLM）的技术路线，包括GPT系列、BERT系列和LLaMA系列的核心创新。"
        "此外，分析多模态AI、具身智能和AI Agent等前沿方向的技术挑战。"
        "最后，讨论AI安全、对齐和治理等关键议题，以及中美欧在AI监管上的不同路径。"
    ),
}


@dataclass
class BenchmarkScenario:
    """压测场景."""

    concurrency: int            # 并发请求数
    input_length: InputLength   # 输入长度
    output_length: OutputLength # 输出长度（max_tokens）
    num_requests: int = 10      # 每个场景的总请求数


@dataclass
class RequestMetrics:
    """单次请求指标."""

    request_id: int
    concurrency: int
    input_length: str
    output_length: str
    success: bool = True

    # 延迟指标（毫秒）
    e2e_latency_ms: float = 0.0    # 端到端延迟
    ttft_ms: float = 0.0           # 首 token 延迟（流式时可用）
    tpot_ms: float = 0.0           # token 间平均延迟

    # 吞吐指标
    prompt_tokens: int = 0
    completion_tokens: int = 0
    tokens_per_second: float = 0.0

    # 错误信息
    error: Optional[str] = None


@dataclass
class ScenarioResult:
    """场景压测结果."""

    scenario: BenchmarkScenario
    metrics: List[RequestMetrics] = field(default_factory=list)

    # 汇总指标
    success_rate: float = 0.0
    avg_e2e_latency_ms: float = 0.0
    p50_e2e_ms: float = 0.0
    p95_e2e_ms: float = 0.0
    p99_e2e_ms: float = 0.0
    avg_ttft_ms: float = 0.0
    avg_tpot_ms: float = 0.0
    avg_throughput: float = 0.0  # tokens/s
    avg_completion_tokens: float = 0.0

    def compute_stats(self) -> None:
        """计算汇总统计."""
        successful = [m for m in self.metrics if m.success]
        if not successful:
            self.success_rate = 0.0
            return

        self.success_rate = len(successful) / len(self.metrics)

        e2e_list = [m.e2e_latency_ms for m in successful]
        ttft_list = [m.ttft_ms for m in successful if m.ttft_ms > 0]
        tpot_list = [m.tpot_ms for m in successful if m.tpot_ms > 0]
        tps_list = [m.tokens_per_second for m in successful]
        tokens_list = [m.completion_tokens for m in successful]

        self.avg_e2e_latency_ms = statistics.mean(e2e_list)
        self.p50_e2e_ms = self._percentile(e2e_list, 50)
        self.p95_e2e_ms = self._percentile(e2e_list, 95)
        self.p99_e2e_ms = self._percentile(e2e_list, 99)
        self.avg_ttft_ms = statistics.mean(ttft_list) if ttft_list else 0.0
        self.avg_tpot_ms = statistics.mean(tpot_list) if tpot_list else 0.0
        self.avg_throughput = statistics.mean(tps_list) if tps_list else 0.0
        self.avg_completion_tokens = statistics.mean(tokens_list) if tokens_list else 0.0

    @staticmethod
    def _percentile(data: List[float], p: int) -> float:
        """计算百分位数."""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        k = (len(sorted_data) - 1) * (p / 100.0)
        f = int(k)
        c = f + 1
        if c >= len(sorted_data):
            return sorted_data[f]
        return sorted_data[f] + (k - f) * (sorted_data[c] - sorted_data[f])


@dataclass
class BenchmarkReport:
    """性能基准测试报告."""

    model_name: str
    scenarios: List[ScenarioResult] = field(default_factory=list)

    # 分析结论
    performance_inflection: Optional[str] = None   # 性能拐点
    bottleneck: Optional[str] = None               # 瓶颈归因
    deployment_advice: Optional[str] = None        # 部署建议

    def to_dict(self) -> Dict[str, Any]:
        """转为字典."""
        return {
            "model_name": self.model_name,
            "scenarios": [
                {
                    "concurrency": s.scenario.concurrency,
                    "input_length": s.scenario.input_length.value,
                    "output_length": s.scenario.output_length.value,
                    "num_requests": s.scenario.num_requests,
                    "success_rate": f"{s.success_rate:.1%}",
                    "avg_e2e_ms": round(s.avg_e2e_latency_ms, 1),
                    "p50_e2e_ms": round(s.p50_e2e_ms, 1),
                    "p95_e2e_ms": round(s.p95_e2e_ms, 1),
                    "avg_ttft_ms": round(s.avg_ttft_ms, 1),
                    "avg_tpot_ms": round(s.avg_tpot_ms, 1),
                    "avg_throughput_tps": round(s.avg_throughput, 1),
                    "avg_completion_tokens": round(s.avg_completion_tokens, 1),
                }
                for s in self.scenarios
            ],
            "analysis": {
                "performance_inflection": self.performance_inflection,
                "bottleneck": self.bottleneck,
                "deployment_advice": self.deployment_advice,
            },
        }


class PerformanceBenchmark:
    """性能基准测试执行器."""

    def __init__(
        self,
        adapter: BaseModelAdapter,
        concurrency_levels: Optional[List[int]] = None,
        input_lengths: Optional[List[InputLength]] = None,
        output_lengths: Optional[List[OutputLength]] = None,
        requests_per_scenario: int = 10,
        timeout_per_request: int = 120,
    ) -> None:
        """初始化.

        Args:
            adapter: 模型适配器
            concurrency_levels: 并发度列表
            input_lengths: 输入长度列表
            output_lengths: 输出长度列表
            requests_per_scenario: 每个场景的请求数
            timeout_per_request: 单请求超时（秒）
        """
        self._adapter = adapter
        self._concurrency_levels = concurrency_levels or [1, 5, 10]
        self._input_lengths = input_lengths or [InputLength.SHORT, InputLength.MEDIUM, InputLength.LONG]
        self._output_lengths = output_lengths or [OutputLength.SHORT, OutputLength.MEDIUM, OutputLength.LONG]
        self._requests_per_scenario = requests_per_scenario
        self._timeout = timeout_per_request

    def build_scenarios(self) -> List[BenchmarkScenario]:
        """构建压测场景矩阵."""
        scenarios = []
        for conc in self._concurrency_levels:
            for inp_len in self._input_lengths:
                for out_len in self._output_lengths:
                    scenarios.append(BenchmarkScenario(
                        concurrency=conc,
                        input_length=inp_len,
                        output_length=out_len,
                        num_requests=self._requests_per_scenario,
                    ))
        return scenarios

    def run_scenario(self, scenario: BenchmarkScenario) -> ScenarioResult:
        """执行单个压测场景.

        使用线程池实现并发请求。

        Args:
            scenario: 压测场景

        Returns:
            场景结果
        """
        prompt = INPUT_TEMPLATES[scenario.input_length]
        max_tokens = int(scenario.output_length.value)
        metrics: List[RequestMetrics] = []

        def _single_request(req_id: int) -> RequestMetrics:
            m = RequestMetrics(
                request_id=req_id,
                concurrency=scenario.concurrency,
                input_length=scenario.input_length.value,
                output_length=scenario.output_length.value,
            )

            try:
                start = time.time()
                result = self._adapter.chat(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.0,
                )
                elapsed_ms = (time.time() - start) * 1000

                if result.error:
                    m.success = False
                    m.error = result.error
                    m.e2e_latency_ms = elapsed_ms
                    return m

                m.e2e_latency_ms = elapsed_ms
                m.prompt_tokens = result.prompt_tokens or 0
                m.completion_tokens = result.completion_tokens or 0

                # 计算吞吐量
                if elapsed_ms > 0 and m.completion_tokens > 0:
                    m.tokens_per_second = m.completion_tokens / (elapsed_ms / 1000)

                # 估算 TTFT 和 TPOT
                # 非 streaming 模式下只能估算：TTFT ≈ E2E * 20%, TPOT ≈ (E2E - TTFT) / tokens
                if m.completion_tokens > 0:
                    m.ttft_ms = elapsed_ms * 0.2  # 估算
                    m.tpot_ms = (elapsed_ms - m.ttft_ms) / m.completion_tokens

            except Exception as e:
                m.success = False
                m.error = str(e)
                m.e2e_latency_ms = (time.time() - start) * 1000 if 'start' in dir() else 0

            return m

        # 使用线程池并发
        with ThreadPoolExecutor(max_workers=scenario.concurrency) as executor:
            futures = [
                executor.submit(_single_request, i)
                for i in range(scenario.num_requests)
            ]
            for future in as_completed(futures, timeout=self._timeout):
                try:
                    m = future.result(timeout=self._timeout)
                    metrics.append(m)
                except Exception as e:
                    metrics.append(RequestMetrics(
                        request_id=len(metrics),
                        concurrency=scenario.concurrency,
                        input_length=scenario.input_length.value,
                        output_length=scenario.output_length.value,
                        success=False,
                        error=str(e),
                    ))

        result = ScenarioResult(scenario=scenario, metrics=metrics)
        result.compute_stats()
        return result

    def run_all(self, quick: bool = False) -> BenchmarkReport:
        """运行全部压测场景.

        Args:
            quick: 快速模式，减少场景数和请求数

        Returns:
            性能基准测试报告
        """
        model_name = getattr(self._adapter, '_config', None)
        model_name = model_name.model_name if model_name else "unknown"

        if quick:
            # 快速模式：只跑并发度 1/5，输入 SHORT/MEDIUM，输出 SHORT
            original_conc = self._concurrency_levels
            original_inp = self._input_lengths
            original_out = self._output_lengths
            self._concurrency_levels = [1, 5]
            self._input_lengths = [InputLength.SHORT, InputLength.MEDIUM]
            self._output_lengths = [OutputLength.SHORT]
            self._requests_per_scenario = 3

        scenarios = self.build_scenarios()
        report = BenchmarkReport(model_name=model_name)

        total = len(scenarios)
        for i, scenario in enumerate(scenarios):
            print(f"  [{i+1}/{total}] 并发={scenario.concurrency}, "
                  f"输入={scenario.input_length.value}, "
                  f"输出={scenario.output_length.value}")

            result = self.run_scenario(scenario)
            report.scenarios.append(result)

        if quick:
            self._concurrency_levels = original_conc
            self._input_lengths = original_inp
            self._output_lengths = original_out

        # 性能分析
        self._analyze(report)
        return report

    def _analyze(self, report: BenchmarkReport) -> None:
        """性能拐点分析 + 瓶颈归因 + 部署建议."""
        if not report.scenarios:
            return

        # 1. 找性能拐点：并发增加时 E2E 延迟跳变的点
        # 按相同输入输出、不同并发度分组
        inflection_points = []

        for inp_len in self._input_lengths:
            for out_len in self._output_lengths:
                group = [
                    s for s in report.scenarios
                    if s.scenario.input_length == inp_len
                    and s.scenario.output_length == out_len
                ]
                group.sort(key=lambda s: s.scenario.concurrency)

                for i in range(1, len(group)):
                    prev = group[i - 1]
                    curr = group[i]
                    if prev.avg_e2e_latency_ms > 0:
                        ratio = curr.avg_e2e_latency_ms / prev.avg_e2e_latency_ms
                        if ratio >= 2.0:
                            inflection_points.append(
                                f"并发 {prev.scenario.concurrency}→{curr.scenario.concurrency} 时，"
                                f"E2E 延迟从 {prev.avg_e2e_latency_ms:.0f}ms 跳到 "
                                f"{curr.avg_e2e_latency_ms:.0f}ms（{ratio:.1f}x），"
                                f"输入={inp_len.value}, 输出={out_len.value}"
                            )

        if inflection_points:
            report.performance_inflection = inflection_points[0]
        else:
            # 找延迟增长最大的
            max_ratio = 0.0
            max_desc = ""
            for inp_len in self._input_lengths:
                for out_len in self._output_lengths:
                    group = [
                        s for s in report.scenarios
                        if s.scenario.input_length == inp_len
                        and s.scenario.output_length == out_len
                    ]
                    group.sort(key=lambda s: s.scenario.concurrency)
                    if len(group) >= 2:
                        first = group[0]
                        last = group[-1]
                        if first.avg_e2e_latency_ms > 0:
                            ratio = last.avg_e2e_latency_ms / first.avg_e2e_latency_ms
                            if ratio > max_ratio:
                                max_ratio = ratio
                                max_desc = (
                                    f"并发 {first.scenario.concurrency}→{last.scenario.concurrency} 时，"
                                    f"E2E 延迟从 {first.avg_e2e_latency_ms:.0f}ms 增长到 "
                                    f"{last.avg_e2e_latency_ms:.0f}ms（{ratio:.1f}x），"
                                    f"输入={inp_len.value}, 输出={out_len.value}"
                                )
            report.performance_inflection = max_desc if max_desc else "无明显拐点"

        # 2. 瓶颈归因
        # 检查成功率是否随并发下降
        low_success = [
            s for s in report.scenarios
            if s.success_rate < 0.9
        ]
        if low_success:
            report.bottleneck = (
                f"高并发时请求失败率升高（{len(low_success)} 个场景成功率<90%），"
                f"可能原因：API 速率限制/服务端排队/超时"
            )
        else:
            # 检查是否延迟随输入长度线性增长（服务端处理瓶颈）
            long_input = [
                s for s in report.scenarios
                if s.scenario.input_length == InputLength.LONG
                and s.scenario.concurrency == 1
            ]
            short_input = [
                s for s in report.scenarios
                if s.scenario.input_length == InputLength.SHORT
                and s.scenario.concurrency == 1
            ]
            if long_input and short_input:
                avg_long = statistics.mean([s.avg_e2e_latency_ms for s in long_input])
                avg_short = statistics.mean([s.avg_e2e_latency_ms for s in short_input])
                if avg_short > 0 and avg_long / avg_short > 2:
                    report.bottleneck = "长输入处理延迟显著高于短输入，瓶颈可能在 prompt 处理/KV Cache"
                else:
                    report.bottleneck = "延迟增长主要由并发排队导致，服务端处理能力是瓶颈"

        # 3. 部署建议
        # 找到 P95 < 3s 的最大并发度
        max_safe_concurrency = 1
        for conc in sorted(self._concurrency_levels, reverse=True):
            scenarios_at_conc = [
                s for s in report.scenarios
                if s.scenario.concurrency == conc
                and s.scenario.input_length == InputLength.SHORT
                and s.scenario.output_length == OutputLength.SHORT
            ]
            if scenarios_at_conc and scenarios_at_conc[0].p95_e2e_ms < 3000:
                max_safe_concurrency = conc
                break

        report.deployment_advice = (
            f"建议单实例并发不超过 {max_safe_concurrency}，"
            f"超过需水平扩展。"
            f"P95 延迟应控制在 3s 以内（当前 {max_safe_concurrency} 并发时 "
            f"P95={report.scenarios[0].p95_e2e_ms:.0f}ms）。"
        )

    def save_report(self, report: BenchmarkReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"performance_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: BenchmarkReport) -> None:
        """打印性能报告."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            console.print(f"\n[bold]性能基准测试报告 - {report.model_name}[/bold]")

            # 场景结果表
            table = Table(title="场景压测结果")
            table.add_column("并发", justify="right", style="cyan")
            table.add_column("输入", justify="right")
            table.add_column("输出", justify="right")
            table.add_column("成功率", justify="right")
            table.add_column("E2E均值(ms)", justify="right")
            table.add_column("P95(ms)", justify="right")
            table.add_column("TTFT(ms)", justify="right")
            table.add_column("吞吐(t/s)", justify="right")

            for s in report.scenarios:
                e2e_color = "green" if s.avg_e2e_latency_ms < 2000 else ("yellow" if s.avg_e2e_latency_ms < 5000 else "red")
                table.add_row(
                    str(s.scenario.concurrency),
                    s.scenario.input_length.value,
                    s.scenario.output_length.value,
                    f"{s.success_rate:.0%}",
                    f"[{e2e_color}]{s.avg_e2e_latency_ms:.0f}[/{e2e_color}]",
                    f"{s.p95_e2e_ms:.0f}",
                    f"{s.avg_ttft_ms:.0f}",
                    f"{s.avg_throughput:.1f}",
                )

            console.print(table)

            # 分析结论
            if report.performance_inflection:
                console.print(f"\n[yellow]性能拐点[/yellow]: {report.performance_inflection}")
            if report.bottleneck:
                console.print(f"[red]瓶颈归因[/red]: {report.bottleneck}")
            if report.deployment_advice:
                console.print(f"[green]部署建议[/green]: {report.deployment_advice}")

        except ImportError:
            print(f"\n性能基准测试报告 - {report.model_name}")
            for s in report.scenarios:
                print(f"  并发={s.scenario.concurrency} 输入={s.scenario.input_length.value} "
                      f"输出={s.scenario.output_length.value} "
                      f"E2E={s.avg_e2e_latency_ms:.0f}ms P95={s.p95_e2e_ms:.0f}ms "
                      f"吞吐={s.avg_throughput:.1f}t/s")
            if report.performance_inflection:
                print(f"  拐点: {report.performance_inflection}")
            if report.bottleneck:
                print(f"  瓶颈: {report.bottleneck}")
            if report.deployment_advice:
                print(f"  建议: {report.deployment_advice}")
