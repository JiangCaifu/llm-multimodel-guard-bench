"""Agent评测统一入口.

编排工具调用 / 多步任务 / Code Agent 评测流程。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter
from .code_agent import CodeAgentEvaluator
from .multi_step import MultiStepEvaluator, MultiStepTestCase
from .tool_call import ToolCallEvaluator, ToolCallTestCase


@dataclass
class AgentReport:
    """Agent 评测综合报告."""

    model_name: str
    # 工具调用
    tool_selection_accuracy: float = 0.0
    params_fill_accuracy: float = 0.0
    tool_overall_accuracy: float = 0.0
    # 多步任务
    planning_score: float = 0.0
    execution_accuracy: float = 0.0
    completion_rate: float = 0.0
    # Code Agent
    code_gen_pass_rate: float = 0.0
    debug_fix_rate: float = 0.0

    tool_report: Optional[Dict[str, Any]] = None
    multi_step_report: Optional[Dict[str, Any]] = None
    code_agent_report: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "tool_call": {
                "tool_selection_accuracy": f"{self.tool_selection_accuracy:.1%}",
                "params_fill_accuracy": f"{self.params_fill_accuracy:.1%}",
                "overall_accuracy": f"{self.tool_overall_accuracy:.1%}",
                "detail": self.tool_report,
            },
            "multi_step": {
                "planning_score": f"{self.planning_score:.1%}",
                "execution_accuracy": f"{self.execution_accuracy:.1%}",
                "completion_rate": f"{self.completion_rate:.1%}",
                "detail": self.multi_step_report,
            },
            "code_agent": {
                "code_gen_pass_rate": f"{self.code_gen_pass_rate:.1%}",
                "debug_fix_rate": f"{self.debug_fix_rate:.1%}",
                "detail": self.code_agent_report,
            },
        }


class AgentRunner:
    """Agent 评测执行器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter
        self._tool_evaluator = ToolCallEvaluator(adapter)
        self._multi_step_evaluator = MultiStepEvaluator(adapter)
        self._code_evaluator = CodeAgentEvaluator(adapter)

    def run_tool_call(
        self,
        cases: Optional[List[ToolCallTestCase]] = None,
    ) -> Dict[str, Any]:
        """运行工具调用评测."""
        report = self._tool_evaluator.evaluate_batch(cases)
        return report.to_dict()

    def run_multi_step(
        self,
        cases: Optional[List[MultiStepTestCase]] = None,
    ) -> Dict[str, Any]:
        """运行多步任务评测."""
        report = self._multi_step_evaluator.evaluate_batch(cases)
        return report.to_dict()

    def run_code_agent(self) -> Dict[str, Any]:
        """运行 Code Agent 评测."""
        report = self._code_evaluator.evaluate_batch()
        return report.to_dict()

    def run_full(
        self,
        tool_cases: Optional[List[ToolCallTestCase]] = None,
        multi_step_cases: Optional[List[MultiStepTestCase]] = None,
    ) -> AgentReport:
        """运行完整 Agent 评测."""
        model_name = (
            getattr(self._adapter, '_config', None)
            and self._adapter._config.model_name or "unknown"
        )
        report = AgentReport(model_name=model_name)

        # 工具调用
        tool_r = self.run_tool_call(tool_cases)
        report.tool_report = tool_r
        acc_str = tool_r.get("overall_accuracy", "0.0%")
        report.tool_overall_accuracy = float(acc_str.rstrip("%")) / 100 if "%" in acc_str else 0.0
        sel_str = tool_r.get("tool_selection_accuracy", "0.0%")
        report.tool_selection_accuracy = float(sel_str.rstrip("%")) / 100 if "%" in sel_str else 0.0
        param_str = tool_r.get("params_fill_accuracy", "0.0%")
        report.params_fill_accuracy = float(param_str.rstrip("%")) / 100 if "%" in param_str else 0.0

        # 多步任务
        ms_r = self.run_multi_step(multi_step_cases)
        report.multi_step_report = ms_r
        plan_str = ms_r.get("avg_planning_score", "0.0%")
        report.planning_score = float(plan_str.rstrip("%")) / 100 if "%" in plan_str else 0.0
        exec_str = ms_r.get("avg_execution_accuracy", "0.0%")
        report.execution_accuracy = float(exec_str.rstrip("%")) / 100 if "%" in exec_str else 0.0
        comp_str = ms_r.get("avg_completion_rate", "0.0%")
        report.completion_rate = float(comp_str.rstrip("%")) / 100 if "%" in comp_str else 0.0

        # Code Agent
        code_r = self.run_code_agent()
        report.code_agent_report = code_r
        code_gen = code_r.get("code_generation", {}).get("pass_rate", "0.0%")
        report.code_gen_pass_rate = float(code_gen.rstrip("%")) / 100 if "%" in code_gen else 0.0
        debug_fix = code_r.get("debug_fix", {}).get("fix_rate", "0.0%")
        report.debug_fix_rate = float(debug_fix.rstrip("%")) / 100 if "%" in debug_fix else 0.0

        return report

    def save_report(self, report: AgentReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"agent_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: AgentReport) -> None:
        """打印报告."""
        try:
            from rich.console import Console
            from rich.panel import Panel

            console = Console()
            console.print(Panel(
                f"[bold]工具调用[/bold]\n"
                f"  选择准确率: {report.tool_selection_accuracy:.1%}\n"
                f"  参数填充率: {report.params_fill_accuracy:.1%}\n"
                f"  综合准确率: {report.tool_overall_accuracy:.1%}\n\n"
                f"[bold]多步任务[/bold]\n"
                f"  规划评分: {report.planning_score:.1%}\n"
                f"  执行正确率: {report.execution_accuracy:.1%}\n"
                f"  端到端完成率: {report.completion_rate:.1%}\n\n"
                f"[bold]Code Agent[/bold]\n"
                f"  代码生成通过率: {report.code_gen_pass_rate:.1%}\n"
                f"  调试修复率: {report.debug_fix_rate:.1%}",
                title=f"Agent评测 - {report.model_name}",
                expand=False,
            ))
        except ImportError:
            print(f"\nAgent评测报告 - {report.model_name}")
            print(f"  工具调用: 选择={report.tool_selection_accuracy:.0%} 参数={report.params_fill_accuracy:.0%} 综合={report.tool_overall_accuracy:.0%}")
            print(f"  多步任务: 规划={report.planning_score:.0%} 执行={report.execution_accuracy:.0%} 完成={report.completion_rate:.0%}")
            print(f"  Code Agent: 生成={report.code_gen_pass_rate:.0%} 修复={report.debug_fix_rate:.0%}")
