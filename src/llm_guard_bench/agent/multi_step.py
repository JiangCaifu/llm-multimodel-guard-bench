"""多步任务完成率评测.

评测内容：
    - 任务规划能力：能否将复杂任务拆解为子步骤
    - 执行正确率：每一步是否执行正确
    - 端到端完成率：最终是否完成了整个任务
    - ReAct 模式评测：推理-行动交替循环
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


@dataclass
class TaskStep:
    """任务步骤."""

    step_id: int
    description: str          # 步骤描述
    expected_action: str      # 期望的行动
    is_critical: bool = True  # 是否关键步骤（关键步骤失败则整个任务失败）


@dataclass
class MultiStepTestCase:
    """多步任务评测用例."""

    test_id: str
    task_description: str            # 完整任务描述
    steps: List[TaskStep] = field(default_factory=list)  # 期望步骤
    max_rounds: int = 10             # 最大对话轮数
    category: str = "general"        # 类别


@dataclass
class StepResult:
    """步骤执行结果."""

    step_id: int
    model_response: str = ""
    action_extracted: str = ""       # 模型采取的行动
    is_correct: bool = False
    is_relevant: bool = False        # 是否与任务相关


@dataclass
class MultiStepResult:
    """多步任务评测结果."""

    test_id: str
    task_description: str
    conversation: List[Dict[str, str]] = field(default_factory=list)
    step_results: List[StepResult] = field(default_factory=list)
    planning_score: float = 0.0      # 规划评分
    execution_accuracy: float = 0.0  # 执行正确率
    completion_rate: float = 0.0     # 端到端完成率
    rounds_used: int = 0


@dataclass
class MultiStepReport:
    """多步任务评测报告."""

    model_name: str
    total_tasks: int = 0
    avg_planning_score: float = 0.0
    avg_execution_accuracy: float = 0.0
    avg_completion_rate: float = 0.0
    results: List[MultiStepResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_tasks": self.total_tasks,
            "avg_planning_score": f"{self.avg_planning_score:.1%}",
            "avg_execution_accuracy": f"{self.avg_execution_accuracy:.1%}",
            "avg_completion_rate": f"{self.avg_completion_rate:.1%}",
            "results": [
                {
                    "test_id": r.test_id,
                    "task": r.task_description[:60],
                    "planning_score": f"{r.planning_score:.1%}",
                    "execution_accuracy": f"{r.execution_accuracy:.1%}",
                    "completion_rate": f"{r.completion_rate:.1%}",
                    "rounds_used": r.rounds_used,
                }
                for r in self.results
            ],
        }


# ========== 内置多步任务用例 ==========

BUILTIN_MULTI_STEP_CASES = [
    MultiStepTestCase(
        test_id="ms_001",
        task_description="帮我规划一次北京三日游，需要包含交通、住宿和景点安排，并估算总预算",
        steps=[
            TaskStep(step_id=1, description="搜索北京的著名景点", expected_action="搜索或列出北京景点"),
            TaskStep(step_id=2, description="规划三天的行程安排", expected_action="按天安排景点顺序"),
            TaskStep(step_id=3, description="推荐住宿区域", expected_action="推荐住宿地点和理由"),
            TaskStep(step_id=4, description="说明交通方式", expected_action="说明各景点间交通方式"),
            TaskStep(step_id=5, description="估算总预算", expected_action="给出预算明细和总计"),
        ],
        category="travel",
    ),
    MultiStepTestCase(
        test_id="ms_002",
        task_description="帮我分析一下购买新能源汽车还是燃油车更划算，需要对比5年总花费",
        steps=[
            TaskStep(step_id=1, description="列出新能源车优缺点", expected_action="分析新能源车优劣"),
            TaskStep(step_id=2, description="列出燃油车优缺点", expected_action="分析燃油车优劣"),
            TaskStep(step_id=3, description="计算新能源车5年花费", expected_action="估算新能源车5年总成本"),
            TaskStep(step_id=4, description="计算燃油车5年花费", expected_action="估算燃油车5年总成本"),
            TaskStep(step_id=5, description="给出对比结论", expected_action="对比总结并给出建议"),
        ],
        category="analysis",
    ),
    MultiStepTestCase(
        test_id="ms_003",
        task_description="帮我写一个Python爬虫程序，抓取某网站标题，并处理异常情况",
        steps=[
            TaskStep(step_id=1, description="理解需求并设计方案", expected_action="描述爬虫方案"),
            TaskStep(step_id=2, description="编写主要代码", expected_action="给出Python代码"),
            TaskStep(step_id=3, description="添加异常处理", expected_action="加入try-except等异常处理"),
            TaskStep(step_id=4, description="说明使用方法", expected_action="给出运行说明"),
        ],
        category="coding",
    ),
    MultiStepTestCase(
        test_id="ms_004",
        task_description="帮我准备一份公司季度汇报PPT大纲，包括业绩数据、问题分析和下季度计划",
        steps=[
            TaskStep(step_id=1, description="设计PPT整体框架", expected_action="列出PPT章节结构"),
            TaskStep(step_id=2, description="填充业绩数据部分", expected_action="列出关键业绩指标"),
            TaskStep(step_id=3, description="填充问题分析部分", expected_action="分析存在的问题"),
            TaskStep(step_id=4, description="填充下季度计划", expected_action="给出下季度目标计划"),
            TaskStep(step_id=5, description="添加总结页", expected_action="给出总结和行动建议"),
        ],
        category="business",
    ),
]


class MultiStepEvaluator:
    """多步任务评测器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter

    def evaluate_single(self, test_case: MultiStepTestCase) -> MultiStepResult:
        """评测单个多步任务.

        使用 ReAct 模式：让模型逐步思考和行动。
        """
        result = MultiStepResult(
            test_id=test_case.test_id,
            task_description=test_case.task_description,
        )

        # 构建初始 prompt（ReAct 风格）
        messages = [
            {
                "role": "system",
                "content": (
                    "你是一个任务执行助手。请按照以下格式逐步完成任务：\n"
                    "思考：分析当前情况，决定下一步\n"
                    "行动：执行具体操作\n"
                    "观察：确认行动结果\n\n"
                    "请开始执行任务，每一步都要明确写出你的思考和行动。"
                ),
            },
            {
                "role": "user",
                "content": f"任务：{test_case.task_description}\n\n请逐步完成这个任务。",
            },
        ]

        for round_idx in range(test_case.max_rounds):
            response = self._adapter.chat(
                messages=messages,
                max_tokens=1024,
                temperature=0.0,
            )

            if not response.success:
                break

            reply = response.text
            messages.append({"role": "assistant", "content": reply})
            result.conversation.append({"role": "assistant", "content": reply[:200]})

            # 检查是否已完成
            if self._is_task_complete(reply, test_case):
                result.rounds_used = round_idx + 1
                break

            # 追问下一步
            if round_idx < test_case.max_rounds - 1:
                follow_up = "请继续完成任务的下一步。"
                messages.append({"role": "user", "content": follow_up})
                result.conversation.append({"role": "user", "content": follow_up})
                result.rounds_used = round_idx + 1

        # 评估步骤完成情况
        all_responses = " ".join(
            msg["content"] for msg in messages if msg["role"] == "assistant"
        )

        for step in test_case.steps:
            step_result = self._evaluate_step(step, all_responses)
            result.step_results.append(step_result)

        # 计算指标
        if result.step_results:
            # 规划评分：步骤覆盖比例
            relevant_steps = [s for s in result.step_results if s.is_relevant]
            result.planning_score = len(relevant_steps) / len(test_case.steps)

            # 执行正确率：正确步骤占比
            correct_steps = [s for s in result.step_results if s.is_correct]
            result.execution_accuracy = len(correct_steps) / len(test_case.steps)

            # 端到端完成率：所有关键步骤都完成
            critical_steps = [
                s for s, ts in zip(result.step_results, test_case.steps)
                if ts.is_critical
            ]
            if critical_steps:
                result.completion_rate = (
                    sum(1 for s in critical_steps if s.is_correct) / len(critical_steps)
                )

        return result

    @staticmethod
    def _is_task_complete(response: str, test_case: MultiStepTestCase) -> bool:
        """判断任务是否完成."""
        complete_keywords = [
            "以上就是", "以上就是我的", "任务完成", "已完成", "总结如下",
            "总结一下", "综上所述", "希望这些信息", "如有其他问题",
        ]
        return any(kw in response for kw in complete_keywords)

    @staticmethod
    def _extract_keywords(text: str) -> List[str]:
        """从期望行动中提取关键词.

        策略：按标点和连词分割，再对过长的片段按2-4字滑窗提取短关键词。
        """
        import re as _re
        # 去除停用词
        stopwords = {"的", "了", "和", "与", "及", "或", "等", "中", "在", "是", "有", "对", "并"}
        # 按标点分割
        parts = _re.split(r'[，、；或和及与/]', text)
        keywords = []
        for p in parts:
            p = p.strip()
            if len(p) <= 1:
                continue
            # 短片段直接作为关键词
            if len(p) <= 4:
                if p not in stopwords:
                    keywords.append(p)
            else:
                # 长片段：用2-4字滑窗提取
                for wlen in (4, 3, 2):
                    for i in range(len(p) - wlen + 1):
                        w = p[i:i + wlen]
                        if not any(s in w for s in stopwords) and w not in keywords:
                            keywords.append(w)
        if not keywords:
            keywords = [text[:4]]
        return keywords

    @staticmethod
    def _evaluate_step(step: TaskStep, all_responses: str) -> StepResult:
        """评估单个步骤完成情况."""
        step_result = StepResult(step_id=step.step_id)

        # 提取关键词并检查是否被提及
        keywords = MultiStepEvaluator._extract_keywords(step.expected_action)
        matched = [kw for kw in keywords if kw in all_responses]
        match_rate = len(matched) / len(keywords) if keywords else 0.0

        # 相关性判断：至少20%的关键词被提及
        step_result.is_relevant = match_rate >= 0.2

        # 正确性判断：至少40%的关键词被提及
        step_result.is_correct = match_rate >= 0.4

        # 提取模型行动
        for kw in matched:
            for line in all_responses.split("\n"):
                if kw in line:
                    step_result.action_extracted = line.strip()[:100]
                    break
            if step_result.action_extracted:
                break

        return step_result

    def evaluate_batch(
        self,
        cases: Optional[List[MultiStepTestCase]] = None,
    ) -> MultiStepReport:
        """批量评测多步任务."""
        cases = cases or BUILTIN_MULTI_STEP_CASES
        model_name = (
            getattr(self._adapter, '_config', None)
            and self._adapter._config.model_name or "unknown"
        )
        report = MultiStepReport(model_name=model_name)

        for case in cases:
            r = self.evaluate_single(case)
            report.results.append(r)

        report.total_tasks = len(report.results)
        if report.total_tasks > 0:
            report.avg_planning_score = sum(r.planning_score for r in report.results) / report.total_tasks
            report.avg_execution_accuracy = sum(r.execution_accuracy for r in report.results) / report.total_tasks
            report.avg_completion_rate = sum(r.completion_rate for r in report.results) / report.total_tasks

        return report

    def save_report(self, report: MultiStepReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"multi_step_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: MultiStepReport) -> None:
        """打印报告."""
        print(f"\n多步任务评测报告 - {report.model_name}")
        print(f"  总任务数: {report.total_tasks}")
        print(f"  平均规划评分: {report.avg_planning_score:.1%}")
        print(f"  平均执行正确率: {report.avg_execution_accuracy:.1%}")
        print(f"  平均端到端完成率: {report.avg_completion_rate:.1%}")

        for r in report.results:
            print(f"\n  任务 {r.test_id}: {r.task_description[:40]}...")
            print(f"    规划={r.planning_score:.0%} 执行={r.execution_accuracy:.0%} 完成={r.completion_rate:.0%} 轮数={r.rounds_used}")
