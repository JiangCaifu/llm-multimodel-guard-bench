"""F1: LLM自动生成测试用例.

输入：需求文档 / 接口定义 / 用户故事
输出：结构化测试用例（正常/边界/异常/安全四类）

流程：
    1. 解析输入（自然语言需求或结构化定义）
    2. LLM生成用例（覆盖四种类型）
    3. 去重 + 质量检查（可执行率、重复率）
    4. 输出结构化JSON
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter


class CaseType(str, Enum):
    """用例类型."""

    NORMAL = "normal"          # 正常流程
    BOUNDARY = "boundary"      # 边界值
    EXCEPTION = "exception"    # 异常场景
    SECURITY = "security"      # 安全场景


class CasePriority(str, Enum):
    """用例优先级."""

    P0 = "P0"  # 冒烟
    P1 = "P1"  # 核心
    P2 = "P2"  # 扩展
    P3 = "P3"  # 补充


@dataclass
class TestCase:
    """单条测试用例."""

    case_id: str
    title: str
    case_type: CaseType
    priority: CasePriority
    preconditions: str = ""
    steps: List[str] = field(default_factory=list)
    expected_result: str = ""
    input_data: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "title": self.title,
            "case_type": self.case_type.value,
            "priority": self.priority.value,
            "preconditions": self.preconditions,
            "steps": self.steps,
            "expected_result": self.expected_result,
            "input_data": self.input_data,
            "tags": self.tags,
        }


@dataclass
class CaseGenerationReport:
    """用例生成报告."""

    source: str
    total_cases: int = 0
    by_type: Dict[str, int] = field(default_factory=dict)
    by_priority: Dict[str, int] = field(default_factory=dict)
    duplicate_count: int = 0
    cases: List[TestCase] = field(default_factory=list)
    generation_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "source": self.source,
            "total_cases": self.total_cases,
            "by_type": self.by_type,
            "by_priority": self.by_priority,
            "duplicate_count": self.duplicate_count,
            "cases": [c.to_dict() for c in self.cases],
            "generation_time_ms": self.generation_time_ms,
        }


# 生成用例的 Prompt 模板
_CASE_GEN_PROMPT = """你是一位资深测试工程师，请根据以下需求生成结构化测试用例。

需求内容：
{requirement}

请生成以下四类测试用例：
1. 正常流程用例（normal）：验证核心功能按预期工作
2. 边界值用例（boundary）：测试输入/输出的边界条件
3. 异常场景用例（exception）：测试错误输入、异常状态
4. 安全场景用例（security）：测试注入、越权、敏感信息泄露

每条用例格式如下（JSON数组）：
```json
[
  {{
    "title": "用例标题",
    "case_type": "normal|boundary|exception|security",
    "priority": "P0|P1|P2|P3",
    "preconditions": "前置条件",
    "steps": ["步骤1", "步骤2"],
    "expected_result": "期望结果",
    "input_data": "测试输入数据",
    "tags": ["标签1"]
  }}
]
```

要求：
- 每类至少生成2条用例
- 优先级分配合理（P0≤20%, P1≤40%, P2≤30%, P3≥10%）
- 步骤要具体可执行
- 只输出JSON，不要其他内容
"""


class CaseGenerator:
    """测试用例自动生成器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter

    def _parse_cases(self, raw: str) -> List[TestCase]:
        """从LLM输出中解析测试用例."""
        # 优先提取 ```json ... ``` 代码块
        code_block = re.search(r'```(?:json)?\s*\n?([\s\S]*?)\n?```', raw)
        json_str = code_block.group(1) if code_block else raw

        # 去掉首尾非JSON字符，找到数组
        arr_start = json_str.find('[')
        arr_end = json_str.rfind(']')
        if arr_start == -1 or arr_end == -1 or arr_end <= arr_start:
            return []

        try:
            items = json.loads(json_str[arr_start:arr_end + 1])
        except json.JSONDecodeError:
            return []

        cases = []
        for i, item in enumerate(items):
            case_type_str = item.get("case_type", "normal")
            try:
                case_type = CaseType(case_type_str)
            except ValueError:
                case_type = CaseType.NORMAL

            priority_str = item.get("priority", "P2")
            try:
                priority = CasePriority(priority_str)
            except ValueError:
                priority = CasePriority.P2

            case = TestCase(
                case_id=f"tc_{i+1:03d}",
                title=item.get("title", f"用例{i+1}"),
                case_type=case_type,
                priority=priority,
                preconditions=item.get("preconditions", ""),
                steps=item.get("steps", []),
                expected_result=item.get("expected_result", ""),
                input_data=item.get("input_data", ""),
                tags=item.get("tags", []),
            )
            cases.append(case)

        return cases

    def _deduplicate(self, cases: List[TestCase]) -> List[TestCase]:
        """去重：基于标题相似度."""
        seen_titles = set()
        unique = []
        for case in cases:
            # 简单去重：标题去掉空格后比较
            normalized = re.sub(r'\s+', '', case.title)
            if normalized not in seen_titles:
                seen_titles.add(normalized)
                unique.append(case)
        return unique

    def generate(self, requirement: str) -> CaseGenerationReport:
        """根据需求生成测试用例."""
        import time

        start = time.time()
        report = CaseGenerationReport(source=requirement[:200])

        # 调用LLM生成用例
        prompt = _CASE_GEN_PROMPT.format(requirement=requirement)
        result = self._adapter.generate(prompt, max_tokens=2048)

        # 调试：保存原始输出
        import sys
        raw_text = result.text
        print(f"  [DEBUG] LLM返回长度: {len(raw_text)} 字符", file=sys.stderr)
        print(f"  [DEBUG] 前500字符: {raw_text[:500]}", file=sys.stderr)

        # 解析
        cases = self._parse_cases(raw_text)
        before_dedup = len(cases)

        # 去重
        cases = self._deduplicate(cases)
        report.duplicate_count = before_dedup - len(cases)

        # 统计
        report.cases = cases
        report.total_cases = len(cases)
        for case in cases:
            ct = case.case_type.value
            pr = case.priority.value
            report.by_type[ct] = report.by_type.get(ct, 0) + 1
            report.by_priority[pr] = report.by_priority.get(pr, 0) + 1

        report.generation_time_ms = (time.time() - start) * 1000
        return report

    def generate_batch(self, requirements: List[str]) -> List[CaseGenerationReport]:
        """批量生成."""
        return [self.generate(r) for r in requirements]

    def save_report(self, report: CaseGenerationReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        # 文件名从source中提取
        safe_name = re.sub(r'[^\w]', '_', report.source[:30]).strip('_')
        path = os.path.join(output_dir, f"cases_{safe_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    @staticmethod
    def print_report(report: CaseGenerationReport) -> None:
        """打印报告."""
        print(f"\n测试用例生成报告")
        print(f"  需求摘要: {report.source[:80]}...")
        print(f"  生成用例数: {report.total_cases}")
        print(f"  去重数: {report.duplicate_count}")
        print(f"  耗时: {report.generation_time_ms:.0f}ms")

        print(f"\n  按类型分布:")
        for ct, count in sorted(report.by_type.items()):
            print(f"    {ct}: {count}")

        print(f"\n  按优先级分布:")
        for pr, count in sorted(report.by_priority.items()):
            print(f"    {pr}: {count}")

        print(f"\n  用例详情:")
        for case in report.cases:
            print(f"    [{case.case_type.value}][{case.priority.value}] {case.title}")
            if case.steps:
                for j, step in enumerate(case.steps, 1):
                    print(f"      步骤{j}: {step}")
            print(f"      期望: {case.expected_result[:60]}")
