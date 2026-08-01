"""工具调用评测.

评测内容：
    - 工具选择准确率：模型能否从N个工具中选对正确的
    - 参数填充正确率：调用参数是否完整且类型正确
    - 多工具编排：能否按正确顺序调用多个工具

基于 ToolBench 子集的评测思路，但用自建场景保证可运行。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from ..adapters.base import BaseModelAdapter, GenerationResult


# ========== 工具定义（模拟 API） ==========

TOOL_DEFINITIONS = [
    {
        "name": "get_weather",
        "description": "获取指定城市的天气信息",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "城市名称"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "温度单位"},
            },
            "required": ["city"],
        },
    },
    {
        "name": "search_web",
        "description": "搜索互联网上的信息",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "搜索关键词"},
                "num_results": {"type": "integer", "description": "返回结果数量", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "send_email",
        "description": "发送电子邮件",
        "parameters": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "收件人邮箱"},
                "subject": {"type": "string", "description": "邮件主题"},
                "body": {"type": "string", "description": "邮件正文"},
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "calculator",
        "description": "执行数学计算",
        "parameters": {
            "type": "object",
            "properties": {
                "expression": {"type": "string", "description": "数学表达式，如 '2+3*4'"},
            },
            "required": ["expression"],
        },
    },
    {
        "name": "translate_text",
        "description": "翻译文本到指定语言",
        "parameters": {
            "type": "object",
            "properties": {
                "text": {"type": "string", "description": "要翻译的文本"},
                "target_lang": {"type": "string", "description": "目标语言，如 'en', 'ja', 'ko'"},
            },
            "required": ["text", "target_lang"],
        },
    },
    {
        "name": "get_stock_price",
        "description": "获取股票实时价格",
        "parameters": {
            "type": "object",
            "properties": {
                "symbol": {"type": "string", "description": "股票代码，如 'AAPL'"},
                "exchange": {"type": "string", "description": "交易所，如 'NASDAQ'", "default": "US"},
            },
            "required": ["symbol"],
        },
    },
]


# ========== 评测用例 ==========

@dataclass
class ToolCallTestCase:
    """工具调用评测用例."""

    test_id: str
    query: str                          # 用户请求
    available_tools: List[str]          # 可用工具名列表
    expected_tool: str                  # 期望调用的工具
    expected_params: Dict[str, Any]     # 期望的参数
    multi_tool: bool = False            # 是否需要多工具编排
    expected_tools_order: Optional[List[str]] = None  # 多工具编排的期望顺序


# 内置测试用例

BUILTIN_TOOL_CASES = [
    # 单工具调用
    ToolCallTestCase(
        test_id="tc_001",
        query="北京今天天气怎么样？",
        available_tools=["get_weather", "search_web", "calculator"],
        expected_tool="get_weather",
        expected_params={"city": "北京"},
    ),
    ToolCallTestCase(
        test_id="tc_002",
        query="帮我算一下 (15 + 27) * 3 等于多少",
        available_tools=["get_weather", "search_web", "calculator", "translate_text"],
        expected_tool="calculator",
        expected_params={"expression": "(15+27)*3"},
    ),
    ToolCallTestCase(
        test_id="tc_003",
        query="把'你好世界'翻译成英文",
        available_tools=["search_web", "translate_text", "send_email"],
        expected_tool="translate_text",
        expected_params={"text": "你好世界", "target_lang": "en"},
    ),
    ToolCallTestCase(
        test_id="tc_004",
        query="查一下苹果公司最新的股价",
        available_tools=["get_weather", "search_web", "get_stock_price"],
        expected_tool="get_stock_price",
        expected_params={"symbol": "AAPL"},
    ),
    ToolCallTestCase(
        test_id="tc_005",
        query="发邮件给 zhangsan@example.com，主题是会议通知，内容是明天下午3点开会",
        available_tools=["search_web", "send_email", "translate_text"],
        expected_tool="send_email",
        expected_params={"to": "zhangsan@example.com", "subject": "会议通知", "body": "明天下午3点开会"},
    ),
    # 多工具编排
    ToolCallTestCase(
        test_id="tc_006",
        query="查一下东京的天气，然后把天气信息翻译成日语发邮件给 tanaka@example.com",
        available_tools=["get_weather", "translate_text", "send_email", "search_web"],
        expected_tool="get_weather",
        expected_params={"city": "东京"},
        multi_tool=True,
        expected_tools_order=["get_weather", "translate_text", "send_email"],
    ),
    ToolCallTestCase(
        test_id="tc_007",
        query="先搜索2024年奥运会举办城市，然后查那个城市的天气",
        available_tools=["search_web", "get_weather", "calculator"],
        expected_tool="search_web",
        expected_params={"query": "2024年奥运会举办城市"},
        multi_tool=True,
        expected_tools_order=["search_web", "get_weather"],
    ),
    ToolCallTestCase(
        test_id="tc_008",
        query="帮我查微软的股价，然后算一下100股多少钱",
        available_tools=["get_stock_price", "calculator", "search_web"],
        expected_tool="get_stock_price",
        expected_params={"symbol": "MSFT"},
        multi_tool=True,
        expected_tools_order=["get_stock_price", "calculator"],
    ),
]


# ========== 评测结果 ==========

@dataclass
class ToolCallResult:
    """单条工具调用评测结果."""

    test_id: str
    query: str
    expected_tool: str
    predicted_tool: str = ""
    tool_selection_correct: bool = False
    expected_params: Dict[str, Any] = field(default_factory=dict)
    predicted_params: Dict[str, Any] = field(default_factory=dict)
    params_match_rate: float = 0.0  # 参数匹配率
    required_params_filled: bool = False
    multi_tool_order_correct: bool = False
    raw_response: str = ""


@dataclass
class ToolCallReport:
    """工具调用评测报告."""

    model_name: str
    total_cases: int = 0
    tool_selection_accuracy: float = 0.0
    params_fill_accuracy: float = 0.0
    required_params_fill_rate: float = 0.0
    multi_tool_order_accuracy: float = 0.0
    overall_accuracy: float = 0.0
    results: List[ToolCallResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_cases": self.total_cases,
            "tool_selection_accuracy": f"{self.tool_selection_accuracy:.1%}",
            "params_fill_accuracy": f"{self.params_fill_accuracy:.1%}",
            "required_params_fill_rate": f"{self.required_params_fill_rate:.1%}",
            "multi_tool_order_accuracy": f"{self.multi_tool_order_accuracy:.1%}",
            "overall_accuracy": f"{self.overall_accuracy:.1%}",
            "results": [
                {
                    "test_id": r.test_id,
                    "query": r.query[:50],
                    "expected_tool": r.expected_tool,
                    "predicted_tool": r.predicted_tool,
                    "tool_correct": r.tool_selection_correct,
                    "params_match": f"{r.params_match_rate:.1%}",
                    "required_filled": r.required_params_filled,
                }
                for r in self.results
            ],
        }


class ToolCallEvaluator:
    """工具调用评测器."""

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter

    def _build_prompt(self, test_case: ToolCallTestCase) -> str:
        """构建工具调用 prompt."""
        # 筛选可用工具
        tools = [t for t in TOOL_DEFINITIONS if t["name"] in test_case.available_tools]
        tools_json = json.dumps(tools, ensure_ascii=False, indent=2)

        prompt = f"""你是一个AI助手，可以使用以下工具来帮助用户完成任务。

可用工具：
{tools_json}

用户请求：{test_case.query}

请分析用户请求，选择合适的工具并填充参数。请以JSON格式输出：
{{
    "tool_calls": [
        {{
            "name": "工具名称",
            "arguments": {{
                "参数名": "参数值"
            }}
        }}
    ]
}}

如果需要多个工具，按执行顺序列出。只输出JSON，不要其他内容。"""

        return prompt

    def _parse_tool_calls(self, response: str) -> List[Dict[str, Any]]:
        """解析模型回复中的工具调用."""
        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                return data.get("tool_calls", [])
            except json.JSONDecodeError:
                pass

        # 降级：尝试从文本中提取工具名
        calls = []
        for tool_def in TOOL_DEFINITIONS:
            if tool_def["name"] in response:
                calls.append({"name": tool_def["name"], "arguments": {}})

        return calls

    def _compute_params_match(
        self,
        expected: Dict[str, Any],
        predicted: Dict[str, Any],
        required_keys: List[str],
    ) -> Tuple[float, bool]:
        """计算参数匹配率.

        Returns:
            (匹配率, 必填参数是否全部填充)
        """
        if not expected:
            return 1.0, True

        matched = 0
        total = len(expected)
        required_filled = True

        for key, expected_val in expected.items():
            if key in predicted:
                pred_val = str(predicted[key]).lower()
                exp_val = str(expected_val).lower()
                # 模糊匹配：只要包含关键部分就算对
                if exp_val in pred_val or pred_val in exp_val:
                    matched += 1
            if key in required_keys and key not in predicted:
                required_filled = False

        match_rate = matched / total if total > 0 else 0.0
        return match_rate, required_filled

    def evaluate_single(self, test_case: ToolCallTestCase) -> ToolCallResult:
        """评测单个工具调用用例."""
        prompt = self._build_prompt(test_case)
        result = self._adapter.chat(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=512,
            temperature=0.0,
        )

        raw_response = result.text if result.success else f"[ERROR: {result.error}]"
        tool_calls = self._parse_tool_calls(raw_response) if result.success else []

        # 评估工具选择
        predicted_tool = tool_calls[0]["name"] if tool_calls else ""
        tool_correct = predicted_tool == test_case.expected_tool

        # 评估参数填充
        predicted_params = tool_calls[0].get("arguments", {}) if tool_calls else {}
        required_keys = self._get_required_params(test_case.expected_tool)
        params_match, required_filled = self._compute_params_match(
            test_case.expected_params, predicted_params, required_keys
        )

        # 多工具编排评估
        multi_tool_correct = False
        if test_case.multi_tool and test_case.expected_tools_order:
            predicted_order = [c["name"] for c in tool_calls]
            multi_tool_correct = predicted_order == test_case.expected_tools_order

        return ToolCallResult(
            test_id=test_case.test_id,
            query=test_case.query,
            expected_tool=test_case.expected_tool,
            predicted_tool=predicted_tool,
            tool_selection_correct=tool_correct,
            expected_params=test_case.expected_params,
            predicted_params=predicted_params,
            params_match_rate=params_match,
            required_params_filled=required_filled,
            multi_tool_order_correct=multi_tool_correct,
            raw_response=raw_response,
        )

    def _get_required_params(self, tool_name: str) -> List[str]:
        """获取工具的必填参数列表."""
        for tool in TOOL_DEFINITIONS:
            if tool["name"] == tool_name:
                return tool["parameters"].get("required", [])
        return []

    def evaluate_batch(self, cases: Optional[List[ToolCallTestCase]] = None) -> ToolCallReport:
        """批量评测工具调用."""
        cases = cases or BUILTIN_TOOL_CASES
        report = ToolCallReport(
            model_name=getattr(self._adapter, '_config', None)
            and self._adapter._config.model_name or "unknown"
        )

        for case in cases:
            r = self.evaluate_single(case)
            report.results.append(r)

        report.total_cases = len(report.results)

        # 工具选择准确率
        tool_correct = sum(1 for r in report.results if r.tool_selection_correct)
        report.tool_selection_accuracy = tool_correct / report.total_cases if report.total_cases > 0 else 0.0

        # 参数填充准确率
        report.params_fill_accuracy = (
            sum(r.params_match_rate for r in report.results) / report.total_cases
            if report.total_cases > 0 else 0.0
        )

        # 必填参数填充率
        required_filled = sum(1 for r in report.results if r.required_params_filled)
        report.required_params_fill_rate = required_filled / report.total_cases if report.total_cases > 0 else 0.0

        # 多工具编排准确率
        multi_tool_cases = [r for r in report.results if r.test_id in
                           [c.test_id for c in cases if c.multi_tool]]
        if multi_tool_cases:
            multi_correct = sum(1 for r in multi_tool_cases if r.multi_tool_order_correct)
            report.multi_tool_order_accuracy = multi_correct / len(multi_tool_cases)
        else:
            report.multi_tool_order_accuracy = 0.0

        # 综合准确率
        report.overall_accuracy = (
            report.tool_selection_accuracy * 0.4
            + report.params_fill_accuracy * 0.3
            + report.required_params_fill_rate * 0.3
        )

        return report

    def save_report(self, report: ToolCallReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"tool_call_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: ToolCallReport) -> None:
        """打印报告."""
        print(f"\n工具调用评测报告 - {report.model_name}")
        print(f"  总用例数: {report.total_cases}")
        print(f"  工具选择准确率: {report.tool_selection_accuracy:.1%}")
        print(f"  参数填充准确率: {report.params_fill_accuracy:.1%}")
        print(f"  必填参数填充率: {report.required_params_fill_rate:.1%}")
        print(f"  多工具编排准确率: {report.multi_tool_order_accuracy:.1%}")
        print(f"  综合准确率: {report.overall_accuracy:.1%}")
