"""AI驱动测试工具 - 统一入口 + 效果评估.

编排 F1/F2/F3 的评测流程，并生成效果评估报告。
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..adapters.base import BaseModelAdapter
from ..adapters.factory import build_adapter, load_model_config
from .bug_detector import BugDetector, BugDetectionReport
from .case_generator import CaseGenerator, CaseGenerationReport
from .data_factory import DataFactory, DataFactoryReport, BUILTIN_TEMPLATES


@dataclass
class AITestReport:
    """AI测试工具效果评估报告."""

    model_name: str = ""
    # F1 结果
    case_generation: Optional[CaseGenerationReport] = None
    # F2 结果
    bug_detection: Optional[BugDetectionReport] = None
    # F3 结果
    data_factory: Optional[DataFactoryReport] = None
    # 整体评估
    total_time_ms: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        d: Dict[str, Any] = {
            "model_name": self.model_name,
            "total_time_ms": self.total_time_ms,
        }
        if self.case_generation:
            d["case_generation"] = self.case_generation.to_dict()
        if self.bug_detection:
            d["bug_detection"] = self.bug_detection.to_dict()
        if self.data_factory:
            d["data_factory"] = self.data_factory.to_dict()
        return d


class AITestRunner:
    """AI测试工具运行器."""

    # 示例需求（用于演示）
    DEMO_REQUIREMENTS = [
        "用户注册功能：支持邮箱和手机号注册，密码需8-20位，包含字母和数字，注册后发送验证邮件",
        "商品搜索功能：支持关键词搜索和分类筛选，支持价格区间过滤，结果按相关度排序",
        "在线支付功能：支持支付宝和微信支付，支付超时15分钟自动取消，失败自动退款",
    ]

    # 示例缺陷检测数据
    DEMO_BUG_SAMPLES = [
        {
            "user_input": "帮我查一下北京明天的天气",
            "expected": "提供北京明天的天气预报，包含温度、天气状况、风力",
            "actual": "北京明天的天气是晴天，气温25度。",  # 遗漏了风力和天气详细描述
        },
        {
            "user_input": "1+1等于多少",
            "expected": "1+1=2",
            "actual": "1+1=2",
        },
        {
            "user_input": "请翻译：Hello World",
            "expected": "你好世界",
            "actual": "哈喽世界",  # 翻译不准确
        },
        {
            "user_input": "写一个Python快排",
            "expected": "提供正确的快速排序Python代码",
            "actual": "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr if x < pivot]\n    right = [x for x in arr if x > pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",  # 正确
        },
        {
            "user_input": "给我推荐一款手机",
            "expected": "根据需求推荐1-3款手机，说明推荐理由",
            "actual": "我推荐iPhone。",  # 遗漏推荐理由，信息不完整
        },
    ]

    def __init__(self, adapter: BaseModelAdapter) -> None:
        self._adapter = adapter
        self._case_gen = CaseGenerator(adapter)
        self._bug_detector = BugDetector(adapter)
        self._data_factory = DataFactory(adapter)

    def run_case_generation(
        self, requirements: Optional[List[str]] = None
    ) -> CaseGenerationReport:
        """运行用例生成."""
        reqs = requirements or self.DEMO_REQUIREMENTS
        # 合并需求为一个prompt
        combined = "\n---\n".join(f"需求{i+1}: {r}" for i, r in enumerate(reqs))
        return self._case_gen.generate(combined)

    def run_bug_detection(
        self, samples: Optional[List[Dict[str, str]]] = None
    ) -> BugDetectionReport:
        """运行缺陷检测."""
        data = samples or self.DEMO_BUG_SAMPLES
        return self._bug_detector.detect_batch(data)

    def run_data_factory(
        self, template_name: str = "user_profile"
    ) -> DataFactoryReport:
        """运行数据构造."""
        template = BUILTIN_TEMPLATES.get(template_name)
        if not template:
            raise ValueError(f"未知模板: {template_name}，可选: {list(BUILTIN_TEMPLATES.keys())}")
        return self._data_factory.generate(template)

    def run_all(self) -> AITestReport:
        """运行全部评测."""
        import time

        start = time.time()
        report = AITestReport(model_name=self._adapter.model_name)

        print("=" * 60)
        print("F1: 测试用例自动生成")
        print("=" * 60)
        report.case_generation = self.run_case_generation()
        CaseGenerator.print_report(report.case_generation)

        print("\n" + "=" * 60)
        print("F2: 缺陷识别与分类")
        print("=" * 60)
        report.bug_detection = self.run_bug_detection()
        BugDetector.print_report(report.bug_detection)

        print("\n" + "=" * 60)
        print("F3: 测试数据构造")
        print("=" * 60)
        report.data_factory = self.run_data_factory()
        DataFactory.print_report(report.data_factory)

        report.total_time_ms = (time.time() - start) * 1000

        print("\n" + "=" * 60)
        print(f"AI测试工具效果评估完成 (耗时 {report.total_time_ms:.0f}ms)")
        print("=" * 60)

        return report

    def save_report(self, report: AITestReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"aitest_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path
