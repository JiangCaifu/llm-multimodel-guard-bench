#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
指标对比脚本 - 用于PR门禁和定时回归的质量退化检测

对比两组评测结果，检测核心指标是否退化。

用法:
    # 对比两个JSON文件
    python scripts/compare_metrics.py \
        --current data/results/pr/metrics.json \
        --previous data/results/baseline/metrics.json \
        --threshold 0.05 \
        --output data/results/pr/comparison.json

    # 对比两个目录（自动查找JSON文件）
    python scripts/compare_metrics.py \
        --current data/results/nightly/ \
        --previous data/results/baseline/ \
        --threshold 0.05
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# 需要监控的核心指标及其阈值
DEFAULT_METRICS = {
    "safety_pass_rate": {"threshold": 0.05, "direction": "up"},      # 越高越好
    "capability_accuracy": {"threshold": 0.05, "direction": "up"},    # 越高越好
    "perf_ttft_ms_p95": {"threshold": 0.10, "direction": "down"},      # 越低越好
    "perf_throughput_rps": {"threshold": 0.10, "direction": "up"},    # 越高越好
    "agent_task_completion": {"threshold": 0.05, "direction": "up"},  # 越高越好
}


@dataclass
class MetricDelta:
    """单个指标的变化."""
    name: str
    current: float = 0.0
    previous: float = 0.0
    delta: float = 0.0
    threshold: float = 0.05
    direction: str = "up"  # up=越高越好, down=越低越好
    is_regression: bool = False
    is_improved: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "current": round(self.current, 4),
            "previous": round(self.previous, 4),
            "delta": round(self.delta, 4),
            "threshold": self.threshold,
            "direction": self.direction,
            "is_regression": self.is_regression,
            "is_improved": self.is_improved,
        }


@dataclass
class ComparisonResult:
    """对比结果."""
    deltas: List[MetricDelta] = field(default_factory=list)
    has_regression: bool = False
    regressions: List[str] = field(default_factory=list)
    improvements: List[str] = field(default_factory=list)
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        result: Dict[str, Any] = {
            "has_regression": self.has_regression,
            "regressions": self.regressions,
            "improvements": self.improvements,
            "summary": self.summary,
        }
        # 展开各指标
        for d in self.deltas:
            result[d.name] = d.current
            result[f"{d.name}_delta"] = d.delta
            result[f"{d.name}_previous"] = d.previous
        return result


def load_metrics(path: str) -> Dict[str, float]:
    """加载指标数据，支持文件或目录."""
    data: Dict[str, float] = {}

    if os.path.isfile(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif os.path.isdir(path):
        # 目录模式：递归查找所有JSON文件并提取指标
        for json_file in glob.glob(os.path.join(path, "**", "*.json"), recursive=True):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = json.load(f)

                # 尝试提取常见指标
                if isinstance(content, dict):
                    # 安全通过率
                    if "total_samples" in content and "p0_count" in content:
                        total = content["total_samples"]
                        p0 = content["p0_count"]
                        rate = (total - p0) / total if total > 0 else 0
                        data["safety_pass_rate"] = round(rate, 4)

                    # 能力准确率
                    if "accuracy" in content:
                        data["capability_accuracy"] = round(content["accuracy"], 4)

                    # 性能指标
                    if "ttft_ms_p95" in content:
                        data["perf_ttft_ms_p95"] = content["ttft_ms_p95"]
                    if "throughput_rps" in content:
                        data["perf_throughput_rps"] = content["throughput_rps"]

                    # Agent指标
                    if "task_completion_rate" in content:
                        data["agent_task_completion"] = content["task_completion_rate"]
            except (json.JSONDecodeError, IOError):
                pass

    return data


def compare_metrics(
    current: Dict[str, float],
    previous: Dict[str, float],
    thresholds: Optional[Dict[str, Dict[str, Any]]] = None,
) -> ComparisonResult:
    """对比两组指标."""
    if thresholds is None:
        thresholds = DEFAULT_METRICS

    result = ComparisonResult()

    for metric_name, config in thresholds.items():
        threshold = config.get("threshold", 0.05)
        direction = config.get("direction", "up")

        cur_val = current.get(metric_name)
        prev_val = previous.get(metric_name)

        if cur_val is None or prev_val is None:
            continue

        delta = cur_val - prev_val

        # 判断是否退化
        is_regression = False
        is_improved = False

        if direction == "up":
            # 越高越好
            if delta < 0 and abs(delta) > threshold:
                is_regression = True
            elif delta > 0 and abs(delta) > threshold:
                is_improved = True
        else:
            # 越低越好
            if delta > 0 and abs(delta) > threshold:
                is_regression = True
            elif delta < 0 and abs(delta) > threshold:
                is_improved = True

        delta_obj = MetricDelta(
            name=metric_name,
            current=cur_val,
            previous=prev_val,
            delta=delta,
            threshold=threshold,
            direction=direction,
            is_regression=is_regression,
            is_improved=is_improved,
        )
        result.deltas.append(delta_obj)

        if is_regression:
            result.has_regression = True
            result.regressions.append(
                f"{metric_name}: {prev_val:.4f} → {cur_val:.4f} (Δ{delta:+.4f}, 阈值={threshold})"
            )
        elif is_improved:
            result.improvements.append(
                f"{metric_name}: {prev_val:.4f} → {cur_val:.4f} (Δ{delta:+.4f})"
            )

    # 生成总结
    if result.has_regression:
        result.summary = f"检测到 {len(result.regressions)} 项质量退化"
    elif result.improvements:
        result.summary = f"所有指标正常，{len(result.improvements)} 项改进"
    else:
        result.summary = "所有指标稳定，无明显变化"

    return result


def print_comparison(result: ComparisonResult) -> None:
    """打印对比结果."""
    print("\n" + "=" * 60)
    print("指标对比结果")
    print("=" * 60)

    for d in result.deltas:
        status = "⚠️ 退化" if d.is_regression else ("✅ 改进" if d.is_improved else "➡️ 稳定")
        arrow = "↑" if d.delta > 0 else ("↓" if d.delta < 0 else "→")
        print(f"  {d.name:<30} {d.previous:.4f} → {d.current:.4f} ({arrow}{abs(d.delta):.4f}) {status}")

    print(f"\n结论: {result.summary}")

    if result.has_regression:
        print("\n退化详情:")
        for r in result.regressions:
            print(f"  - {r}")


def main() -> int:
    parser = argparse.ArgumentParser(description="对比两组评测指标")
    parser.add_argument("--current", required=True, help="当前指标（JSON文件或目录）")
    parser.add_argument("--previous", required=True, help="基线指标（JSON文件或目录）")
    parser.add_argument("--threshold", type=float, default=0.05, help="默认退化阈值")
    parser.add_argument("--output", default=None, help="输出对比报告路径")
    parser.add_argument("--verbose", "-v", action="store_true", help="详细输出")
    args = parser.parse_args()

    # 加载指标
    current_metrics = load_metrics(args.current)
    previous_metrics = load_metrics(args.previous)

    if not current_metrics:
        print(f"[ERROR] 无法从 {args.current} 加载指标")
        return 1
    if not previous_metrics:
        print(f"[WARN] 无法从 {args.previous} 加载指标，使用默认基线")
        previous_metrics = {"safety_pass_rate": 0.95, "capability_accuracy": 0.85}

    if args.verbose:
        print(f"\n当前指标 ({len(current_metrics)}项):")
        for k, v in current_metrics.items():
            print(f"  {k}: {v}")
        print(f"\n基线指标 ({len(previous_metrics)}项):")
        for k, v in previous_metrics.items():
            print(f"  {k}: {v}")

    # 对比
    result = compare_metrics(current_metrics, previous_metrics)

    # 打印结果
    print_comparison(result)

    # 保存
    if args.output:
        os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result.to_dict(), f, indent=2, ensure_ascii=False)
        print(f"\n对比报告已保存: {args.output}")

    # 退出码
    return 1 if result.has_regression else 0


if __name__ == "__main__":
    sys.exit(main())
