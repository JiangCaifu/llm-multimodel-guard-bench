#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
统一评测入口脚本 - 跨平台（Windows/Linux/Mac）

用法:
    python scripts/run_eval.py smoke
    python scripts/run_eval.py core
    python scripts/run_eval.py full
    python scripts/run_eval.py --only safety
    python scripts/run_eval.py --skip agent
    python scripts/run_eval.py --model qwen-plus
    python scripts/run_eval.py --max-samples 10
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from typing import List, Optional


# ---------- 配置 ----------
AVAILABLE_MODULES = ["tests", "safety", "capability", "perf", "agent", "experience", "aitest", "alignment"]

LEVEL_CONFIG = {
    "smoke": {
        "desc": "快速评测（<1min）",
        "max_samples": {"safety": 3, "capability": 5},
        "skip": ["perf", "agent", "experience", "aitest"],
    },
    "core": {
        "desc": "核心评测（<5min）",
        "max_samples": {},
        "perf_mode": "quick",
        "alignment_samples": 10,
    },
    "full": {
        "desc": "全量评测（<30min）",
        "max_samples": {},
        "perf_mode": "full",
        "alignment_samples": 30,
    },
}


def log(level: str, msg: str) -> None:
    colors = {"INFO": "\033[36m", "WARN": "\033[33m", "ERROR": "\033[31m", "SUCCESS": "\033[32m"}
    reset = "\033[0m"
    prefix = f"[{level}] {time.strftime('%H:%M:%S')}"
    print(f"{colors.get(level, '')}{prefix} {msg}{reset}")


def run_cmd(cmd: List[str], timeout: int = 600) -> bool:
    """运行命令并实时输出."""
    log("INFO", f"执行: {' '.join(cmd)}")
    try:
        result = subprocess.run(
            cmd,
            capture_output=False,
            text=True,
            timeout=timeout,
            env={**os.environ, "PYTHONIOENCODING": "utf-8"},
        )
        if result.returncode != 0:
            log("ERROR", f"命令失败 (退出码={result.returncode})")
            return False
        return True
    except subprocess.TimeoutExpired:
        log("ERROR", f"命令超时 ({timeout}s)")
        return False
    except Exception as e:
        log("ERROR", f"命令异常: {e}")
        return False


def should_run(module: str, only: str = "", skip: str = "") -> bool:
    """检查模块是否应执行."""
    if only:
        return module in [m.strip() for m in only.split(",")]
    if skip:
        return module not in [m.strip() for m in skip.split(",")]
    return True


# 全局状态（简化跨函数传参）
_ONLY = ""
_SKIP = ""


def run_tests() -> bool:
    if not should_run("tests", _ONLY, _SKIP):
        return True
    log("INFO", "运行单元测试...")
    return run_cmd([sys.executable, "-m", "pytest", "tests/", "-v", "--tb=short", "-p", "no:cacheprovider", "--override-ini", "addopts="])


def run_safety(level: str, model: str, output_dir: str, max_samples: dict) -> bool:
    if not should_run("safety", _ONLY, _SKIP):
        return True
    log("INFO", "运行安全评测...")

    levels_to_run = []
    if level == "smoke":
        levels_to_run = [1]
    elif level == "core":
        levels_to_run = [1, 2]
    else:
        levels_to_run = [1, 2, 3]

    for lv in levels_to_run:
        cmd = [
            sys.executable, "-m", "llm_guard_bench.cli", "safety",
            "--level", str(lv),
            "--model", model,
            "--output", output_dir,
        ]
        if level == "smoke":
            cmd.extend(["--max-samples", str(max_samples.get("safety", 3))])
        if not run_cmd(cmd):
            return False
    log("INFO", "安全评测完成")
    return True


def run_capability(level: str, model: str, output_dir: str, max_samples: dict) -> bool:
    if not should_run("capability", _ONLY, _SKIP):
        return True
    log("INFO", "运行能力评测...")

    cmd = [
        sys.executable, "-m", "llm_guard_bench.cli", "run",
        "--config", "configs/evaluation/capability.yaml",
        "--model", model,
        "--output", output_dir,
    ]
    # run 子命令不支持 --max-samples，跳过
    if not run_cmd(cmd):
        return False
    log("INFO", "能力评测完成")
    return True


def run_performance(level: str, model: str, output_dir: str) -> bool:
    if not should_run("perf", _ONLY, _SKIP):
        return True
    if level == "smoke":
        log("INFO", "Smoke模式跳过性能基准")
        return True
    log("INFO", "运行性能基准...")

    cmd = [
        sys.executable, "-m", "llm_guard_bench.cli", "perf",
        "--model", model,
        "--output", output_dir,
    ]
    if level == "core":
        cmd.insert(5, "--quick")
    if not run_cmd(cmd):
        return False
    log("INFO", "性能基准完成")
    return True


def run_agent(level: str, model: str, output_dir: str) -> bool:
    if not should_run("agent", _ONLY, _SKIP):
        return True
    if level == "smoke":
        log("INFO", "Smoke模式跳过Agent评测")
        return True
    log("INFO", "运行Agent评测...")
    return run_cmd([
        sys.executable, "-m", "llm_guard_bench.cli", "agent",
        "--task", "full",
        "--model", model,
        "--output", output_dir,
    ])


def run_experience(level: str, model: str, output_dir: str) -> bool:
    if not should_run("experience", _ONLY, _SKIP):
        return True
    if level == "smoke":
        log("INFO", "Smoke模式跳过体验评测")
        return True
    log("INFO", "运行体验评测...")
    return run_cmd([
        sys.executable, "-m", "llm_guard_bench.cli", "experience",
        "--task", "full",
        "--model", model,
        "--output", output_dir,
    ])


def run_aitest(level: str, model: str, output_dir: str) -> bool:
    if not should_run("aitest", _ONLY, _SKIP):
        return True
    if level == "smoke":
        log("INFO", "Smoke模式跳过AI测试工具")
        return True
    log("INFO", "运行AI测试工具...")
    return run_cmd([
        sys.executable, "-m", "llm_guard_bench.cli", "aitest",
        "--task", "full",
        "--model", model,
        "--output", output_dir,
    ])


def run_alignment(level: str, model: str, output_dir: str, samples: int) -> bool:
    if not should_run("alignment", _ONLY, _SKIP):
        return True
    log("INFO", "运行Judge对齐验证...")
    # alignment 子命令不支持 --model 参数
    return run_cmd([
        sys.executable, "-m", "llm_guard_bench.cli", "alignment",
        "--task", "demo",
        "--samples", str(samples),
        "--output", output_dir,
    ])


def save_summary(output_dir: str, level: str, model: str) -> None:
    """保存评测汇总."""
    summary = {
        "level": level,
        "model": model,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "results": {},
    }

    for mod in ["safety", "capability", "perf", "agent", "experience", "aitest", "alignment"]:
        count = len([f for f in os.listdir(output_dir) if f.startswith(mod) and f.endswith(".json")])
        if count > 0:
            summary["results"][mod] = count

    path = os.path.join(output_dir, "summary.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    log("INFO", f"汇总已生成: {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="LLM Guard Bench 统一评测入口")
    parser.add_argument("level", nargs="?", default="core", choices=["smoke", "core", "full"], help="评测级别")
    parser.add_argument("--model", default="qwen-turbo", help="模型名称")
    parser.add_argument("--output", default="data/results", help="输出目录根路径")
    parser.add_argument("--only", default="", help="仅执行指定模块（逗号分隔）")
    parser.add_argument("--skip", default="", help="跳过指定模块（逗号分隔）")
    parser.add_argument("--max-samples", type=int, default=0, help="限制每个评测的样本数")
    args = parser.parse_args()

    # 设置全局状态
    global _ONLY, _SKIP
    _ONLY = args.only
    _SKIP = args.skip

    level = args.level
    model = args.model
    output_dir = os.path.join(args.output, level)
    config = LEVEL_CONFIG[level]

    print(f"\n{'='*60}")
    print(f"  LLM Guard Bench - 统一评测入口")
    print(f"{'='*60}")
    print(f"  级别: {level} ({config['desc']})")
    print(f"  模型: {model}")
    print(f"  输出: {output_dir}")
    if args.only:
        print(f"  仅执行: {args.only}")
    if args.skip:
        print(f"  跳过: {args.skip}")
    print(f"{'='*60}\n")

    os.makedirs(output_dir, exist_ok=True)

    max_samples = config.get("max_samples", {})
    if args.max_samples > 0:
        max_samples = {k: args.max_samples for k in max_samples}

    alignment_samples = config.get("alignment_samples", 10)

    tasks = [
        ("单元测试", lambda: run_tests()),
        ("安全评测", lambda: run_safety(level, model, output_dir, max_samples)),
        ("能力评测", lambda: run_capability(level, model, output_dir, max_samples)),
        ("性能基准", lambda: run_performance(level, model, output_dir)),
        ("Agent评测", lambda: run_agent(level, model, output_dir)),
        ("体验评测", lambda: run_experience(level, model, output_dir)),
        ("AI测试工具", lambda: run_aitest(level, model, output_dir)),
        ("Judge对齐", lambda: run_alignment(level, model, output_dir, alignment_samples)),
    ]

    failed = []
    for name, func in tasks:
        log("INFO", f"===== {name} =====")
        if not func():
            failed.append(name)
            log("ERROR", f"{name} 失败")
            if level == "smoke":
                log("ERROR", "Smoke模式失败，停止执行")
                break

    save_summary(output_dir, level, model)

    print(f"\n{'='*60}")
    if failed:
        print(f"  ⚠️  评测完成，但以下模块失败: {', '.join(failed)}")
    else:
        print(f"  ✅ 所有评测完成")
    print(f"  报告目录: {output_dir}")
    print(f"{'='*60}\n")

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
