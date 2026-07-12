"""命令行入口 - 评测任务执行器.

用法:
    guard-bench run --config configs/evaluation/capability.yaml
    guard-bench run --config configs/evaluation/capability.yaml --model gpt-4
    guard-bench list-datasets
    guard-bench list-models
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from loguru import logger

from llm_guard_bench import __version__
from llm_guard_bench.constants import CONFIGS_DIR


def _cmd_run(args: argparse.Namespace) -> int:
    """执行评测任务."""
    from llm_guard_bench.engine import EvaluationConfig, EvaluationRunner

    config_path = Path(args.config)
    if not config_path.exists():
        logger.error(f"配置文件不存在: {config_path}")
        return 1

    config = EvaluationConfig.from_yaml(config_path)
    # CLI 覆盖: --model 优先
    if args.model:
        config.models = [args.model]

    logger.info(f"启动评测任务: {config.name}")
    logger.info(f"  模型: {config.models}")
    logger.info(f"  数据集: {config.datasets}")

    runner = EvaluationRunner(config)
    result = runner.run()

    logger.info(f"评测完成. 结果输出: {result.output_dir}")
    return 0


def _cmd_list_datasets(_args: argparse.Namespace) -> int:
    """列出可用的数据集配置."""
    datasets_dir = CONFIGS_DIR / "datasets"
    if not datasets_dir.exists():
        print("未找到数据集配置目录")
        return 1

    print("可用数据集:")
    for f in sorted(datasets_dir.glob("*.yaml")):
        print(f"  - {f.stem}")
    return 0


def _cmd_list_models(_args: argparse.Namespace) -> int:
    """列出可用的模型配置."""
    models_dir = CONFIGS_DIR / "models"
    if not models_dir.exists():
        print("未找到模型配置目录")
        return 1

    print("可用模型:")
    for f in sorted(models_dir.glob("*.yaml")):
        print(f"  - {f.stem}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """构建 CLI 参数解析器."""
    parser = argparse.ArgumentParser(
        prog="guard-bench",
        description="大模型全链路评测平台 - 安全/能力/性能/多模态/Agent/AI测试/应用体验",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # run: 执行评测
    run_parser = subparsers.add_parser("run", help="执行评测任务")
    run_parser.add_argument(
        "--config", "-c", required=True, help="评测任务配置文件路径 (YAML)"
    )
    run_parser.add_argument("--model", "-m", help="覆盖配置中的模型 (单个模型名)")
    run_parser.add_argument("--output", "-o", help="结果输出目录")
    run_parser.set_defaults(func=_cmd_run)

    # list-datasets: 列出数据集
    list_ds_parser = subparsers.add_parser("list-datasets", help="列出可用数据集")
    list_ds_parser.set_defaults(func=_cmd_list_datasets)

    # list-models: 列出模型
    list_model_parser = subparsers.add_parser("list-models", help="列出可用模型")
    list_model_parser.set_defaults(func=_cmd_list_models)

    return parser


def main() -> int:
    """CLI 主入口."""
    parser = build_parser()
    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help()
        return 1

    # 配置日志
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<level>{level: <8}</level> | {message}")

    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
