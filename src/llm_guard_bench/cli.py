"""命令行入口 - 评测任务执行器.

用法:
    guard-bench run --config configs/evaluation/capability.yaml
    guard-bench run --config configs/evaluation/capability.yaml --model gpt-4
    guard-bench safety --level 1 --max-samples 10
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


def _cmd_perf(args: argparse.Namespace) -> int:
    """执行性能基准测试."""
    from llm_guard_bench.adapters.factory import build_adapter, load_model_config
    from llm_guard_bench.performance.benchmark import InputLength, OutputLength, PerformanceBenchmark

    model_name = args.model or "qwen-turbo"
    logger.info(f"启动性能基准测试: model={model_name}")

    config = load_model_config(model_name)
    adapter = build_adapter(config)

    benchmark = PerformanceBenchmark(
        adapter=adapter,
        concurrency_levels=args.concurrency or [1, 3, 5],
        requests_per_scenario=args.requests,
    )

    report = benchmark.run_all(quick=args.quick)
    benchmark.print_report(report)

    output_dir = args.output or "./data/results/performance"
    path = benchmark.save_report(report, output_dir)
    logger.info(f"性能报告已保存: {path}")

    return 0


def _cmd_multimodal(args: argparse.Namespace) -> int:
    """执行多模态评测."""
    from llm_guard_bench.adapters.factory import build_adapter, load_model_config
    from llm_guard_bench.multimodal.adapter import MultimodalAdapter
    from llm_guard_bench.multimodal.hallucination import HallucinationSample
    from llm_guard_bench.multimodal.runner import MultimodalRunner
    from llm_guard_bench.multimodal.vqa import VQASample

    model_name = args.model or "qwen-vl-plus"
    images_dir = args.images_dir or "./data/images"
    output_dir = args.output or "./data/results/multimodal"

    logger.info(f"启动多模态评测: model={model_name}")

    # 创建适配器
    config = load_model_config(model_name)
    base_adapter = build_adapter(config)
    mm_adapter = MultimodalAdapter(base_adapter)

    # 检查图片目录
    img_path = Path(images_dir)
    if not img_path.exists():
        logger.warning(f"图片目录不存在: {images_dir}，将创建示例数据")
        img_path.mkdir(parents=True, exist_ok=True)

    # 收集图片
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png", "*.webp"]:
        image_files.extend(img_path.glob(ext))

    if not image_files:
        logger.warning("未找到评测图片，跳过多模态评测")
        return 0

    logger.info(f"找到 {len(image_files)} 张图片")

    # 构建 VQA 样本
    vqa_samples = []
    hallucination_samples = []
    for i, img in enumerate(image_files):
        vqa_samples.append(VQASample(
            sample_id=f"vqa_{i:03d}",
            image_path=str(img),
            question="请描述这张图片的内容。",
            answer="",  # 无标准答案时用空字符串
            category="general",
        ))
        hallucination_samples.append(HallucinationSample(
            sample_id=f"hallu_{i:03d}",
            image_path=str(img),
            description_prompt="请详细描述这张图片的内容。",
        ))

    # 运行评测
    runner = MultimodalRunner(mm_adapter)

    if vqa_samples:
        logger.info("  运行 VQA 评测...")
        vqa_report = runner.run_vqa(vqa_samples)
        logger.info(f"  VQA 准确率: {vqa_report.get('accuracy', 'N/A')}")

    if hallucination_samples:
        logger.info("  运行幻觉检测...")
        h_report = runner.run_hallucination(hallucination_samples)
        logger.info(f"  幻觉率: {h_report.get('overall_hallucination_rate', 'N/A')}")

    # 综合报告
    full_report = runner.run_full(vqa_samples, hallucination_samples)
    runner.print_report(full_report)

    path = runner.save_report(full_report, output_dir)
    logger.info(f"多模态评测报告已保存: {path}")

    return 0


def _cmd_agent(args: argparse.Namespace) -> int:
    """执行Agent评测."""
    from llm_guard_bench.adapters.factory import build_adapter, load_model_config
    from llm_guard_bench.agent.runner import AgentRunner

    model_name = args.model or "qwen-turbo"
    task_type = args.task or "full"
    output_dir = args.output or "./data/results/agent"

    logger.info(f"启动Agent评测: model={model_name}, task={task_type}")

    config = load_model_config(model_name)
    adapter = build_adapter(config)
    runner = AgentRunner(adapter)

    if task_type == "tool":
        report_dict = runner.run_tool_call()
        logger.info(f"  工具选择准确率: {report_dict.get('tool_selection_accuracy', 'N/A')}")
        logger.info(f"  参数填充准确率: {report_dict.get('params_fill_accuracy', 'N/A')}")
    elif task_type == "multi_step":
        report_dict = runner.run_multi_step()
        logger.info(f"  规划评分: {report_dict.get('avg_planning_score', 'N/A')}")
        logger.info(f"  执行正确率: {report_dict.get('avg_execution_accuracy', 'N/A')}")
    elif task_type == "code":
        report_dict = runner.run_code_agent()
        logger.info(f"  代码生成通过率: {report_dict.get('code_generation', {}).get('pass_rate', 'N/A')}")
        logger.info(f"  调试修复率: {report_dict.get('debug_fix', {}).get('fix_rate', 'N/A')}")
    else:
        report = runner.run_full()
        runner.print_report(report)
        path = runner.save_report(report, output_dir)
        logger.info(f"Agent评测报告已保存: {path}")
        return 0

    if task_type != "full":
        logger.info("评测完成")

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


def _cmd_safety(args: argparse.Namespace) -> int:
    """执行安全评测任务."""
    from llm_guard_bench.adapters.factory import build_adapter, load_model_config
    from llm_guard_bench.safety.runner import SafetyRunner

    model_name = args.model or "qwen-turbo"
    max_samples = args.max_samples or None
    use_mutation = not args.no_mutation

    logger.info(f"启动安全评测: model={model_name}")

    # 创建模型适配器
    config = load_model_config(model_name)
    adapter = build_adapter(config)

    # 创建评判适配器（用同一模型）
    judge_adapter = adapter

    runner = SafetyRunner(
        model_adapter=adapter,
        judge_adapter=judge_adapter,
        use_semantic_mutation=use_mutation,
        mutation_count=3,
        max_samples=max_samples,
    )

    # 选择评测级别
    if args.level == "1":
        logger.info("  攻击级别: Level 1 (越狱 + 角色扮演 + 编码绕过)")
        report = runner.run_level1()
    elif args.level == "2":
        logger.info("  攻击级别: Level 2 (多轮诱导)")
        report = runner.run_level2_multi_turn()
    elif args.level == "3":
        logger.info("  攻击级别: Level 1 + Level 2 (全量)")
        report = runner.run_full()
    else:
        logger.info("  攻击级别: 全量")
        report = runner.run_full()

    # 保存报告
    output_dir = args.output or "./data/results/safety"
    output_path = runner.save_report(report, output_dir)
    logger.info(f"安全评测完成. 报告: {output_path}")

    # 打印报告
    runner.print_report(report)

    # 生成安全评分卡
    from llm_guard_bench.safety.scorecard import SafetyScorecardBuilder
    scorecard_builder = SafetyScorecardBuilder()
    scorecard = scorecard_builder.build(report)
    scorecard_builder.print_scorecard(scorecard)
    scorecard_path = scorecard_builder.save_scorecard(scorecard, output_dir)
    logger.info(f"评分卡已保存: {scorecard_path}")

    # P0 检查
    if report.p0_count > 0:
        logger.error(f"⚠ 发现 {report.p0_count} 个 P0 级安全风险！建议立即处理。")
        if args.strict:
            return 1

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

    # safety: 安全评测
    safety_parser = subparsers.add_parser("safety", help="执行安全对抗评测")
    safety_parser.add_argument("--model", "-m", default="qwen-turbo", help="被评测模型名")
    safety_parser.add_argument("--level", "-l", choices=["1", "2", "3"], default="3", help="攻击级别 (1/2/3)")
    safety_parser.add_argument("--max-samples", type=int, default=None, help="限制攻击样本数")
    safety_parser.add_argument("--no-mutation", action="store_true", help="禁用语义变异")
    safety_parser.add_argument("--output", "-o", default="./data/results/safety", help="结果输出目录")
    safety_parser.add_argument("--strict", action="store_true", help="P0风险时返回非零退出码")
    safety_parser.set_defaults(func=_cmd_safety)

    # perf: 性能基准测试
    perf_parser = subparsers.add_parser("perf", help="执行性能基准测试")
    perf_parser.add_argument("--model", "-m", default="qwen-turbo", help="被评测模型名")
    perf_parser.add_argument("--quick", "-q", action="store_true", help="快速模式（减少场景数和请求数）")
    perf_parser.add_argument("--concurrency", "-c", type=int, nargs="+", default=None, help="并发度列表，如 1 5 10")
    perf_parser.add_argument("--requests", type=int, default=10, help="每个场景的请求数")
    perf_parser.add_argument("--output", "-o", default="./data/results/performance", help="结果输出目录")
    perf_parser.set_defaults(func=_cmd_perf)

    # multimodal: 多模态评测
    mm_parser = subparsers.add_parser("multimodal", help="执行多模态评测")
    mm_parser.add_argument("--model", "-m", default="qwen-vl-plus", help="多模态模型名")
    mm_parser.add_argument("--images-dir", "-i", default="./data/images", help="评测图片目录")
    mm_parser.add_argument("--output", "-o", default="./data/results/multimodal", help="结果输出目录")
    mm_parser.set_defaults(func=_cmd_multimodal)

    # agent: Agent评测
    agent_parser = subparsers.add_parser("agent", help="执行Agent评测")
    agent_parser.add_argument("--model", "-m", default="qwen-turbo", help="被评测模型名")
    agent_parser.add_argument("--task", "-t", choices=["tool", "multi_step", "code", "full"], default="full", help="评测任务类型")
    agent_parser.add_argument("--output", "-o", default="./data/results/agent", help="结果输出目录")
    agent_parser.set_defaults(func=_cmd_agent)

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
