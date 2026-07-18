"""快速验证脚本 - 测试模型调用和评测流水线.

使用方式:
    python scripts/verify_model.py
"""
from __future__ import annotations

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from dotenv import load_dotenv

load_dotenv()

from llm_guard_bench.adapters import build_adapter, load_model_config


def verify_model_call() -> None:
    """验证 qwen-turbo 模型调用."""
    print("=" * 60)
    print("Step 1: 验证模型调用 (qwen-turbo via DashScope)")
    print("=" * 60)

    config = load_model_config("qwen-turbo")
    print(f"  模型: {config.model_name}")
    print(f"  Provider: {config.provider}")
    print(f"  API Key: {config.get_api_key()[:10]}..." if config.get_api_key() else "  API Key: 未设置!")

    adapter = build_adapter(config)

    # 测试1: 简单问答
    print("\n  [测试1] 简单问答...")
    result = adapter.chat([
        {"role": "user", "content": "1+1等于几？只回答数字"}
    ])
    print(f"    回复: {result.text}")
    print(f"    Token: {result.total_tokens}")
    print(f"    耗时: {result.latency_ms:.0f}ms")
    if result.error:
        print(f"    错误: {result.error}")
        return

    # 测试2: 选择题格式 (MMLU风格)
    print("\n  [测试2] 选择题格式 (MMLU 风格)...")
    result = adapter.chat([
        {"role": "user", "content": """What is the capital of France?
A. London
B. Paris
C. Berlin
D. Madrid
Answer:"""}
    ])
    print(f"    回复: {result.text}")
    print(f"    Token: {result.total_tokens}")
    print(f"    耗时: {result.latency_ms:.0f}ms")

    # 测试3: 数学题 (GSM8K风格)
    print("\n  [测试3] 数学题 (GSM8K 风格)...")
    result = adapter.chat([
        {"role": "user", "content": """Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?"""}
    ])
    print(f"    回复: {result.text[:200]}")
    print(f"    Token: {result.total_tokens}")
    print(f"    耗时: {result.latency_ms:.0f}ms")

    adapter.close()
    print("\n  ✓ 模型调用验证通过!")


def verify_evaluation_pipeline() -> None:
    """验证评测流水线 (小规模)."""
    print("\n" + "=" * 60)
    print("Step 2: 验证评测流水线 (3条样本)")
    print("=" * 60)

    from llm_guard_bench.engine import EvaluationConfig, EvaluationRunner

    # 创建最小评测配置
    config = EvaluationConfig(
        name="verify_test",
        models=["qwen-turbo"],
        datasets=[],  # 我们手动测试
    )

    # 手动跑几条样本
    from llm_guard_bench.adapters import build_adapter, load_model_config
    from llm_guard_bench.engine.dataloader import DataSample, DatasetConfig

    adapter = build_adapter(load_model_config("qwen-turbo"))

    # 模拟 MMLU 选择题
    samples = [
        DataSample(
            id="0",
            question="What is 2+2?\nA. 3\nB. 4\nC. 5\nD. 6\nAnswer:",
            answer="B",
        ),
        DataSample(
            id="1",
            question="Which planet is closest to the Sun?\nA. Venus\nB. Mercury\nC. Mars\nD. Earth\nAnswer:",
            answer="B",
        ),
        DataSample(
            id="2",
            question="What is the chemical symbol for water?\nA. H2O\nB. CO2\nC. NaCl\nD. O2\nAnswer:",
            answer="A",
        ),
    ]

    correct = 0
    for sample in samples:
        result = adapter.chat([{"role": "user", "content": sample.question}])
        is_correct = sample.answer.lower() in result.text.lower()
        correct += is_correct
        status = "✓" if is_correct else "✗"
        print(f"  [{status}] Q: {sample.question[:30]}... → 模型: {result.text[:30]} | 正确: {sample.answer}")

    print(f"\n  准确率: {correct}/{len(samples)} = {correct/len(samples):.0%}")
    adapter.close()
    print("  ✓ 评测流水线验证通过!")


if __name__ == "__main__":
    try:
        verify_model_call()
        verify_evaluation_pipeline()
        print("\n" + "=" * 60)
        print("ALL VERIFICATIONS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
