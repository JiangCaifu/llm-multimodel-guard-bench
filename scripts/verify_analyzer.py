"""验证深度分析模块.

使用方式:
    python scripts/verify_analyzer.py
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from llm_guard_bench.evaluation.analyzer import DeepAnalyzer


def test_analyzer_with_mock_data() -> None:
    """用模拟数据测试分析器."""
    print("=" * 60)
    print("验证深度分析模块")
    print("=" * 60)

    # 模拟评测结果（金融+医疗混合）
    mock_result = {
        "dataset_name": "finance_50",
        "model_name": "qwen-turbo",
        "total_samples": 50,
        "correct_samples": 38,
        "accuracy": 0.76,
        "results": [
            {
                "sample_id": "fin_001",
                "question": "亏损金额是多少？",
                "model_answer": "B",
                "correct_answer": "B",
                "is_correct": True,
                "latency_ms": 350,
                "difficulty": "easy",
            },
            {
                "sample_id": "fin_002",
                "question": "以下哪种属于固定收益？",
                "model_answer": "B",
                "correct_answer": "B",
                "is_correct": True,
                "latency_ms": 280,
                "difficulty": "easy",
            },
            {
                "sample_id": "fin_022",
                "question": "单一客户贷款比例不超过？",
                "model_answer": "A",
                "correct_answer": "B",
                "is_correct": False,
                "latency_ms": 420,
                "difficulty": "hard",
            },
            {
                "sample_id": "med_001",
                "question": "正常成人静息心率？",
                "model_answer": "B",
                "correct_answer": "B",
                "is_correct": True,
                "latency_ms": 310,
                "difficulty": "easy",
            },
            {
                "sample_id": "med_035",
                "question": "二甲双胍禁忌症？",
                "model_answer": "A",
                "correct_answer": "B",
                "is_correct": False,
                "latency_ms": 450,
                "difficulty": "medium",
            },
            {
                "sample_id": "mmlu_001",
                "question": "What is the capital of France?",
                "model_answer": "Paris",
                "correct_answer": "Paris",
                "is_correct": True,
                "latency_ms": 260,
                "difficulty": "easy",
            },
        ],
    }

    analyzer = DeepAnalyzer()
    report = analyzer.analyze_result_data(mock_result)

    # 打印报告
    analyzer.print_report(report)

    # 导出报告
    output_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results", "analysis")
    output_path = analyzer.export_report(report, output_dir)
    print(f"\n  报告已导出: {output_path}")

    # 验证结果
    assert report.overall_accuracy == 4 / 6
    assert len(report.domain_breakdowns) == 3  # 金融、医疗、通用
    print("\n  ✓ 深度分析模块验证通过!")


def test_model_comparison() -> None:
    """测试模型对比."""
    print("\n" + "=" * 60)
    print("验证模型对比")
    print("=" * 60)

    # 创建两个模型的模拟结果文件
    tmp_dir = os.path.join(os.path.dirname(__file__), "..", "data", "results", "tmp_compare")
    os.makedirs(tmp_dir, exist_ok=True)

    model_a_result = {
        "dataset_name": "finance_50",
        "model_name": "qwen-turbo",
        "total_samples": 50,
        "correct_samples": 38,
        "accuracy": 0.76,
        "results": [],
    }
    model_b_result = {
        "dataset_name": "finance_50",
        "model_name": "gpt-4",
        "total_samples": 50,
        "correct_samples": 44,
        "accuracy": 0.88,
        "results": [],
    }

    path_a = os.path.join(tmp_dir, "qwen-turbo_finance_50.json")
    path_b = os.path.join(tmp_dir, "gpt-4_finance_50.json")

    with open(path_a, "w", encoding="utf-8") as f:
        json.dump(model_a_result, f, ensure_ascii=False, indent=2)
    with open(path_b, "w", encoding="utf-8") as f:
        json.dump(model_b_result, f, ensure_ascii=False, indent=2)

    analyzer = DeepAnalyzer()
    comparisons = analyzer.compare_models({
        "qwen-turbo": path_a,
        "gpt-4": path_b,
    })

    for comp in comparisons:
        print(f"\n  数据集: {comp.dataset_name}")
        for model, stats in comp.models.items():
            print(f"    {model}: {stats['accuracy']:.1%} ({stats['correct_samples']}/{stats['total_samples']})")
        print(f"    最优模型: [green]{comp.winner}[/green]")

    assert len(comparisons) == 1
    assert comparisons[0].winner == "gpt-4"
    print("\n  ✓ 模型对比验证通过!")

    # 清理
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    try:
        test_analyzer_with_mock_data()
        test_model_comparison()
        print("\n" + "=" * 60)
        print("ALL ANALYZER VERIFICATIONS PASSED!")
        print("=" * 60)
    except Exception as e:
        print(f"\n验证失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
