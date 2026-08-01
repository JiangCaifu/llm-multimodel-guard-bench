"""性能基准测试 Streamlit 看板.

展示内容：
    - 场景压测结果表格
    - 延迟分布热力图（并发×输入长度）
    - 吞吐量对比图
    - 性能拐点分析
    - 部署建议
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import streamlit as st


def load_performance_reports(results_dir: str = "./data/results/performance") -> List[Dict[str, Any]]:
    """加载所有性能报告."""
    reports = []
    results_path = Path(results_dir)
    if not results_path.exists():
        return reports

    for f in sorted(results_path.glob("performance_*.json")):
        with open(f, "r", encoding="utf-8") as fp:
            reports.append(json.load(fp))

    return reports


def render_dashboard() -> None:
    """渲染性能看板."""
    st.set_page_config(page_title="性能基准测试看板", layout="wide")
    st.title("性能基准测试看板")

    # 加载数据
    reports = load_performance_reports()
    if not reports:
        st.warning("暂无性能测试数据。请先运行: `guard-bench perf --quick`")
        return

    # 选择模型
    model_names = [r["model_name"] for r in reports]
    selected_idx = st.selectbox("选择模型", range(len(model_names)), format_func=lambda i: model_names[i])
    report = reports[selected_idx]

    # === 总览 ===
    st.header("总览")

    col1, col2, col3 = st.columns(3)
    scenarios = report.get("scenarios", [])

    if scenarios:
        avg_e2e = sum(s["avg_e2e_ms"] for s in scenarios) / len(scenarios)
        avg_tps = sum(s["avg_throughput_tps"] for s in scenarios) / len(scenarios)
        avg_success = sum(float(s["success_rate"].rstrip("%")) for s in scenarios) / len(scenarios)

        col1.metric("平均 E2E 延迟", f"{avg_e2e:.0f} ms")
        col2.metric("平均吞吐量", f"{avg_tps:.1f} t/s")
        col3.metric("平均成功率", f"{avg_success:.1f}%")

    # === 场景结果表 ===
    st.header("场景压测结果")

    if scenarios:
        st.dataframe(
            [{
                "并发": s["concurrency"],
                "输入(tokens)": s["input_length"],
                "输出(tokens)": s["output_length"],
                "成功率": s["success_rate"],
                "E2E均值(ms)": s["avg_e2e_ms"],
                "P95(ms)": s["p95_e2e_ms"],
                "TTFT(ms)": s["avg_ttft_ms"],
                "TPOT(ms)": s["avg_tpot_ms"],
                "吞吐(t/s)": s["avg_throughput_tps"],
            } for s in scenarios],
            use_container_width=True,
        )

    # === 延迟热力图 ===
    st.header("延迟分布")

    if scenarios:
        try:
            import pandas as pd
            import plotly.express as px

            df = pd.DataFrame(scenarios)
            df["concurrency"] = df["concurrency"].astype(str)
            df["input_length"] = df["input_length"].astype(str)

            # 延迟随并发变化
            fig = px.bar(
                df,
                x="concurrency",
                y="avg_e2e_ms",
                color="input_length",
                barmode="group",
                title="E2E 延迟 vs 并发度（按输入长度分组）",
                labels={"concurrency": "并发度", "avg_e2e_ms": "E2E 延迟 (ms)", "input_length": "输入长度"},
            )
            st.plotly_chart(fig, use_container_width=True)

            # 吞吐量图
            fig2 = px.bar(
                df,
                x="concurrency",
                y="avg_throughput_tps",
                color="input_length",
                barmode="group",
                title="吞吐量 vs 并发度（按输入长度分组）",
                labels={"concurrency": "并发度", "avg_throughput_tps": "吞吐量 (t/s)", "input_length": "输入长度"},
            )
            st.plotly_chart(fig2, use_container_width=True)

        except ImportError:
            st.info("安装 plotly 可查看交互图表: pip install plotly")

    # === 分析结论 ===
    st.header("分析结论")

    analysis = report.get("analysis", {})
    if analysis.get("performance_inflection"):
        st.warning(f"**性能拐点**: {analysis['performance_inflection']}")

    if analysis.get("bottleneck"):
        st.error(f"**瓶颈归因**: {analysis['bottleneck']}")

    if analysis.get("deployment_advice"):
        st.success(f"**部署建议**: {analysis['deployment_advice']}")


if __name__ == "__main__":
    render_dashboard()
