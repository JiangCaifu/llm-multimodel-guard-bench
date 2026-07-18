"""Streamlit 评测看板 - Phase 1 Day 13-14.

功能：
    - 能力雷达图（多维度得分对比）
    - 分领域得分柱状图
    - 评测结果详情表
    - 模型对比

启动方式：
    streamlit run src/llm_guard_bench/visualization/dashboard.py
"""
from __future__ import annotations

import json
import os
import sys

# 确保 src 在路径中
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "..", "src"))

import streamlit as st


def load_results(results_dir: str) -> dict:
    """加载评测结果."""
    results = {}
    summary_path = os.path.join(results_dir, "summary.json")

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            results["summary"] = json.load(f)

    # 加载各模型各数据集的详细结果
    for root, dirs, files in os.walk(results_dir):
        for fname in files:
            if fname.endswith(".json") and fname != "summary.json":
                fpath = os.path.join(root, fname)
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    model = data.get("model_name", "unknown")
                    dataset = data.get("dataset_name", "unknown")
                    if model not in results:
                        results[model] = {}
                    results[model][dataset] = data

    return results


def render_overview(summary: dict) -> None:
    """渲染总览卡片."""
    st.subheader("评测总览")

    models = summary.get("models", {})
    cols = st.columns(len(models)) if models else st.columns(1)

    for i, (model_name, datasets) in enumerate(models.items()):
        with cols[i % len(cols)]:
            total_correct = sum(d.get("correct_samples", 0) for d in datasets.values())
            total_samples = sum(d.get("total_samples", 0) for d in datasets.values())
            overall_acc = total_correct / total_samples if total_samples > 0 else 0

            st.metric(
                label=f"模型: {model_name}",
                value=f"{overall_acc:.1%}",
                delta=f"{total_correct}/{total_samples} 正确",
            )


def render_radar_chart(models_data: dict) -> None:
    """渲染能力雷达图."""
    st.subheader("能力雷达图")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("请安装 plotly: pip install plotly")
        return

    fig = go.Figure()

    for model_name, datasets in models_data.items():
        if not isinstance(datasets, dict):
            continue
        categories = []
        values = []

        for ds_name, ds_result in datasets.items():
            if isinstance(ds_result, dict) and "accuracy" in ds_result:
                categories.append(ds_name)
                values.append(ds_result["accuracy"])

        if categories:
            # 闭合雷达图
            categories_closed = categories + [categories[0]]
            values_closed = values + [values[0]]

            fig.add_trace(go.Scatterpolar(
                r=values_closed,
                theta=categories_closed,
                fill="toself",
                name=model_name,
                opacity=0.6,
            ))

    if fig.data:
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("暂无评测数据，请先运行评测任务")


def render_domain_bar_chart(model_data: dict) -> None:
    """渲染分领域得分柱状图."""
    st.subheader("分领域得分分布")

    try:
        import plotly.graph_objects as go
    except ImportError:
        st.warning("请安装 plotly: pip install plotly")
        return

    for model_name, datasets in model_data.items():
        if not isinstance(datasets, dict):
            continue

        for ds_name, ds_result in datasets.items():
            if not isinstance(ds_result, dict) or "results" not in ds_result:
                continue

            results = ds_result["results"]
            # 按领域分组
            domain_map: dict = {}
            for r in results:
                sid = r.get("sample_id", "")
                if sid.startswith("fin_"):
                    domain = "金融"
                elif sid.startswith("med_"):
                    domain = "医疗"
                else:
                    domain = "通用"
                if domain not in domain_map:
                    domain_map[domain] = {"total": 0, "correct": 0}
                domain_map[domain]["total"] += 1
                if r.get("is_correct"):
                    domain_map[domain]["correct"] += 1

            if domain_map:
                domains = list(domain_map.keys())
                accuracies = [
                    domain_map[d]["correct"] / domain_map[d]["total"]
                    if domain_map[d]["total"] > 0 else 0
                    for d in domains
                ]

                fig = go.Figure(data=[
                    go.Bar(
                        x=domains,
                        y=accuracies,
                        text=[f"{a:.1%}" for a in accuracies],
                        textposition="auto",
                    )
                ])
                fig.update_layout(
                    title=f"{model_name} - {ds_name}",
                    yaxis=dict(range=[0, 1], tickformat=".0%"),
                    height=400,
                )
                st.plotly_chart(fig, use_container_width=True)


def render_detail_table(model_data: dict) -> None:
    """渲染评测结果详情表."""
    st.subheader("评测结果详情")

    for model_name, datasets in model_data.items():
        if not isinstance(datasets, dict):
            continue

        for ds_name, ds_result in datasets.items():
            if not isinstance(ds_result, dict) or "results" not in ds_result:
                continue

            results = ds_result["results"]
            st.markdown(f"**{model_name} / {ds_name}** ({len(results)} 条)")

            # 只显示关键字段
            display_rows = []
            for r in results:
                display_rows.append({
                    "ID": r.get("sample_id", ""),
                    "问题": r.get("question", "")[:50] + "...",
                    "模型回答": r.get("model_answer", "")[:30],
                    "正确答案": r.get("correct_answer", ""),
                    "正确": "✓" if r.get("is_correct") else "✗",
                    "延迟(ms)": f"{r.get('latency_ms', 0):.0f}",
                })

            if display_rows:
                st.dataframe(display_rows, use_container_width=True, height=200)


def main() -> None:
    """主入口."""
    st.set_page_config(
        page_title="大模型评测看板",
        page_icon="📊",
        layout="wide",
    )

    st.title("大模型全链路评测平台 - 能力评测看板")
    st.caption("Phase 1: 能力基准评测 | 安全 | 性能 | 多模态 | Agent | AI测试 | 体验")

    # 选择结果目录
    default_dir = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "results", "capability")
    results_dir = st.text_input("评测结果目录", value=default_dir)

    if not os.path.exists(results_dir):
        st.warning(f"结果目录不存在: {results_dir}")
        st.info("请先运行评测: `python -m llm_guard_bench.cli run -c configs/evaluation/capability.yaml`")
        return

    results = load_results(results_dir)

    # 侧边栏
    st.sidebar.header("导航")
    page = st.sidebar.radio(
        "选择页面",
        ["总览", "能力雷达图", "分领域得分", "结果详情"],
    )

    if page == "总览":
        if "summary" in results:
            render_overview(results["summary"])
        else:
            st.info("未找到 summary.json")

    elif page == "能力雷达图":
        models_data = {k: v for k, v in results.items() if k != "summary"}
        if models_data:
            render_radar_chart(models_data)
        else:
            st.info("暂无评测数据")

    elif page == "分领域得分":
        models_data = {k: v for k, v in results.items() if k != "summary"}
        if models_data:
            render_domain_bar_chart(models_data)
        else:
            st.info("暂无评测数据")

    elif page == "结果详情":
        models_data = {k: v for k, v in results.items() if k != "summary"}
        if models_data:
            render_detail_table(models_data)
        else:
            st.info("暂无评测数据")


if __name__ == "__main__":
    main()
