"""人工标注 + Judge对齐模块.

对应 Phase 6：
    - 人工标注工具（对模型输出打标签）
    - Judge对齐分析（LLM-as-Judge vs 人工判定的一致性）
    - 核心指标：Accuracy / Precision / Recall / F1 / Cohen's Kappa / 混淆矩阵

目的：验证LLM评判器是否可靠，找出Judge盲点。
"""
from __future__ import annotations
