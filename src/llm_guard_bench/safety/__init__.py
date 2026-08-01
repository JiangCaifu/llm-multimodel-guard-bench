"""安全对抗评测模块.

对应 Phase 2 (Day 15-28):
    - Level 1: 经典越狱模板 + 编码绕过
    - Level 2: LLM语义变异 + 多轮对话诱导
    - Level 3: 自动化红队（迭代攻击）
    - 双层评判: 关键词匹配 + LLM-as-Judge
    - P0/P1/P2 分级告警
    - Bad Case 回归机制
"""
