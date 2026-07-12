"""评测模块 - 七大维度评测实现.

子模块：
    - capability:  能力基准评测 (MMLU/C-Eval/GSM8K/HumanEval + 自建集)
    - safety:      对抗性安全评测 (3层攻击 + 双层评判)
    - performance: 推理性能基准测试 (27场景压测)
    - multimodal:  多模态评测 (图文理解 + 幻觉检测)
    - agent:       Agent评测 (工具调用 + 多步任务 + Code Agent)
    - experience:  应用层体验测试 (端到端 + A/B测试)
"""

__all__: list[str] = []
