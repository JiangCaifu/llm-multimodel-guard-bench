"""模型接入层 - 统一 Adapter 接口.

支持多种模型后端：
    - OpenAI API (GPT-4, GPT-3.5 等)
    - HuggingFace Transformers
    - vLLM 推理服务
    - 多模态模型扩展接口

将在 Phase 1 Day 3-4 实现具体接口:
    - BaseModelAdapter: 统一适配器基类
    - OpenAIAdapter / HFAdapter / VLLMAdapter
    - build_adapter(config): 工厂函数
"""

__all__: list[str] = []
