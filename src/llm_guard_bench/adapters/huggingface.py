"""HuggingFace Transformers 适配器.

支持本地加载开源模型进行评测。

使用方式：
    config = HFConfig.from_yaml("configs/models/qwen2-7b.yaml")
    adapter = HFAdapter(config)
    result = adapter.generate("你好")
"""
from __future__ import annotations

import time
from typing import Any, Dict, List, Optional, Union

from llm_guard_bench.adapters.base import (
    BaseModelAdapter,
    GenerationResult,
    ModelConfigProtocol,
)


class HFConfig:
    """HuggingFace 模型配置."""

    def __init__(self, config_dict: Dict[str, Any]) -> None:
        self.name = config_dict["name"]
        self.provider = config_dict["provider"]
        self.model_name = config_dict["model_name"]
        self.temperature = config_dict.get("temperature", 0.0)
        self.max_tokens = config_dict.get("max_tokens", 1024)
        self.timeout = config_dict.get("timeout", 60)

        self.device = config_dict.get("device", "auto")
        self.torch_dtype = config_dict.get("torch_dtype", "bfloat16")
        self.quantization = config_dict.get("quantization", None)

        self.metadata = config_dict.get("metadata", {})

    def get_api_key(self) -> Optional[str]:
        return None

    def get_base_url(self) -> Optional[str]:
        return None

    @classmethod
    def from_yaml(cls, yaml_path: str) -> "HFConfig":
        import yaml

        with open(yaml_path, "r", encoding="utf-8") as f:
            return cls(yaml.safe_load(f))


@BaseModelAdapter.register("huggingface")
class HFAdapter(BaseModelAdapter):
    """HuggingFace Transformers 适配器."""

    def __init__(self, config: HFConfig) -> None:
        super().__init__(config)
        self._config = config
        self._model = None
        self._tokenizer = None
        self._init_model()

    def _init_model(self) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tokenizer = AutoTokenizer.from_pretrained(self._config.model_name)

        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        device_map = "auto" if self._config.device == "auto" else self._config.device

        torch_dtype = getattr(torch, self._config.torch_dtype)

        self._model = AutoModelForCausalLM.from_pretrained(
            self._config.model_name,
            torch_dtype=torch_dtype,
            device_map=device_map,
            load_in_4bit=self._config.quantization == "4bit",
            load_in_8bit=self._config.quantization == "8bit",
        )

        self._model.eval()

    def generate(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
        **kwargs: Any,
    ) -> GenerationResult:
        import torch

        start_time = time.time()

        try:
            effective_max_tokens, effective_temperature, _ = self._get_effective_params(
                max_tokens, temperature, timeout
            )

            inputs = self._tokenizer(prompt, return_tensors="pt").to(self._model.device)
            prompt_tokens = inputs["input_ids"].shape[1]

            with torch.no_grad():
                outputs = self._model.generate(
                    **inputs,
                    max_new_tokens=effective_max_tokens,
                    temperature=effective_temperature,
                    do_sample=effective_temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **kwargs,
                )

            latency_ms = self._record_latency(start_time)
            generated_text = self._tokenizer.decode(
                outputs[0][prompt_tokens:],
                skip_special_tokens=True,
            )

            completion_tokens = outputs[0].shape[0] - prompt_tokens

            return GenerationResult(
                text=generated_text.strip(),
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=prompt_tokens + completion_tokens,
                latency_ms=latency_ms,
                finish_reason="length" if completion_tokens >= effective_max_tokens else "eos",
            )

        except Exception as e:
            latency_ms = self._record_latency(start_time)
            return GenerationResult(
                text="",
                latency_ms=latency_ms,
                error=str(e),
            )

    def chat(
        self,
        messages: List[Dict[str, str]],
        **kwargs: Any,
    ) -> GenerationResult:
        start_time = time.time()

        try:
            if hasattr(self._tokenizer, "apply_chat_template"):
                prompt = self._tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                prompt = "\n".join(
                    f"{m['role']}: {m['content']}" for m in messages
                ) + "\nassistant:"

            return self.generate(prompt, **kwargs)

        except Exception as e:
            latency_ms = self._record_latency(start_time)
            return GenerationResult(
                text="",
                latency_ms=latency_ms,
                error=str(e),
            )

    def close(self) -> None:
        import torch

        if self._model is not None:
            del self._model
            torch.cuda.empty_cache()

    def get_model_info(self) -> Any:
        info = super().get_model_info()
        info.context_window = self._config.metadata.get("context_window")
        info.vendor = self._config.metadata.get("vendor")
        info.tags = self._config.metadata.get("tags", [])
        return info
