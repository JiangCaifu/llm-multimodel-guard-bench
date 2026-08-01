"""CLIP Score 图文匹配评测.

评测图片描述与图片内容的匹配程度。
CLIP Score = cos(CLIP_image_embedding, CLIP_text_embedding)

由于 CLIP 模型需要单独安装，本模块提供：
    1. 基于模型的 CLIP Score（需安装 transformers）
    2. 基于规则的基础匹配分（作为降级方案）
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .adapter import MultimodalAdapter


@dataclass
class CLIPScoreResult:
    """CLIP Score 评测结果."""

    sample_id: str
    image_path: str
    description: str
    clip_score: float = 0.0  # 0-1, 越高越匹配
    match_level: str = "unknown"  # excellent / good / fair / poor


@dataclass
class CLIPScoreReport:
    """CLIP Score 报告."""

    model_name: str
    total_samples: int = 0
    avg_clip_score: float = 0.0
    results: List[CLIPScoreResult] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "avg_clip_score": round(self.avg_clip_score, 3),
            "results": [
                {
                    "sample_id": r.sample_id,
                    "clip_score": round(r.clip_score, 3),
                    "match_level": r.match_level,
                    "description_preview": r.description[:80],
                }
                for r in self.results
            ],
        }


class CLIPScoreEvaluator:
    """CLIP Score 评测器."""

    def __init__(self, adapter: MultimodalAdapter) -> None:
        self._adapter = adapter
        self._clip_model = None

    def _try_load_clip(self) -> bool:
        """尝试加载 CLIP 模型."""
        try:
            import torch
            from transformers import CLIPModel, CLIPProcessor

            self._clip_model = {
                "model": CLIPModel.from_pretrained("openai/clip-vit-base-patch32"),
                "processor": CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32"),
            }
            return True
        except Exception:
            return False

    def compute_clip_score_model(
        self,
        image_path: str,
        text: str,
    ) -> float:
        """使用 CLIP 模型计算图文匹配分数.

        Args:
            image_path: 图片路径
            text: 文本描述

        Returns:
            CLIP Score (0-1)
        """
        if self._clip_model is None:
            if not self._try_load_clip():
                return self._compute_rule_based_score(image_path, text)

        try:
            from PIL import Image
            import torch

            model = self._clip_model["model"]
            processor = self._clip_model["processor"]

            image = Image.open(image_path).convert("RGB")
            inputs = processor(text=[text], images=[image], return_tensors="pt", padding=True)

            with torch.no_grad():
                outputs = model(**inputs)
                # 归一化
                image_embeds = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
                text_embeds = outputs.text_embeds / outputs.text_embeds.norm(dim=-1, keepdim=True)
                # 余弦相似度
                score = (image_embeds @ text_embeds.T).item()

            return max(0.0, min(1.0, (score + 1) / 2))  # 归一化到 0-1

        except Exception:
            return self._compute_rule_based_score(image_path, text)

    @staticmethod
    def _compute_rule_based_score(image_path: str, text: str) -> float:
        """基于规则的基础匹配分（降级方案）.

        检查描述中的关键词是否与图片文件名/路径相关。
        """
        # 基础分：有描述就给 0.5
        score = 0.5

        # 描述长度奖励（0-0.2）
        desc_len = len(text)
        if desc_len > 50:
            score += 0.1
        if desc_len > 100:
            score += 0.1

        # 文件名关键词匹配（0-0.2）
        filename = os.path.basename(image_path).lower()
        text_lower = text.lower()
        # 从文件名中提取有意义的词
        name_words = [w for w in filename.replace(".", " ").replace("_", " ").split() if len(w) > 2]
        matched = sum(1 for w in name_words if w in text_lower)
        if name_words:
            score += 0.2 * (matched / len(name_words))

        # 描述中包含视觉相关词汇（0-0.1）
        visual_words = ["颜色", "形状", "大小", "位置", "背景", "前景", "左", "右", "上", "下"]
        visual_count = sum(1 for w in visual_words if w in text)
        score += min(0.1, visual_count * 0.02)

        return min(1.0, score)

    @staticmethod
    def _score_to_level(score: float) -> str:
        """分数转匹配等级."""
        if score >= 0.8:
            return "excellent"
        if score >= 0.6:
            return "good"
        if score >= 0.4:
            return "fair"
        return "poor"

    def evaluate_single(
        self,
        sample_id: str,
        image_path: str,
        description: str,
        use_model: bool = True,
    ) -> CLIPScoreResult:
        """评测单个图文匹配."""
        if use_model and self._clip_model is not None:
            score = self.compute_clip_score_model(image_path, description)
        else:
            score = self._compute_rule_based_score(image_path, description)

        return CLIPScoreResult(
            sample_id=sample_id,
            image_path=image_path,
            description=description,
            clip_score=score,
            match_level=self._score_to_level(score),
        )

    def evaluate_batch(
        self,
        samples: List[Tuple[str, str, str]],  # (sample_id, image_path, description)
        use_model: bool = True,
    ) -> CLIPScoreReport:
        """批量评测图文匹配."""
        report = CLIPScoreReport(model_name=self._adapter.model_name)

        for sample_id, image_path, description in samples:
            result = self.evaluate_single(sample_id, image_path, description, use_model)
            report.results.append(result)

        report.total_samples = len(report.results)
        report.avg_clip_score = (
            sum(r.clip_score for r in report.results) / report.total_samples
            if report.total_samples > 0
            else 0.0
        )

        return report

    def save_report(self, report: CLIPScoreReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"clip_score_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path
