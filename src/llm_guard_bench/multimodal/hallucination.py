"""多模态幻觉检测模块（核心亮点）.

评测流程：
    1. 让模型描述图片
    2. 用另一个 VLM 判断描述中哪些内容图片里根本没有
    3. 计算幻觉率 = 包含幻觉内容的描述数 / 总描述数

这是大模型多模态的通病，准确率只看"答对了没"，
幻觉率看"有没有瞎编"，面试官一听就知道你懂行。
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

from .adapter import MultimodalAdapter


HALLUCINATION_JUDGE_PROMPT = """你是一个专业的多模态幻觉检测专家。

任务：判断模型对图片的描述中是否存在幻觉（描述了图片中根本不存在的内容）。

图片描述：
{description}

请逐一检查以下每个声明是否能在图片中找到依据：
{claims}

对每个声明，给出判定：
- SUPPORTED: 图片中有明确依据
- HALLUCINATED: 图片中没有依据，属于幻觉
- UNVERIFIABLE: 无法判断

请以 JSON 格式输出：
{{
    "claims_analysis": [
        {{"claim": "...", "verdict": "SUPPORTED/HALLUCINATED/UNVERIFIABLE", "reason": "..."}}
    ],
    "hallucination_count": 0,
    "total_claims": 0,
    "hallucination_rate": 0.0
}}
"""


@dataclass
class HallucinationClaim:
    """单个声明."""

    claim: str
    verdict: str = "UNVERIFIABLE"  # SUPPORTED / HALLUCINATED / UNVERIFIABLE
    reason: str = ""


@dataclass
class HallucinationResult:
    """单个样本的幻觉检测结果."""

    sample_id: str
    image_path: str
    model_description: str
    claims: List[HallucinationClaim] = field(default_factory=list)
    hallucination_count: int = 0
    total_claims: int = 0
    hallucination_rate: float = 0.0
    has_hallucination: bool = False


@dataclass
class HallucinationReport:
    """幻觉检测报告."""

    model_name: str
    total_samples: int = 0
    hallucinated_samples: int = 0
    overall_hallucination_rate: float = 0.0
    results: List[HallucinationResult] = field(default_factory=list)
    category_stats: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "total_samples": self.total_samples,
            "hallucinated_samples": self.hallucinated_samples,
            "overall_hallucination_rate": f"{self.overall_hallucination_rate:.1%}",
            "results": [
                {
                    "sample_id": r.sample_id,
                    "description_preview": r.model_description[:100],
                    "hallucination_count": r.hallucination_count,
                    "total_claims": r.total_claims,
                    "hallucination_rate": f"{r.hallucination_rate:.1%}",
                    "has_hallucination": r.has_hallucination,
                    "claims": [
                        {"claim": c.claim, "verdict": c.verdict, "reason": c.reason}
                        for c in r.claims
                    ],
                }
                for r in self.results
            ],
        }


@dataclass
class HallucinationSample:
    """幻觉检测评测样本."""

    sample_id: str
    image_path: str
    description_prompt: str = "请详细描述这张图片的内容。"
    category: str = "general"


class HallucinationDetector:
    """多模态幻觉检测器."""

    def __init__(
        self,
        model_adapter: MultimodalAdapter,
        judge_adapter: Optional[MultimodalAdapter] = None,
    ) -> None:
        """初始化.

        Args:
            model_adapter: 被评测的多模态模型
            judge_adapter: 评判用的多模态模型（不传则用同一模型）
        """
        self._model = model_adapter
        self._judge = judge_adapter or model_adapter

    def detect_single(
        self,
        sample: HallucinationSample,
    ) -> HallucinationResult:
        """对单个样本进行幻觉检测.

        Args:
            sample: 幻觉检测样本

        Returns:
            幻觉检测结果
        """
        # Step 1: 让模型描述图片
        desc_result = self._model.describe_image(
            image_path=sample.image_path,
            prompt=sample.description_prompt,
            max_tokens=512,
        )

        description = desc_result.text if desc_result.success else f"[ERROR: {desc_result.error}]"

        # Step 2: 提取描述中的声明
        claims = self._extract_claims(description)

        if not claims:
            return HallucinationResult(
                sample_id=sample.sample_id,
                image_path=sample.image_path,
                model_description=description,
                claims=[],
                hallucination_count=0,
                total_claims=0,
                hallucination_rate=0.0,
                has_hallucination=False,
            )

        # Step 3: 用评判模型检查每个声明
        claims_text = "\n".join(f"{i+1}. {c}" for i, c in enumerate(claims))
        judge_prompt = HALLUCINATION_JUDGE_PROMPT.format(
            description=description,
            claims=claims_text,
        )

        judge_result = self._judge.chat_with_images(
            text=judge_prompt,
            image_paths=[sample.image_path],
            max_tokens=1024,
            temperature=0.0,
        )

        # Step 4: 解析评判结果
        hallucination_claims = self._parse_judge_response(
            judge_result.text, claims
        )

        h_count = sum(1 for c in hallucination_claims if c.verdict == "HALLUCINATED")
        total = len(hallucination_claims)
        h_rate = h_count / total if total > 0 else 0.0

        return HallucinationResult(
            sample_id=sample.sample_id,
            image_path=sample.image_path,
            model_description=description,
            claims=hallucination_claims,
            hallucination_count=h_count,
            total_claims=total,
            hallucination_rate=h_rate,
            has_hallucination=h_count > 0,
        )

    def detect_batch(
        self,
        samples: List[HallucinationSample],
    ) -> HallucinationReport:
        """批量幻觉检测."""
        report = HallucinationReport(model_name=self._model.model_name)

        for sample in samples:
            result = self.detect_single(sample)
            report.results.append(result)

        report.total_samples = len(report.results)
        report.hallucinated_samples = sum(1 for r in report.results if r.has_hallucination)
        report.overall_hallucination_rate = (
            report.hallucinated_samples / report.total_samples
            if report.total_samples > 0
            else 0.0
        )

        return report

    @staticmethod
    def _extract_claims(description: str) -> List[str]:
        """从描述中提取声明.

        简单实现：按句号/换行分割，每条作为一个声明。
        """
        # 按句号、问号、感叹号、换行分割
        sentences = re.split(r'[。！？\n]', description)
        # 过滤空句和过短的句
        claims = [s.strip() for s in sentences if len(s.strip()) > 5]
        # 最多取10条
        return claims[:10]

    @staticmethod
    def _parse_judge_response(
        response: str,
        original_claims: List[str],
    ) -> List[HallucinationClaim]:
        """解析评判模型的回复."""
        results: List[HallucinationClaim] = []

        # 尝试提取 JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group())
                for item in data.get("claims_analysis", []):
                    results.append(HallucinationClaim(
                        claim=item.get("claim", ""),
                        verdict=item.get("verdict", "UNVERIFIABLE"),
                        reason=item.get("reason", ""),
                    ))
                return results
            except json.JSONDecodeError:
                pass

        # JSON 解析失败，用关键词匹配
        for claim_text in original_claims:
            claim_lower = claim_text.lower()
            if "hallucinated" in response.lower() and claim_lower in response.lower():
                verdict = "HALLUCINATED"
            elif "supported" in response.lower() and claim_lower in response.lower():
                verdict = "SUPPORTED"
            else:
                verdict = "UNVERIFIABLE"

            results.append(HallucinationClaim(
                claim=claim_text,
                verdict=verdict,
            ))

        return results

    def save_report(self, report: HallucinationReport, output_dir: str) -> str:
        """保存报告."""
        os.makedirs(output_dir, exist_ok=True)
        path = os.path.join(output_dir, f"hallucination_{report.model_name}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2, ensure_ascii=False)
        return path

    def print_report(self, report: HallucinationReport) -> None:
        """打印报告."""
        try:
            from rich.console import Console
            from rich.table import Table

            console = Console()
            console.print(f"\n[bold]幻觉检测报告 - {report.model_name}[/bold]")
            console.print(f"  总样本: {report.total_samples}")
            console.print(f"  幻觉样本: [red]{report.hallucinated_samples}[/red]")
            console.print(f"  幻觉率: [red]{report.overall_hallucination_rate:.1%}[/red]")

            if report.results:
                table = Table(title="幻觉详情")
                table.add_column("样本ID", style="cyan")
                table.add_column("声明数", justify="right")
                table.add_column("幻觉数", justify="right", style="red")
                table.add_column("幻觉率", justify="right")

                for r in report.results:
                    color = "red" if r.has_hallucination else "green"
                    table.add_row(
                        r.sample_id,
                        str(r.total_claims),
                        str(r.hallucination_count),
                        f"[{color}]{r.hallucination_rate:.1%}[/{color}]",
                    )

                console.print(table)

                # 显示具体幻觉声明
                for r in report.results:
                    if r.has_hallucination:
                        console.print(f"\n[yellow]样本 {r.sample_id} 的幻觉声明:[/yellow]")
                        for c in r.claims:
                            if c.verdict == "HALLUCINATED":
                                console.print(f"  [red]✗[/red] {c.claim}")
                            elif c.verdict == "SUPPORTED":
                                console.print(f"  [green]✓[/green] {c.claim}")

        except ImportError:
            print(f"\n幻觉检测报告 - {report.model_name}")
            print(f"  总样本: {report.total_samples}, 幻觉: {report.hallucinated_samples}, 幻觉率: {report.overall_hallucination_rate:.1%}")
