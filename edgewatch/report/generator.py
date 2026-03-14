"""Report generation primitives for single runs and model comparisons."""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional


@dataclass
class ComparisonMetrics:
    tokens_per_sec_diff: float
    tokens_per_sec_ratio: float
    ttft_ms_diff: float
    gpu_temp_diff: Optional[float]
    tj_temp_diff: Optional[float]
    power_diff: Optional[float]
    speedup_pct: float


@dataclass
class RunReport:
    model: str
    run_id: str
    timestamp: datetime
    duration_sec: float
    avg_tokens_per_sec: float
    avg_ttft_ms: float
    peak_gpu_temp: float
    peak_tj_temp: float
    throttle_events: int
    is_stable: bool


@dataclass
class ModelComparison:
    model_reports: List[RunReport]
    comparison_summary: Dict[str, ComparisonMetrics]
    overall_best_model: str
    overall_worst_model: str
    generated_at: datetime
    total_runs: int
    export_format: str
    tool_version: str


class ReportGenerator:
    """Formats simple human-readable reports."""

    def generate_single_run_report(self, report: RunReport) -> str:
        return "\n".join(
            [
                f"Model: {report.model}",
                f"Run ID: {report.run_id}",
                f"Time: {report.timestamp.isoformat()}",
                f"Duration: {report.duration_sec:.2f}s",
                f"Tokens/sec: {report.avg_tokens_per_sec:.2f}",
                f"TTFT: {report.avg_ttft_ms:.2f} ms",
                f"Peak GPU Temp: {report.peak_gpu_temp:.1f}C",
                f"Peak TJ Temp: {report.peak_tj_temp:.1f}C",
                f"Throttle Events: {report.throttle_events}",
                f"Stable: {'yes' if report.is_stable else 'no'}",
            ]
        )

    def summarize_models(self, reports: List[RunReport]) -> ModelComparison:
        if not reports:
            raise ValueError("No reports to summarize")

        best = max(reports, key=lambda r: r.avg_tokens_per_sec)
        worst = min(reports, key=lambda r: r.avg_tokens_per_sec)

        comparison_summary: Dict[str, ComparisonMetrics] = {}
        baseline = best.avg_tokens_per_sec if best.avg_tokens_per_sec > 0 else 1.0

        for rep in reports:
            diff = rep.avg_tokens_per_sec - baseline
            ratio = rep.avg_tokens_per_sec / baseline if baseline > 0 else 0.0
            comparison_summary[rep.model] = ComparisonMetrics(
                tokens_per_sec_diff=diff,
                tokens_per_sec_ratio=ratio,
                ttft_ms_diff=rep.avg_ttft_ms - best.avg_ttft_ms,
                gpu_temp_diff=rep.peak_gpu_temp - best.peak_gpu_temp,
                tj_temp_diff=rep.peak_tj_temp - best.peak_tj_temp,
                power_diff=None,
                speedup_pct=(ratio - 1.0) * 100.0,
            )

        return ModelComparison(
            model_reports=reports,
            comparison_summary=comparison_summary,
            overall_best_model=best.model,
            overall_worst_model=worst.model,
            generated_at=datetime.now(),
            total_runs=sum(1 for _ in reports),
            export_format="text",
            tool_version="0.1.0",
        )
