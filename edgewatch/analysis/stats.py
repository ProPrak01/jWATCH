"""Statistical helpers for multi-run benchmark analysis."""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
from scipy import stats

from edgewatch.correlation.engine import CorrelatedResult
from edgewatch.ollama.client import InferenceResult


@dataclass
class StatisticalResult:
    """Aggregate benchmark statistics for one model."""

    model: str
    n_runs: int
    warmup_runs: int
    effective_runs: int
    tokens_per_sec_mean: float
    tokens_per_sec_median: float
    tokens_per_sec_std: float
    tokens_per_sec_ci95_low: float
    tokens_per_sec_ci95_high: float
    ttft_mean_ms: float
    ttft_std_ms: float
    inference_duration_mean_sec: float
    outliers_detected: int
    is_stable: bool
    correlated_results: List[CorrelatedResult]
    total_tokens: int
    total_duration_sec: float


class StatisticalEngine:
    """Computes stable stats from a set of inference results."""

    def __init__(self, warmup_runs: int = 1, cv_threshold: float = 0.15):
        self.warmup_runs = max(0, warmup_runs)
        self.cv_threshold = cv_threshold

    def summarize(
        self,
        model: str,
        inference_results: Sequence[InferenceResult],
        correlated_results: Optional[Sequence[CorrelatedResult]] = None,
        warmup_runs: Optional[int] = None,
    ) -> StatisticalResult:
        warmups = self.warmup_runs if warmup_runs is None else max(0, warmup_runs)
        runs = list(inference_results)
        effective = runs[warmups:] if len(runs) > warmups else []

        tps_values = [r.tokens_per_sec_wall for r in effective if r.tokens_per_sec_wall is not None]
        ttft_values = [r.ttft_ms for r in effective if r.ttft_ms is not None]
        duration_values = [
            r.inference_duration_sec for r in effective if r.inference_duration_sec is not None
        ]

        mean_tps = _safe_mean(tps_values)
        std_tps = _safe_std(tps_values)
        ci_low, ci_high = self._ci95(tps_values)
        outliers = self._outlier_count_iqr(tps_values)

        return StatisticalResult(
            model=model,
            n_runs=len(runs),
            warmup_runs=min(warmups, len(runs)),
            effective_runs=len(effective),
            tokens_per_sec_mean=mean_tps,
            tokens_per_sec_median=_safe_median(tps_values),
            tokens_per_sec_std=std_tps,
            tokens_per_sec_ci95_low=ci_low,
            tokens_per_sec_ci95_high=ci_high,
            ttft_mean_ms=_safe_mean(ttft_values),
            ttft_std_ms=_safe_std(ttft_values),
            inference_duration_mean_sec=_safe_mean(duration_values),
            outliers_detected=outliers,
            is_stable=(std_tps / mean_tps < self.cv_threshold) if mean_tps > 0 else True,
            correlated_results=list(correlated_results or []),
            total_tokens=sum(r.total_tokens for r in effective),
            total_duration_sec=sum(duration_values),
        )

    @staticmethod
    def _ci95(values: Sequence[float]) -> Tuple[float, float]:
        if len(values) < 2:
            v = values[0] if values else 0.0
            return (v, v)

        mean = float(np.mean(values))
        sem = stats.sem(values)
        if sem == 0:
            return (mean, mean)

        interval = stats.t.interval(
            confidence=0.95,
            df=len(values) - 1,
            loc=mean,
            scale=sem,
        )
        return float(interval[0]), float(interval[1])

    @staticmethod
    def _outlier_count_iqr(values: Sequence[float]) -> int:
        if len(values) < 4:
            return 0

        q1 = np.percentile(values, 25)
        q3 = np.percentile(values, 75)
        iqr = q3 - q1
        if iqr == 0:
            return 0

        low = q1 - 1.5 * iqr
        high = q3 + 1.5 * iqr
        return sum(1 for v in values if v < low or v > high)

    def calculate_comparison_stats(self, results: Sequence[StatisticalResult]) -> dict:
        if not results:
            return {}

        best = max(results, key=lambda r: r.tokens_per_sec_mean)
        worst = min(results, key=lambda r: r.tokens_per_sec_mean)
        best_tps = best.tokens_per_sec_mean

        speedups = [
            (r.tokens_per_sec_mean / best_tps) if best_tps > 0 else 0.0
            for r in results
        ]

        return {
            "best_model": best.model,
            "worst_model": worst.model,
            "best_tps": best.tokens_per_sec_mean,
            "worst_tps": worst.tokens_per_sec_mean,
            "avg_speedup_ratio": float(np.mean(speedups)) if speedups else 0.0,
            "number_of_models": len(results),
        }


def run_statistical_benchmark(
    model: str,
    inference_results: Sequence[InferenceResult],
    correlated_results: Optional[Sequence[CorrelatedResult]] = None,
    warmup_runs: int = 1,
) -> StatisticalResult:
    """Convenience wrapper for summarizing already-collected run data."""
    engine = StatisticalEngine(warmup_runs=warmup_runs)
    return engine.summarize(model, inference_results, correlated_results, warmup_runs)


def _safe_mean(values: Sequence[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _safe_median(values: Sequence[float]) -> float:
    return float(np.median(values)) if values else 0.0


def _safe_std(values: Sequence[float]) -> float:
    return float(np.std(values)) if values else 0.0
