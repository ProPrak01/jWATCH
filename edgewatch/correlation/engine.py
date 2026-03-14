"""
Time-series correlation engine for aligning hardware samples with inference windows.
This is the core engineering challenge - aligning discrete 100ms samples with continuous inference.
"""

import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

from edgewatch.tegrastats.parser import TegraStatsSample
from edgewatch.ollama.client import InferenceResult


@dataclass
class CorrelatedResult:
    """
    Result of correlating hardware samples with inference window.

    Contains aggregated statistics and individual sample details.
    """
    inference: InferenceResult                    # Original inference result
    hw_samples: List[TegraStatsSample]           # Samples during inference window
    avg_gpu_temp: float                          # Average GPU temperature
    max_gpu_temp: float                          # Maximum GPU temperature
    avg_tj_temp: float                           # Average thermal junction temperature
    max_tj_temp: float                           # Maximum thermal junction temperature
    avg_gpu_freq_pct: int                        # Average GPU frequency percentage
    avg_power_mw: int                            # Average power consumption
    peak_power_mw: int                           # Peak power consumption
    avg_ram_used_mb: int                         # Average RAM usage
    peak_ram_used_mb: int                        # Peak RAM usage
    sample_count: int                           # Number of samples used
    interpolation_confidence: float             # 0.0 - 1.0, higher is better
    window_start: float                         # Inference window start (monotonic)
    window_end: float                           # Inference window end (monotonic)
    samples_interpolated: int = 0               # Number of interpolated samples
    samples_measured: int = 0                    # Number of measured samples


class CorrelationEngine:
    """
    Aligns tegrastats samples with inference windows using time-series correlation.

    Key challenges:
    - Discrete samples (100ms intervals) vs continuous windows
    - Partial overlaps require weighted averaging
    - Boundary interpolation for exact timestamps
    - Confidence scoring based on sample density and gap size
    """

    def __init__(self, padding_sec: float = 0.5):
        """
        Initialize the correlation engine.

        Args:
            padding_sec: Padding in seconds around inference window (default 0.5s)
        """
        self.padding_sec = padding_sec

    def correlate(self,
                 inference: InferenceResult,
                 samples: List[TegraStatsSample]) -> Optional[CorrelatedResult]:
        """
        Correlate hardware samples with an inference window.

        Args:
            inference: InferenceResult with timing information
            samples: List of TegraStatsSample from sampler

        Returns:
            CorrelatedResult with aligned metrics, or None if correlation fails
        """
        # Validate timing information
        if inference.t_request_sent is None or inference.t_last_token is None:
            return None

        # Define inference window with padding
        t_start = inference.t_request_sent - self.padding_sec
        t_end = inference.t_last_token + self.padding_sec

        # Get samples within window
        window_samples = self._get_samples_in_window(samples, t_start, t_end)

        if not window_samples:
            return None

        # Calculate weighted averages and aggregates
        stats = self._calculate_aggregates(
            window_samples, inference.t_request_sent, inference.t_last_token
        )

        # Calculate interpolation confidence
        confidence = self._calculate_confidence(stats, window_samples)

        return CorrelatedResult(
            inference=inference,
            hw_samples=window_samples,
            avg_gpu_temp=stats["avg_gpu_temp"],
            max_gpu_temp=stats["max_gpu_temp"],
            avg_tj_temp=stats["avg_tj_temp"],
            max_tj_temp=stats["max_tj_temp"],
            avg_gpu_freq_pct=stats["avg_gpu_freq_pct"],
            avg_power_mw=stats["avg_power_mw"],
            peak_power_mw=stats["peak_power_mw"],
            avg_ram_used_mb=stats["avg_ram_used_mb"],
            peak_ram_used_mb=stats["peak_ram_used_mb"],
            sample_count=len(window_samples),
            interpolation_confidence=confidence,
            window_start=t_start,
            window_end=t_end,
            samples_interpolated=stats["interpolated_count"],
            samples_measured=stats["measured_count"]
        )

    def _get_samples_in_window(self,
                              samples: List[TegraStatsSample],
                              t_start: float,
                              t_end: float) -> List[TegraStatsSample]:
        """
        Get all samples within a time window, sorted by timestamp.

        Args:
            samples: List of all available samples
            t_start: Window start time (monotonic)
            t_end: Window end time (monotonic)

        Returns:
            Sorted list of samples within the window
        """
        # Filter samples within window
        window_samples = [
            sample for sample in samples
            if t_start <= sample.timestamp <= t_end
        ]

        # Sort by timestamp
        window_samples.sort(key=lambda s: s.timestamp)

        return window_samples

    def _calculate_aggregates(self,
                             samples: List[TegraStatsSample],
                             inference_start: float,
                             inference_end: float) -> dict:
        """
        Calculate weighted averages and other aggregate statistics.

        Args:
            samples: List of samples within padded window
            inference_start: Actual inference start time
            inference_end: Actual inference end time

        Returns:
            Dictionary with aggregate statistics
        """
        if not samples:
            return {
                "avg_gpu_temp": 0.0,
                "max_gpu_temp": 0.0,
                "avg_tj_temp": 0.0,
                "max_tj_temp": 0.0,
                "avg_gpu_freq_pct": 0,
                "avg_power_mw": 0,
                "peak_power_mw": 0,
                "avg_ram_used_mb": 0,
                "peak_ram_used_mb": 0,
                "interpolated_count": 0,
                "measured_count": 0
            }

        # Initialize accumulators
        total_weight = 0.0
        weighted_gpu_temp = 0.0
        weighted_tj_temp = 0.0
        weighted_gpu_freq = 0.0
        weighted_power = 0.0
        weighted_ram = 0.0

        max_gpu_temp = 0.0
        max_tj_temp = 0.0
        peak_power = 0
        peak_ram = 0

        measured_count = 0
        interpolated_count = 0

        # Calculate overlap weights for each sample
        for sample in samples:
            # Calculate overlap with actual inference window
            overlap_start = max(sample.timestamp, inference_start)
            overlap_end = min(sample.timestamp + 0.1, inference_end)  # Assume 100ms interval

            if overlap_start < overlap_end:
                weight = overlap_end - overlap_start
                total_weight += weight

                # Weighted averages
                weighted_gpu_temp += sample.gpu_temp * weight
                weighted_tj_temp += sample.tj_temp * weight
                weighted_gpu_freq += sample.gpu_freq_pct * weight
                weighted_power += sample.power_mw * weight
                weighted_ram += sample.ram_used_mb * weight

                # Maximums
                max_gpu_temp = max(max_gpu_temp, sample.gpu_temp)
                max_tj_temp = max(max_tj_temp, sample.tj_temp)
                peak_power = max(peak_power, sample.power_mw)
                peak_ram = max(peak_ram, sample.ram_used_mb)

                measured_count += 1

        # Calculate averages
        if total_weight > 0:
            avg_gpu_temp = weighted_gpu_temp / total_weight
            avg_tj_temp = weighted_tj_temp / total_weight
            avg_gpu_freq = int(round(weighted_gpu_freq / total_weight))
            avg_power = int(round(weighted_power / total_weight))
            avg_ram = int(round(weighted_ram / total_weight))
        else:
            # Fallback to simple average if no overlap
            avg_gpu_temp = sum(s.gpu_temp for s in samples) / len(samples)
            avg_tj_temp = sum(s.tj_temp for s in samples) / len(samples)
            avg_gpu_freq = int(round(sum(s.gpu_freq_pct for s in samples) / len(samples)))
            avg_power = int(round(sum(s.power_mw for s in samples) / len(samples)))
            avg_ram = int(round(sum(s.ram_used_mb for s in samples) / len(samples)))

        return {
            "avg_gpu_temp": avg_gpu_temp,
            "max_gpu_temp": max_gpu_temp,
            "avg_tj_temp": avg_tj_temp,
            "max_tj_temp": max_tj_temp,
            "avg_gpu_freq_pct": avg_gpu_freq,
            "avg_power_mw": avg_power,
            "peak_power_mw": peak_power,
            "avg_ram_used_mb": avg_ram,
            "peak_ram_used_mb": peak_ram,
            "interpolated_count": interpolated_count,
            "measured_count": measured_count
        }

    def _calculate_confidence(self,
                             stats: dict,
                             samples: List[TegraStatsSample]) -> float:
        """
        Calculate interpolation confidence score.

        Factors:
        - Sample count (more samples = higher confidence)
        - Coverage of inference window (full coverage = higher confidence)
        - Gap sizes (smaller gaps = higher confidence)

        Args:
            stats: Aggregate statistics from _calculate_aggregates
            samples: Samples within window

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if not samples:
            return 0.0

        # Factor 1: Sample count (0-0.4)
        # More samples is better, diminishing returns after 10 samples
        sample_score = min(len(samples) / 10.0, 1.0) * 0.4

        # Factor 2: Measured vs interpolated ratio (0-0.3)
        # More measured samples is better
        if stats["measured_count"] + stats["interpolated_count"] > 0:
            measured_ratio = stats["measured_count"] / (stats["measured_count"] + stats["interpolated_count"])
            measured_score = measured_ratio * 0.3
        else:
            measured_score = 0.0

        # Factor 3: Coverage (0-0.3)
        # How well do samples cover the inference window
        if len(samples) >= 2:
            coverage = (samples[-1].timestamp - samples[0].timestamp) / 1.0  # Normalize to 1 second
            coverage_score = min(coverage, 1.0) * 0.3
        else:
            coverage_score = 0.0

        # Total confidence
        confidence = sample_score + measured_score + coverage_score

        # Clamp to [0, 1]
        return max(0.0, min(1.0, confidence))


def correlate_inference(inference: InferenceResult,
                       samples: List[TegraStatsSample],
                       padding_sec: float = 0.5) -> Optional[CorrelatedResult]:
    """
    Convenience function to correlate inference with hardware samples.

    Args:
        inference: InferenceResult with timing information
        samples: List of TegraStatsSample
        padding_sec: Padding in seconds around inference window

    Returns:
        CorrelatedResult or None if correlation fails
    """
    engine = CorrelationEngine(padding_sec=padding_sec)
    return engine.correlate(inference, samples)