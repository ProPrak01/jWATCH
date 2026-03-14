"""
Interpolator for filling gaps between hardware samples.
Handles different interpolation methods for continuous vs discrete metrics.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from edgewatch.tegrastats.parser import TegraStatsSample


@dataclass
class InterpolationMethod:
    """
    Enum-like class for interpolation methods.
    """
    LINEAR = "linear"      # For continuous metrics (temperature, power)
    STEP = "step"          # For discrete metrics (GPU frequency, CPU load)
    NONE = "none"          # No interpolation


@dataclass
class InterpolatedSample:
    """
    A sample that has been interpolated between actual measurements.
    """
    timestamp: float                              # Interpolated timestamp
    wall_time: Optional[None] = None             # No wall time for interpolated samples
    ram_used_mb: int = 0                         # Interpolated RAM usage
    ram_total_mb: int = 0                         # Total RAM (unchanged)
    gpu_freq_pct: int = 0                         # Interpolated GPU frequency
    cpu_loads: List[int] = field(default_factory=list)  # Interpolated CPU loads
    cpu_temp: float = 0.0                         # Interpolated CPU temperature
    gpu_temp: float = 0.0                         # Interpolated GPU temperature
    tj_temp: float = 0.0                          # Interpolated TJ temperature
    power_mw: int = 0                            # Interpolated power
    is_interpolated: bool = True                  # Always true for interpolated samples
    interpolation_method: str = InterpolationMethod.LINEAR  # Method used
    gap_size_ms: float = 0.0                      # Size of gap in milliseconds
    confidence: float = 1.0                       # Confidence score (0.0-1.0)


class Interpolator:
    """
    Interpolates between hardware samples to fill gaps.

    Uses different methods for different metric types:
    - Linear interpolation: Continuous metrics (temperature, power, RAM)
    - Step interpolation: Discrete metrics (GPU frequency, CPU load)

    Confidence scoring based on gap size:
    - Small gaps (<500ms): High confidence
    - Medium gaps (500-1000ms): Medium confidence
    - Large gaps (>1000ms): Low confidence
    """

    def __init__(self, max_gap_ms: float = 500.0):
        """
        Initialize the interpolator.

        Args:
            max_gap_ms: Maximum gap size in ms before marking as low confidence (default 500ms)
        """
        self.max_gap_ms = max_gap_ms

    def interpolate_linear(self,
                          value_start: float,
                          value_end: float,
                          progress: float) -> float:
        """
        Linear interpolation between two values.

        Args:
            value_start: Starting value
            value_end: Ending value
            progress: Progress between start and end (0.0 to 1.0)

        Returns:
            Interpolated value
        """
        return value_start + (value_end - value_start) * progress

    def interpolate_step(self,
                        value_start: float,
                        value_end: float,
                        progress: float) -> float:
        """
        Step interpolation (discrete metrics).

        Uses the starting value until midpoint, then switches to ending value.

        Args:
            value_start: Starting value
            value_end: Ending value
            progress: Progress between start and end (0.0 to 1.0)

        Returns:
            Interpolated value (stepped)
        """
        return value_start if progress < 0.5 else value_end

    def calculate_confidence(self, gap_size_ms: float) -> float:
        """
        Calculate confidence score based on gap size.

        Args:
            gap_size_ms: Size of gap in milliseconds

        Returns:
            Confidence score between 0.0 and 1.0
        """
        if gap_size_ms <= self.max_gap_ms:
            # Small gap: high confidence
            return 1.0 - (gap_size_ms / (2.0 * self.max_gap_ms))
        elif gap_size_ms <= 2.0 * self.max_gap_ms:
            # Medium gap: medium confidence
            return 0.5 - ((gap_size_ms - self.max_gap_ms) / (2.0 * self.max_gap_ms))
        else:
            # Large gap: low confidence
            return 0.1

    def interpolate_at_timestamp(self,
                                samples: List[TegraStatsSample],
                                target_timestamp: float) -> Optional[InterpolatedSample]:
        """
        Interpolate hardware metrics at a specific timestamp.

        Args:
            samples: List of hardware samples (must be sorted by timestamp)
            target_timestamp: Target timestamp for interpolation

        Returns:
            InterpolatedSample at target timestamp, or None if interpolation not possible
        """
        if not samples:
            return None

        # Find surrounding samples
        lower_idx = None
        upper_idx = None

        for i, sample in enumerate(samples):
            if sample.timestamp <= target_timestamp:
                lower_idx = i
            if sample.timestamp >= target_timestamp:
                upper_idx = i
                break

        # Handle edge cases
        if lower_idx is None:
            # Target is before all samples
            if len(samples) > 0:
                return self._extrapolate_before(samples[0], target_timestamp)
            return None

        if upper_idx is None:
            # Target is after all samples
            return self._extrapolate_after(samples[-1], target_timestamp)

        if lower_idx == upper_idx:
            # Target exactly matches a sample timestamp
            return self._sample_to_interpolated(samples[lower_idx])

        # Interpolate between surrounding samples
        lower_sample = samples[lower_idx]
        upper_sample = samples[upper_idx]

        return self._interpolate_between(lower_sample, upper_sample, target_timestamp)

    def _interpolate_between(self,
                           lower_sample: TegraStatsSample,
                           upper_sample: TegraStatsSample,
                           target_timestamp: float) -> InterpolatedSample:
        """
        Interpolate between two samples.

        Args:
            lower_sample: Sample before target timestamp
            upper_sample: Sample after target timestamp
            target_timestamp: Target timestamp

        Returns:
            InterpolatedSample at target timestamp
        """
        gap_size_ms = (upper_sample.timestamp - lower_sample.timestamp) * 1000.0

        # Calculate progress (0.0 to 1.0)
        if gap_size_ms > 0:
            progress = (target_timestamp - lower_sample.timestamp) / (upper_sample.timestamp - lower_sample.timestamp)
        else:
            progress = 0.0

        # Calculate confidence
        confidence = self.calculate_confidence(gap_size_ms)

        # Interpolate continuous metrics (linear)
        cpu_temp = self.interpolate_linear(
            lower_sample.cpu_temp, upper_sample.cpu_temp, progress
        )
        gpu_temp = self.interpolate_linear(
            lower_sample.gpu_temp, upper_sample.gpu_temp, progress
        )
        tj_temp = self.interpolate_linear(
            lower_sample.tj_temp, upper_sample.tj_temp, progress
        )
        power_mw = int(round(self.interpolate_linear(
            lower_sample.power_mw, upper_sample.power_mw, progress
        )))
        ram_used_mb = int(round(self.interpolate_linear(
            lower_sample.ram_used_mb, upper_sample.ram_used_mb, progress
        )))

        # Interpolate discrete metrics (step)
        gpu_freq_pct = int(round(self.interpolate_step(
            lower_sample.gpu_freq_pct, upper_sample.gpu_freq_pct, progress
        )))

        # Interpolate CPU loads (step)
        cpu_loads = []
        for i in range(max(len(lower_sample.cpu_loads), len(upper_sample.cpu_loads))):
            lower_load = lower_sample.cpu_loads[i] if i < len(lower_sample.cpu_loads) else 0
            upper_load = upper_sample.cpu_loads[i] if i < len(upper_sample.cpu_loads) else 0
            cpu_loads.append(int(round(self.interpolate_step(lower_load, upper_load, progress))))

        return InterpolatedSample(
            timestamp=target_timestamp,
            wall_time=None,  # No wall time for interpolated samples
            ram_used_mb=ram_used_mb,
            ram_total_mb=lower_sample.ram_total_mb,
            gpu_freq_pct=gpu_freq_pct,
            cpu_loads=cpu_loads,
            cpu_temp=cpu_temp,
            gpu_temp=gpu_temp,
            tj_temp=tj_temp,
            power_mw=power_mw,
            is_interpolated=True,
            interpolation_method=InterpolationMethod.LINEAR,
            gap_size_ms=gap_size_ms,
            confidence=confidence
        )

    def _extrapolate_before(self,
                           sample: TegraStatsSample,
                           target_timestamp: float) -> InterpolatedSample:
        """
        Extrapolate before the first sample (use first sample values with low confidence).

        Args:
            sample: First available sample
            target_timestamp: Target timestamp (before sample)

        Returns:
            InterpolatedSample at target timestamp
        """
        gap_size_ms = (sample.timestamp - target_timestamp) * 1000.0
        confidence = self.calculate_confidence(gap_size_ms)

        return InterpolatedSample(
            timestamp=target_timestamp,
            wall_time=None,
            ram_used_mb=sample.ram_used_mb,
            ram_total_mb=sample.ram_total_mb,
            gpu_freq_pct=sample.gpu_freq_pct,
            cpu_loads=sample.cpu_loads.copy(),
            cpu_temp=sample.cpu_temp,
            gpu_temp=sample.gpu_temp,
            tj_temp=sample.tj_temp,
            power_mw=sample.power_mw,
            is_interpolated=True,
            interpolation_method=InterpolationMethod.NONE,
            gap_size_ms=gap_size_ms,
            confidence=confidence
        )

    def _extrapolate_after(self,
                          sample: TegraStatsSample,
                          target_timestamp: float) -> InterpolatedSample:
        """
        Extrapolate after the last sample (use last sample values with low confidence).

        Args:
            sample: Last available sample
            target_timestamp: Target timestamp (after sample)

        Returns:
            InterpolatedSample at target timestamp
        """
        gap_size_ms = (target_timestamp - sample.timestamp) * 1000.0
        confidence = self.calculate_confidence(gap_size_ms)

        return InterpolatedSample(
            timestamp=target_timestamp,
            wall_time=None,
            ram_used_mb=sample.ram_used_mb,
            ram_total_mb=sample.ram_total_mb,
            gpu_freq_pct=sample.gpu_freq_pct,
            cpu_loads=sample.cpu_loads.copy(),
            cpu_temp=sample.cpu_temp,
            gpu_temp=sample.gpu_temp,
            tj_temp=sample.tj_temp,
            power_mw=sample.power_mw,
            is_interpolated=True,
            interpolation_method=InterpolationMethod.NONE,
            gap_size_ms=gap_size_ms,
            confidence=confidence
        )

    def _sample_to_interpolated(self, sample: TegraStatsSample) -> InterpolatedSample:
        """
        Convert a real sample to interpolated format (for exact timestamp matches).

        Args:
            sample: Real sample

        Returns:
            InterpolatedSample with same values
        """
        return InterpolatedSample(
            timestamp=sample.timestamp,
            wall_time=sample.wall_time,
            ram_used_mb=sample.ram_used_mb,
            ram_total_mb=sample.ram_total_mb,
            gpu_freq_pct=sample.gpu_freq_pct,
            cpu_loads=sample.cpu_loads.copy(),
            cpu_temp=sample.cpu_temp,
            gpu_temp=sample.gpu_temp,
            tj_temp=sample.tj_temp,
            power_mw=sample.power_mw,
            is_interpolated=False,  # This is a real sample
            interpolation_method=InterpolationMethod.NONE,
            gap_size_ms=0.0,
            confidence=1.0
        )


def interpolate_at_timestamp(samples: List[TegraStatsSample],
                             target_timestamp: float,
                             max_gap_ms: float = 500.0) -> Optional[InterpolatedSample]:
    """
    Convenience function to interpolate at a specific timestamp.

    Args:
        samples: List of hardware samples (must be sorted by timestamp)
        target_timestamp: Target timestamp for interpolation
        max_gap_ms: Maximum gap size before low confidence

    Returns:
        InterpolatedSample at target timestamp, or None if not possible
    """
    interpolator = Interpolator(max_gap_ms=max_gap_ms)
    return interpolator.interpolate_at_timestamp(samples, target_timestamp)