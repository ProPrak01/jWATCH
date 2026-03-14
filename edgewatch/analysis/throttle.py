"""
Thermal throttling detection algorithm for Jetson devices.
Analyzes hardware samples to detect thermal throttling events and severity.
"""

from dataclasses import dataclass
from typing import List, Optional

from edgewatch.tegrastats.parser import TegraStatsSample
from edgewatch.ollama.client import InferenceResult


@dataclass
class ThrottleEvent:
    """
    Represents a thermal throttling event.

    The detection is based on analyzing:
    - Thermal junction temperature approaching thresholds
    - GPU frequency drops during high temperature periods
    - Performance impact estimation
    """
    t_start: float                    # Event start time (monotonic)
    t_end: float                      # Event end time (monotonic)
    duration_sec: float                # Event duration
    peak_temp: float                  # Peak thermal junction temperature during event
    freq_drop_pct: int                # How much GPU frequency dropped
    severity: str                      # "mild" | "moderate" | "severe"
    estimated_perf_loss_pct: float    # Estimated tokens/sec loss


class ThrottleDetector:
    """
    Detects thermal throttling events using sliding window analysis.

    Algorithm:
    - Monitor thermal junction temperature (tj_temp) approaching thresholds
    - Detect concurrent GPU frequency drops (>20%)
    - Classify severity based on temperature and frequency drop magnitude
    - Estimate performance impact in tokens/sec

    Thresholds vary by device type:
    - Orin: 85°C
    - Xavier: 80°C
    - Nano: 75°C
    - Default: 80°C
    """

    # Device-specific thermal thresholds
    THROTTLE_THRESHOLDS = {
        "orin": 85.0,
        "xavier": 80.0,
        "nano": 75.0,
        "default": 80.0
    }

    # Severity classification criteria
    # Mild: temp approaching threshold-5°C, freq drop <20%
    # Moderate: temp approaching threshold-2°C, freq drop 20-40%
    # Severe: temp at or above threshold, freq drop >40%

    WINDOW_SIZE_SEC = 1.0           # 1 second sliding window
    THROTTLE_APPROACH_THRESHOLD = 5.0  # Temp within 5°C of threshold

    def __init__(self, device_type: str = "default"):
        """
        Initialize throttle detector.

        Args:
            device_type: Device type ("orin", "xavier", "nano", "default")
        """
        self.device_type = device_type
        self._threshold = self.THROTTLE_THRESHOLDS.get(device_type, self.THROTTLE_THRESHOLDS["default"])

    def detect_throttle_events(self,
                           samples: List[TegraStatsSample],
                           inference: InferenceResult) -> List[ThrottleEvent]:
        """
        Detect thermal throttling events during an inference run.

        Args:
            samples: List of hardware samples from tegrastats
            inference: Inference result with timing information

        Returns:
            List of detected throttle events, sorted by start time
        """
        if not samples:
            return []

        if not inference.t_request_sent or not inference.t_last_token:
            return []

        # Get samples within inference window
        inference_start = inference.t_request_sent
        inference_end = inference.t_last_token

        window_samples = [
            sample for sample in samples
            if sample.timestamp >= inference_start and sample.timestamp <= inference_end
        ]

        if len(window_samples) < 2:
            return []

        # Analyze samples for throttling patterns
        events = self._analyze_samples(window_samples, inference)

        return events

    def _analyze_samples(self,
                       samples: List[TegraStatsSample],
                       inference: InferenceResult) -> List[ThrottleEvent]:
        """
        Analyze hardware samples for throttling patterns.

        Algorithm:
        1. Calculate baseline metrics (avg temp, avg frequency)
        2. Detect periods of high temperature
        3. Look for frequency drops during high temp periods
        4. Classify severity and estimate impact

        Args:
            samples: Hardware samples sorted by timestamp
            inference: Inference result for timing reference

        Returns:
            List of detected throttle events
        """
        events = []

        if len(samples) < 3:
            return []

        # Calculate baseline metrics
        avg_temp = sum(s.tj_temp for s in samples) / len(samples)
        avg_freq = sum(s.gpu_freq_pct for s in samples) / len(samples)

        # Find high temperature periods
        high_temp_samples = self._find_high_temp_periods(samples, avg_temp)

        if not high_temp_samples:
            return []

        # Analyze each high temp period
        for period_start, period_end, high_temp_samples in high_temp_samples:
            # Get samples in this period
            period_samples = high_temp_samples[period_start:period_end]

            # Calculate frequency during this period
            period_avg_freq = sum(s.gpu_freq_pct for s in period_samples) / len(period_samples)

            # Check for significant frequency drop
            freq_drop = avg_freq - period_avg_freq

            # Only classify as throttling if significant drop
            if freq_drop > 20 and period_avg_freq > 30:
                # This is a throttling event
                duration_sec = (period_end - period_start) * 0.1
                severity = self._classify_severity(
                    high_temp,
                    freq_drop,
                    duration_sec
                )

                # Estimate performance loss
                perf_loss = self._estimate_performance_loss(
                    freq_drop,
                    duration_sec,
                    period_avg_freq
                )

                # Find exact start and end of the drop
                event_start, event_end = self._find_drop_boundaries(
                    period_samples,
                    period_avg_freq,
                    freq_drop
                )

                event = ThrottleEvent(
                    t_start=event_start,
                    t_end=event_end,
                    duration_sec=(event_end - event_start),
                    peak_temp=max(s.tj_temp for s in period_samples),
                    freq_drop_pct=freq_drop,
                    severity=severity.value,
                    estimated_perf_loss_pct=perf_loss
                )

                events.append(event)

        return events

    def _find_high_temp_periods(self,
                            samples: List[TegraStatsSample],
                            baseline_temp: float) -> List[tuple[int, int, float]]:
        """
        Find periods where temperature is significantly above baseline.

        Args:
            samples: Hardware samples sorted by timestamp
            baseline_temp: Average temperature

        Returns:
            List of (start_idx, end_idx, temp) tuples for high temp periods
        """
        periods = []

        # Find continuous periods above threshold
        above_threshold = baseline_temp + self.THROTTLE_APPROACH_THRESHOLD

        i = 0
        n = len(samples)

        while i < n:
            if samples[i].tj_temp > above_threshold:
                # Start of high temp period
                start_idx = i

                # Find end of high temp period
                while i < n and samples[i].tj_temp > above_threshold:
                    i += 1

                end_idx = i - 1

                # Only consider periods that last at least 0.5 seconds
                if (end_idx - start_idx) * 0.1 >= 0.5:
                    # Calculate average temp during this period
                    period_temp = sum(
                        s.tj_temp for s in samples[start_idx:end_idx + 1]
                    ) / (end_idx - start_idx + 1)

                    periods.append((start_idx, end_idx, period_temp))

                # Skip to next sample after this period
                i = end_idx + 1
            else:
                i += 1

        return periods

    def _find_drop_boundaries(self,
                           period_samples: List[TegraStatsSample],
                           baseline_freq: float,
                           drop_threshold: int) -> tuple[float, float]:
        """
        Find the exact start and end of a frequency drop.

        Args:
            period_samples: Samples in a high temp period
            baseline_freq: Expected frequency before drop
            drop_threshold: Minimum frequency drop to consider

        Returns:
            (drop_start, drop_end) timestamps
        """
        # Find where frequency drops below threshold
        drop_start = None
        drop_end = None

        for i, sample in enumerate(period_samples):
            if sample.gpu_freq_pct < (baseline_freq - drop_threshold):
                if drop_start is None:
                    drop_start = sample.timestamp
                drop_end = sample.timestamp

        return (drop_start, drop_end)

    def _classify_severity(self,
                       temp: float,
                       freq_drop: int,
                       duration_sec: float) -> str:
        """
        Classify throttling severity based on temperature and frequency drop.

        Args:
            temp: Peak thermal junction temperature
            freq_drop: GPU frequency drop in percentage
            duration_sec: Duration of the event

        Returns:
            Severity: "mild", "moderate", or "severe"
        """
        # Get temperature margin from threshold
        temp_margin = temp - self._threshold

        # Calculate frequency drop factor (0-1, lower is worse)
        freq_drop_factor = 1.0 - (freq_drop / 100.0)

        if temp_margin > 5.0 and freq_drop_factor > 0.2:
            # Moderate: temp approaching threshold-2°C, 20-40% drop
            return "moderate"
        elif temp_margin >= 0 and freq_drop_factor > 0.4:
            # Severe: temp at/above threshold, >40% drop
            return "severe"
        else:
            # Mild: temp approaching threshold-5°C, <20% drop
            return "mild"

    def _estimate_performance_loss(self,
                                 freq_drop: int,
                                 duration_sec: float,
                                 baseline_freq: float) -> float:
        """
        Estimate performance loss in tokens/sec.

        Based on the assumption that frequency is proportional to performance.

        Args:
            freq_drop: Frequency drop percentage
            duration_sec: Duration of throttling event
            baseline_freq: Expected frequency

        Returns:
            Estimated performance loss percentage
        """
        if duration_sec <= 0 or baseline_freq <= 0:
            return 0.0

        # Assume linear relationship between frequency and tokens/sec
        # Frequency drop of X% = X% performance loss
        # Weighted by duration (longer events = more impact)

        freq_loss_factor = freq_drop / 100.0
        duration_weight = min(duration_sec, 5.0) / 5.0  # Cap at 5 seconds for weighting

        estimated_loss = freq_loss_factor * duration_weight

        # Cap at reasonable maximum
        return min(estimated_loss, 0.8)  # Maximum 80% loss


def detect_throttling(samples: List[TegraStatsSample],
                     inference: InferenceResult,
                     device_type: str = "default") -> List[ThrottleEvent]:
    """
    Convenience function to detect throttling events.

    Args:
        samples: Hardware samples from tegrastats
        inference: Inference result with timing information
        device_type: Device type for threshold selection

    Returns:
        List of detected throttle events
    """
    detector = ThrottleDetector(device_type=device_type)
    return detector.detect_throttle_events(samples, inference)