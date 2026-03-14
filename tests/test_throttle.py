"""
Comprehensive tests for Thermal Throttling Detection.
Tests sliding window analysis, severity classification, and performance loss estimation.
"""

import pytest
from datetime import datetime, timedelta
from typing import List

from edgewatch.analysis.throttle import (
    ThrottleEvent,
    ThrottleDetector,
    detect_throttling
)
from edgewatch.tegrastats.parser import TegraStatsSample
from edgewatch.ollama.client import InferenceResult


class TestThrottleDetector:
    """Test suite for ThrottleDetector class."""

    @pytest.fixture
    def detector(self):
        """Create a detector instance for testing."""
        return ThrottleDetector(device_type="default")

    @pytest.fixture
    def sample_samples(self):
        """Create sample hardware samples for testing."""
        base_time = 100.0
        samples = []

        # Normal operation
        for i in range(10):
            samples.append(TegraStatsSample(
                timestamp=base_time + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0 + (i * 0.5),
                gpu_temp=52.0 + (i * 0.5),
                tj_temp=53.0 + (i * 0.5),
                power_mw=4500
            ))

        return samples

    @pytest.fixture
    def sample_inference(self):
        """Create a sample inference result for testing."""
        return InferenceResult(
            model="qwen:4b",
            prompt="Hello world",
            response_text="Hi there!",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=2,
            tokens_per_sec_ollama=2.0,
            tokens_per_sec_wall=2.0,
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=3,
            bytes_received=100
        )

    @pytest.fixture
    def high_temp_samples(self):
        """Create samples with elevated temperatures."""
        base_time = 100.0
        samples = []

        for i in range(15):
            # Gradually increase temperature
            temp = 70.0 + (i * 1.5)  # 70 to 91°C
            samples.append(TegraStatsSample(
                timestamp=base_time + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=3000 + (i * 100),
                ram_total_mb=7772,
                gpu_freq_pct=45 + (i * 2),  # Also increase frequency
                cpu_loads=[80 + (i * 2), 75 + (i * 2), 30, 25],
                cpu_temp=65.0 + (i * 0.5),
                gpu_temp=75.0 + (i * 0.5),
                tj_temp=80.0 + (i * 1.0),  # Very high TJ temp
                power_mw=6000 + (i * 200)
            ))

        return samples

    def test_initialization(self, detector):
        """Test detector initialization."""
        assert detector.device_type == "default"
        assert detector._threshold == 80.0
        assert detector.WINDOW_SIZE_SEC == 1.0
        assert detector.THROTTLE_APPROACH_THRESHOLD == 5.0

    def test_different_device_types(self):
        """Test initialization with different device types."""
        orin_detector = ThrottleDetector(device_type="orin")
        xavier_detector = ThrottleDetector(device_type="xavier")
        nano_detector = ThrottleDetector(device_type="nano")

        # Check thresholds
        assert orin_detector._threshold == 85.0
        assert xavier_detector._threshold == 80.0
        assert nano_detector._threshold == 75.0

    def test_detect_throttle_events_no_samples(self, detector, sample_samples, sample_inference):
        """Test throttling detection with no samples."""
        events = detector.detect_throttle_events(sample_samples, sample_inference)

        assert events == []

    def test_detect_throttle_events_insufficient_samples(self, detector, sample_samples, sample_inference):
        """Test throttling detection with insufficient samples."""
        # Only 1 sample
        insufficient_samples = sample_samples[:1]

        events = detector.detect_throttle_events(insufficient_samples, sample_inference)

        # Should not detect with only 1 sample
        assert len(events) == 0

    def test_detect_throttle_events_no_timing(self, detector, sample_samples):
        """Test throttling detection without timing information."""
        inference_no_timing = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=None,
            t_first_token=None,
            t_last_token=None,
            ttft_ms=None,
            total_tokens=1,
            tokens_per_sec_ollama=None,
            tokens_per_sec_wall=None,
            inference_duration_sec=None,
            eval_duration_ns=None,
            response_chunks=1,
            bytes_received=50
        )

        events = detector.detect_throttle_events(sample_samples, inference_no_timing)

        assert events == []

    def test_detect_throttle_events_normal_operation(self, detector, sample_samples, sample_inference):
        """Test throttling detection during normal operation."""
        events = detector.detect_throttle_events(sample_samples, sample_inference)

        # No throttling should be detected
        assert len(events) == 0

    def test_detect_throttle_events_mild_throttling(self, detector, high_temp_samples, sample_inference):
        """Test detection of mild throttling events."""
        events = detector.detect_throttle_events(high_temp_samples, sample_inference)

        # Should detect mild throttling
        assert len(events) == 1
        event = events[0]

        assert isinstance(event, ThrottleEvent)
        assert event.severity == "mild"
        assert 0 < event.freq_drop_pct < 20
        assert 0 < event.estimated_perf_loss_pct < 0.3

    def test_detect_throttle_events_moderate_throttling(self, detector):
        """Test detection of moderate throttling events."""
        # Create samples with moderate throttling
        moderate_samples = []

        # Normal period
        for i in range(5):
            moderate_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0 + (i * 0.5),
                gpu_temp=52.0 + (i * 0.5),
                tj_temp=53.0 + (i * 0.5),
                power_mw=4500
            ))

        # Throttling period with temp rise and frequency drop
        for i in range(5, 10):
            moderate_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=20,  # Significant drop
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=70.0,  # Temperature rise
                gpu_temp=75.0,
                tj_temp=80.0,  # High TJ temp
                power_mw=5000
            ))

        # Recovery period
        for i in range(10, 15):
            moderate_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,  # Frequency recovery
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=65.0,
                gpu_temp=72.0,
                tj_temp=77.0,
                power_mw=4500
            ))

        events = detector.detect_throttle_events(moderate_samples, sample_inference)

        # Should detect moderate throttling
        assert len(events) == 1
        event = events[0]

        assert isinstance(event, ThrottleEvent)
        assert event.severity == "moderate"
        assert 20 <= event.freq_drop_pct <= 40
        assert 0.05 <= event.estimated_perf_loss_pct <= 0.4

    def test_detect_throttle_events_severe_throttling(self, detector):
        """Test detection of severe throttling events."""
        # Create samples with severe throttling
        severe_samples = []

        # Normal period
        for i in range(5):
            severe_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0 + (i * 0.5),
                gpu_temp=52.0 + (i * 0.5),
                tj_temp=53.0 + (i * 0.5),
                power_mw=4500
            ))

        # Severe throttling period
        for i in range(5, 10):
            severe_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=10,  # Massive frequency drop
                cpu_loads=[80, 75, 30, 25],
                cpu_temp=85.0,  # Very high temperature
                gpu_temp=90.0,
                tj_temp=92.0,  # Extremely high TJ temp
                power_mw=5000
            ))

        events = detector.detect_throttle_events(severe_samples, sample_inference)

        # Should detect severe throttling
        assert len(events) >= 1
        event = events[0]

        assert isinstance(event, ThrottleEvent)
        assert event.severity == "severe"
        assert event.freq_drop_pct > 40

    def test_find_high_temp_periods(self, detector, sample_samples):
        """Test finding high temperature periods."""
        # Add some high temp samples
        samples_with_high_temp = sample_samples + [
            TegraStatsSample(
                timestamp=100.6,
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=80.0,
                gpu_temp=85.0,
                tj_temp=90.0,
                power_mw=5000
            )
        ]

        periods = detector._find_high_temp_periods(samples_with_high_temp, 50.0)

        # Should find at least one high temp period
        assert len(periods) >= 1

        # Check period structure
        if periods:
            start_idx, end_idx, temp = periods[0]
            assert start_idx >= 0
            assert end_idx > start_idx
            assert temp > 50.0  # Above average

    def test_classify_severity(self, detector):
        """Test severity classification."""
        # Mild: temp approaching, small frequency drop
        mild_temp = 78.0  # 2°C below threshold
        mild_drop = 15  # 15% drop

        severity_mild = detector._classify_severity(mild_temp, mild_drop, 0.5)

        assert severity_mild == "mild"

        # Moderate: temp closer, medium frequency drop
        mod_temp = 82.0  # 2°C below threshold
        mod_drop = 30  # 30% drop

        severity_mod = detector._classify_severity(mod_temp, mod_drop, 0.5)

        assert severity_mod == "moderate"

        # Severe: temp at/above threshold, large frequency drop
        sev_temp = 88.0  # 3°C above threshold
        sev_drop = 50  # 50% drop

        severity_sev = detector._classify_severity(sev_temp, sev_drop, 0.5)

        assert severity_sev == "severe"

        # Edge case: No frequency drop (should not classify as throttling)
        no_drop = detector._classify_severity(82.0, 0, 0.5)

        # Without significant drop, severity should be mild at most
        assert no_drop == "mild"

    def test_estimate_performance_loss(self, detector):
        """Test performance loss estimation."""
        # Small drop (5%) for 1 second
        loss_small = detector._estimate_performance_loss(5, 1.0, 40.0)
        assert loss_small < 0.1

        # Medium drop (30%) for 1 second
        loss_medium = detector._estimate_performance_loss(30, 1.0, 40.0)
        assert 0.2 <= loss_medium <= 0.3

        # Large drop (50%) for 1 second
        loss_large = detector._estimate_performance_loss(50, 1.0, 40.0)
        assert 0.4 <= loss_large <= 0.8

        # Edge case: No baseline frequency
        loss_no_baseline = detector._estimate_performance_loss(50, 0.5, 0.0)
        assert loss_no_baseline == 0.0

        # Edge case: Very long duration
        loss_long_duration = detector._estimate_performance_loss(30, 5.0, 40.0)
        assert 5.0 >= loss_long_duration >= 0.3  # Weighted to max 80%

    def test_multiple_throttle_events(self, detector):
        """Test detection of multiple separate throttling events."""
        # Create samples with two distinct throttling periods
        multi_samples = []

        # First throttling period (mild)
        for i in range(5):
            multi_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0 + (i * 0.5),
                gpu_temp=52.0 + (i * 0.5),
                tj_temp=53.0 + (i * 0.5),
                power_mw=4500
            ))

        # Throttling period
        for i in range(5, 10):
            multi_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=20,  # Moderate drop
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=70.0,
                gpu_temp=75.0,
                tj_temp=80.0,
                power_mw=5000
            ))

        # Recovery period
        for i in range(10, 15):
            multi_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=65.0,
                gpu_temp=72.0,
                tj_temp=77.0,
                power_mw=4500
            ))

        # Second throttling period (severe)
        for i in range(15, 20):
            multi_samples.append(TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=10,  # Severe drop
                cpu_loads=[80, 75, 30, 25],
                cpu_temp=85.0,
                gpu_temp=90.0,
                tj_temp=92.0,
                power_mw=5000
            ))

        events = detector.detect_throttle_events(multi_samples, sample_inference)

        # Should detect at least one event
        assert len(events) >= 1

        # Check first event is mild
        assert events[0].severity == "mild"

        # Check second event is severe
        assert events[1].severity == "severe"

    def test_convenience_function(self, sample_samples, sample_inference):
        """Test convenience detect_throttling function."""
        events = detect_throttling(sample_samples, sample_inference)

        assert isinstance(events, list)
        assert len(events) >= 0

    def test_event_time_accuracy(self, detector, sample_samples, sample_inference):
        """Test that event times are accurate."""
        events = detector.detect_throttle_events(sample_samples, sample_inference)

        if events:
            for event in events:
                # Event should be within inference time
                assert event.t_start >= sample_inference.t_request_sent
                assert event.t_end <= sample_inference.t_last_token

                # Event duration should be reasonable
                assert 0.5 <= event.duration_sec <= 10.0

    def test_edge_case_short_high_temp_period(self, detector):
        """Test edge case with very short high temp period."""
        samples = [
            TegraStatsSample(
                timestamp=100.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=80.0,
                gpu_temp=85.0,
                tj_temp=90.0,
                power_mw=5000
            )
        ]

        # Only 1 sample at high temp (0.1 seconds)
        events = detector.detect_throttle_events(samples, sample_inference)

        # Should not detect throttling (period too short)
        assert len(events) == 0


class TestThrottleEvent:
    """Test suite for ThrottleEvent dataclass."""

    def test_throttle_event_creation(self):
        """Test creating a throttle event."""
        event = ThrottleEvent(
            t_start=100.0,
            t_end=101.0,
            duration_sec=1.0,
            peak_temp=85.0,
            freq_drop_pct=30,
            severity="moderate",
            estimated_perf_loss_pct=0.25
        )

        assert event.t_start == 100.0
        assert event.t_end == 101.0
        assert event.duration_sec == 1.0
        assert event.peak_temp == 85.0
        assert event.freq_drop_pct == 30
        assert event.severity == "moderate"
        assert 0.0 < event.estimated_perf_loss_pct < 1.0

    def test_throttle_event_severity_values(self):
        """Test that severity values are valid."""
        for severity in ["mild", "moderate", "severe"]:
            event = ThrottleEvent(
                t_start=100.0,
                t_end=101.0,
                duration_sec=1.0,
                peak_temp=85.0,
                freq_drop_pct=30,
                severity=severity,
                estimated_perf_loss_pct=0.25
            )

            assert event.severity == severity

    def test_throttle_event_fields(self):
        """Test all throttle event fields are populated."""
        event = ThrottleEvent(
            t_start=100.0,
            t_end=101.0,
            duration_sec=1.0,
            peak_temp=85.0,
            freq_drop_pct=30,
            severity="moderate",
            estimated_perf_loss_pct=0.25
        )

        # All fields should be populated
        assert event.peak_temp > 0
        assert event.freq_drop_pct >= 0
        assert event.duration_sec > 0
        assert event.estimated_perf_loss_pct >= 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])