"""
Comprehensive tests for Interpolator.
Tests linear/step interpolation, confidence scoring, and gap handling.
"""

import pytest
from datetime import datetime

from edgewatch.correlation.interpolator import (
    Interpolator,
    InterpolatedSample,
    InterpolationMethod,
    interpolate_at_timestamp
)
from edgewatch.tegrastats.parser import TegraStatsSample


class TestInterpolator:
    """Test suite for Interpolator class."""

    @pytest.fixture
    def interpolator(self):
        """Create an interpolator instance for testing."""
        return Interpolator(max_gap_ms=500.0)

    @pytest.fixture
    def sample_samples(self):
        """Create sample hardware samples for testing."""
        samples = []

        # Create samples every 100ms with linear progression
        for i in range(10):
            sample = TegraStatsSample(
                timestamp=100.0 + (i * 0.1),  # 100.0 to 109.0
                wall_time=datetime.now(),
                ram_used_mb=2000 + (i * 100),  # 2000 to 2900
                ram_total_mb=7772,
                gpu_freq_pct=30 + (i * 5),  # 30 to 75
                cpu_loads=[10 + i, 5 + i, 0, 0],  # Varying loads
                cpu_temp=40.0 + (i * 1.0),  # 40.0 to 49.0
                gpu_temp=45.0 + (i * 1.0),  # 45.0 to 54.0
                tj_temp=46.0 + (i * 1.0),  # 46.0 to 55.0
                power_mw=4000 + (i * 200)  # 4000 to 5800
            )
            samples.append(sample)

        return samples

    def test_interpolate_linear_continuous(self, interpolator):
        """Test linear interpolation for continuous metrics."""
        value_start = 40.0
        value_end = 50.0
        progress = 0.5  # Midpoint

        result = interpolator.interpolate_linear(value_start, value_end, progress)

        assert result == 45.0  # Exactly halfway

    def test_interpolate_linear_edge_cases(self, interpolator):
        """Test linear interpolation at boundaries."""
        start_val = 40.0
        end_val = 50.0

        # At start
        assert interpolator.interpolate_linear(start_val, end_val, 0.0) == start_val

        # At end
        assert interpolator.interpolate_linear(start_val, end_val, 1.0) == end_val

        # Before start (should handle gracefully)
        result = interpolator.interpolate_linear(start_val, end_val, -0.1)
        assert result < start_val

        # After end (should handle gracefully)
        result = interpolator.interpolate_linear(start_val, end_val, 1.1)
        assert result > end_val

    def test_interpolate_step_discrete(self, interpolator):
        """Test step interpolation for discrete metrics."""
        value_start = 40
        value_end = 60

        # Before midpoint - should return start value
        assert interpolator.interpolate_step(value_start, value_end, 0.4) == value_start

        # At midpoint - should return end value
        assert interpolator.interpolate_step(value_start, value_end, 0.5) == value_end

        # After midpoint - should return end value
        assert interpolator.interpolate_step(value_start, value_end, 0.6) == value_end

    def test_calculate_confidence_small_gap(self, interpolator):
        """Test confidence calculation for small gaps."""
        gap_100ms = 100.0
        gap_200ms = 200.0
        gap_500ms = 500.0

        confidence_100 = interpolator.calculate_confidence(gap_100ms)
        confidence_200 = interpolator.calculate_confidence(gap_200ms)
        confidence_500 = interpolator.calculate_confidence(gap_500ms)

        # Smaller gaps should have higher confidence
        assert confidence_100 > confidence_200
        assert confidence_200 > confidence_500

        # All should be >= 0.5 for small gaps
        assert confidence_500 >= 0.5

    def test_calculate_confidence_medium_gap(self, interpolator):
        """Test confidence calculation for medium gaps."""
        gap_600ms = 600.0
        gap_800ms = 800.0
        gap_1000ms = 1000.0

        confidence_600 = interpolator.calculate_confidence(gap_600ms)
        confidence_800 = interpolator.calculate_confidence(gap_800ms)
        confidence_1000 = interpolator.calculate_confidence(gap_1000ms)

        # Medium gaps should have medium confidence
        assert 0.0 <= confidence_600 <= 0.5
        assert 0.0 <= confidence_800 <= 0.5
        assert 0.0 <= confidence_1000 <= 0.5

    def test_calculate_confidence_large_gap(self, interpolator):
        """Test confidence calculation for large gaps."""
        gap_1500ms = 1500.0
        gap_2000ms = 2000.0

        confidence_1500 = interpolator.calculate_confidence(gap_1500ms)
        confidence_2000 = interpolator.calculate_confidence(gap_2000ms)

        # Large gaps should have low confidence
        assert 0.0 <= confidence_1500 <= 0.2
        assert 0.0 <= confidence_2000 <= 0.2

    def test_interpolate_at_timestamp_midpoint(self, interpolator, sample_samples):
        """Test interpolation at midpoint between samples."""
        # Target at 104.55 (between samples at 104.0 and 104.5)
        target_timestamp = 104.55

        result = interpolator.interpolate_at_timestamp(sample_samples, target_timestamp)

        assert result is not None
        assert isinstance(result, InterpolatedSample)
        assert result.timestamp == target_timestamp
        assert result.is_interpolated is True

        # Check linear interpolation of continuous metrics
        # Should be between the two surrounding samples
        assert 40.0 < result.cpu_temp < 50.0
        assert 45.0 < result.gpu_temp < 55.0

    def test_interpolate_at_timestamp_exact_match(self, interpolator, sample_samples):
        """Test interpolation at exact sample timestamp."""
        # Target exactly at sample timestamp
        target_timestamp = 105.0  # Exact match

        result = interpolator.interpolate_at_timestamp(sample_samples, target_timestamp)

        assert result is not None
        assert isinstance(result, InterpolatedSample)
        assert result.timestamp == target_timestamp
        # Should not be marked as interpolated since it's exact match
        assert result.is_interpolated is False
        assert result.confidence == 1.0

    def test_interpolate_at_timestamp_before_first_sample(self, interpolator, sample_samples):
        """Test interpolation before the first sample."""
        target_timestamp = 99.5  # Before first sample at 100.0

        result = interpolator.interpolate_at_timestamp(sample_samples, target_timestamp)

        assert result is not None
        assert isinstance(result, InterpolatedSample)
        assert result.timestamp == target_timestamp
        assert result.is_interpolated is True
        # Should use first sample values but with lower confidence
        assert result.confidence < 1.0

    def test_interpolate_at_timestamp_after_last_sample(self, interpolator, sample_samples):
        """Test interpolation after the last sample."""
        target_timestamp = 110.0  # After last sample at 109.0

        result = interpolator.interpolate_at_timestamp(sample_samples, target_timestamp)

        assert result is not None
        assert isinstance(result, InterpolatedSample)
        assert result.timestamp == target_timestamp
        assert result.is_interpolated is True
        # Should use last sample values but with lower confidence
        assert result.confidence < 1.0

    def test_interpolate_at_timestamp_empty_samples(self, interpolator):
        """Test interpolation with no samples."""
        target_timestamp = 105.0

        result = interpolator.interpolate_at_timestamp([], target_timestamp)

        assert result is None

    def test_interpolate_continuous_metrics_linear(self, interpolator, sample_samples):
        """Test that continuous metrics use linear interpolation."""
        target_timestamp = 104.5  # Exactly halfway between 104.0 and 105.0

        result = interpolator.interpolate_at_timestamp(sample_samples, target_timestamp)

        assert result is not None

        # Check that values are linearly interpolated
        # Temperature: between 44.0 and 45.0, should be ~44.5
        assert 44.0 <= result.cpu_temp <= 45.0
        # Power: between 4800 and 5000, should be ~4900
        assert 4800 <= result.power_mw <= 5000

    def test_interpolate_discrete_metrics_step(self, interpolator):
        """Test that discrete metrics use step interpolation."""
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=30,  # Low frequency
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=70,  # High frequency
                cpu_loads=[50, 45, 0, 0],
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=6000
            )
        ]

        # Before midpoint - should have low frequency
        target_before = 100.4
        result_before = interpolator.interpolate_at_timestamp(samples, target_before)
        assert result_before is not None
        assert result_before.gpu_freq_pct == 30

        # After midpoint - should have high frequency
        target_after = 100.6
        result_after = interpolator.interpolate_at_timestamp(samples, target_after)
        assert result_after is not None
        assert result_after.gpu_freq_pct == 70

    def test_interpolate_cpu_loads_step(self, interpolator):
        """Test that CPU loads use step interpolation."""
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[80, 75, 30, 20],
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=6000
            )
        ]

        target = 100.6
        result = interpolator.interpolate_at_timestamp(samples, target)

        assert result is not None
        # Should use first sample's CPU loads (step interpolation)
        assert result.cpu_loads == [10, 5, 0, 0]

    def test_interpolate_different_gap_sizes(self, interpolator):
        """Test interpolation with different gap sizes."""
        small_gap_samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=100.4,  # 400ms gap
                wall_time=datetime.now(),
                ram_used_mb=2100,
                ram_total_mb=7772,
                gpu_freq_pct=55,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=42.0,
                gpu_temp=47.0,
                tj_temp=48.0,
                power_mw=4200
            )
        ]

        large_gap_samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=102.0,  # 2000ms gap
                wall_time=datetime.now(),
                ram_used_mb=2100,
                ram_total_mb=7772,
                gpu_freq_pct=55,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=42.0,
                gpu_temp=47.0,
                tj_temp=48.0,
                power_mw=4200
            )
        ]

        # Small gap should have high confidence
        result_small = interpolator.interpolate_at_timestamp(small_gap_samples, 100.2)
        assert result_small is not None
        assert result_small.confidence > 0.7

        # Large gap should have low confidence
        result_large = interpolator.interpolate_at_timestamp(large_gap_samples, 101.0)
        assert result_large is not None
        assert result_large.confidence < 0.3

    def test_interpolate_custom_max_gap(self, sample_samples):
        """Test interpolation with custom max gap size."""
        interpolator_custom = Interpolator(max_gap_ms=1000.0)  # More lenient

        # Large gap (800ms) that would have low confidence with default
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=100.8,  # 800ms gap
                wall_time=datetime.now(),
                ram_used_mb=2100,
                ram_total_mb=7772,
                gpu_freq_pct=55,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=42.0,
                gpu_temp=47.0,
                tj_temp=48.0,
                power_mw=4200
            )
        ]

        result = interpolator_custom.interpolate_at_timestamp(samples, 100.4)

        assert result is not None
        # With more lenient max_gap, confidence should be higher
        assert result.confidence > 0.5

    def test_interpolate_varying_cpu_cores(self, interpolator):
        """Test interpolation with varying CPU core counts."""
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5],  # 2 cores
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[80, 75, 30, 20],  # 4 cores
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=6000
            )
        ]

        result = interpolator.interpolate_at_timestamp(samples, 100.5)

        assert result is not None
        # Should handle different core counts gracefully
        assert len(result.cpu_loads) == 4  # Should match larger set

    def test_interpolate_progress_calculation(self, interpolator):
        """Test that progress is calculated correctly."""
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.0,
                wall_time=datetime.now(),
                ram_used_mb=2100,
                ram_total_mb=7772,
                gpu_freq_pct=55,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=6000
            )
        ]

        # 25% through the gap (100.25)
        result_25 = interpolator.interpolate_at_timestamp(samples, 100.25)
        assert result_25 is not None
        # Linear interpolation: 40 + (50-40)*0.25 = 42.5
        assert abs(result_25.cpu_temp - 42.5) < 0.1

        # 75% through the gap (100.75)
        result_75 = interpolator.interpolate_at_timestamp(samples, 100.75)
        assert result_75 is not None
        # Linear interpolation: 40 + (50-40)*0.75 = 47.5
        assert abs(result_75.cpu_temp - 47.5) < 0.1


class TestConvenienceFunction:
    """Test suite for convenience functions."""

    def test_interpolate_at_timestamp_convenience(self):
        """Test the convenience interpolate_at_timestamp function."""
        samples = [
            TegraStatsSample(
                timestamp=100.0,
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.0,
                wall_time=datetime.now(),
                ram_used_mb=2100,
                ram_total_mb=7772,
                gpu_freq_pct=55,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=6000
            )
        ]

        result = interpolate_at_timestamp(samples, 100.5)

        assert result is not None
        assert isinstance(result, InterpolatedSample)
        assert result.timestamp == 100.5


if __name__ == "__main__":
    pytest.main([__file__, "-v"])