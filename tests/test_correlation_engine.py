"""
Comprehensive tests for Correlation Engine.
Tests time-series alignment, partial overlaps, weighted averaging, and confidence scoring.
"""

import time
import pytest
from datetime import datetime

from edgewatch.correlation.engine import (
    CorrelationEngine,
    CorrelatedResult,
    correlate_inference
)
from edgewatch.ollama.client import InferenceResult
from edgewatch.tegrastats.parser import TegraStatsSample


class TestCorrelatedResult:
    """Test suite for CorrelatedResult dataclass."""

    def test_full_result(self):
        """Test CorrelatedResult with all fields populated."""
        # Create a mock inference result
        inference = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=1,
            tokens_per_sec_ollama=1.0,
            tokens_per_sec_wall=1.0,
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        # Create mock sample
        sample = TegraStatsSample(
            timestamp=100.5,
            wall_time=datetime.now(),
            ram_used_mb=2048,
            ram_total_mb=7772,
            gpu_freq_pct=45,
            cpu_loads=[23, 15, 0, 0],
            cpu_temp=45.5,
            gpu_temp=52.3,
            tj_temp=53.0,
            power_mw=4521
        )

        result = CorrelatedResult(
            inference=inference,
            hw_samples=[sample],
            avg_gpu_temp=52.3,
            max_gpu_temp=52.3,
            avg_tj_temp=53.0,
            max_tj_temp=53.0,
            avg_gpu_freq_pct=45,
            avg_power_mw=4521,
            peak_power_mw=4521,
            avg_ram_used_mb=2048,
            peak_ram_used_mb=2048,
            sample_count=1,
            interpolation_confidence=0.9,
            window_start=99.5,
            window_end=101.8
        )

        assert result.inference.model == "qwen:4b"
        assert result.sample_count == 1
        assert result.interpolation_confidence == 0.9


class TestCorrelationEngine:
    """Test suite for CorrelationEngine class."""

    @pytest.fixture
    def engine(self):
        """Create a correlation engine instance for testing."""
        return CorrelationEngine(padding_sec=0.5)

    @pytest.fixture
    def sample_inference(self):
        """Create a sample inference result."""
        return InferenceResult(
            model="qwen:4b",
            prompt="Hello",
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
    def sample_samples(self):
        """Create sample hardware samples."""
        base_time = 100.0
        samples = []

        # Create samples every 100ms
        for i in range(15):  # 1.5 seconds of samples
            sample = TegraStatsSample(
                timestamp=base_time + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048 + (i * 50),  # Increasing RAM usage
                ram_total_mb=7772,
                gpu_freq_pct=45 + (i * 2),  # Increasing GPU frequency
                cpu_loads=[20 + i, 15 + i, 0, 0],
                cpu_temp=45.0 + (i * 0.5),  # Increasing temperature
                gpu_temp=52.0 + (i * 0.5),
                tj_temp=53.0 + (i * 0.5),
                power_mw=4500 + (i * 100)  # Increasing power
            )
            samples.append(sample)

        return samples

    def test_correlate_basic(self, engine, sample_inference, sample_samples):
        """Test basic correlation with good samples."""
        result = engine.correlate(sample_inference, sample_samples)

        assert result is not None
        assert isinstance(result, CorrelatedResult)
        assert result.sample_count > 0
        assert result.inference.model == "qwen:4b"
        assert 0.0 <= result.interpolation_confidence <= 1.0

    def test_correlate_timing(self, engine, sample_inference, sample_samples):
        """Test that timing is correctly aligned."""
        result = engine.correlate(sample_inference, sample_samples)

        assert result is not None
        assert result.window_start == sample_inference.t_request_sent - 0.5
        assert result.window_end == sample_inference.t_last_token + 0.5

    def test_correlate_aggregates(self, engine, sample_inference, sample_samples):
        """Test aggregate calculations."""
        result = engine.correlate(sample_inference, sample_samples)

        assert result is not None

        # Check that aggregates are reasonable
        assert result.avg_gpu_temp > 0
        assert result.max_gpu_temp >= result.avg_gpu_temp
        assert result.avg_tj_temp > 0
        assert result.max_tj_temp >= result.avg_tj_temp
        assert result.avg_gpu_freq_pct >= 0
        assert result.avg_power_mw > 0
        assert result.peak_power_mw >= result.avg_power_mw
        assert result.avg_ram_used_mb > 0
        assert result.peak_ram_used_mb >= result.avg_ram_used_mb

    def test_correlate_no_samples(self, engine, sample_inference):
        """Test correlation with no samples."""
        result = engine.correlate(sample_inference, [])

        assert result is None

    def test_correlate_samples_outside_window(self, engine, sample_inference):
        """Test correlation with samples outside the window."""
        # Create samples far outside the window
        samples = [
            TegraStatsSample(
                timestamp=50.0,  # Far before window
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            ),
            TegraStatsSample(
                timestamp=200.0,  # Far after window
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
        ]

        result = engine.correlate(sample_inference, samples)

        # Should return None since no samples in window
        assert result is None

    def test_correlate_incomplete_timing(self, engine, sample_samples):
        """Test correlation with incomplete timing information."""
        # Inference with missing last token time
        incomplete_inference = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=None,  # Missing
            t_last_token=None,  # Missing
            ttft_ms=None,
            total_tokens=1,
            tokens_per_sec_ollama=None,
            tokens_per_sec_wall=None,
            inference_duration_sec=None,
            eval_duration_ns=None,
            response_chunks=1,
            bytes_received=50
        )

        result = engine.correlate(incomplete_inference, sample_samples)

        assert result is None

    def test_correlate_single_sample(self, engine, sample_inference):
        """Test correlation with only one sample in window."""
        single_sample = [
            TegraStatsSample(
                timestamp=100.5,  # Within window
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
        ]

        result = engine.correlate(sample_inference, single_sample)

        assert result is not None
        assert result.sample_count == 1
        # Confidence should be lower with fewer samples
        assert result.interpolation_confidence < 0.4

    def test_correlate_many_samples(self, engine, sample_inference):
        """Test correlation with many samples (high confidence)."""
        # Create many samples
        samples = []
        for i in range(30):  # 3 seconds of samples
            sample = TegraStatsSample(
                timestamp=99.0 + (i * 0.1),  # From 99.0 to 102.0
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
            samples.append(sample)

        result = engine.correlate(sample_inference, samples)

        assert result is not None
        assert result.sample_count >= 15  # At least samples in padded window
        # Confidence should be high with many samples
        assert result.interpolation_confidence > 0.7

    def test_correlate_edge_samples(self, engine, sample_inference):
        """Test correlation with samples at window boundaries."""
        samples = [
            TegraStatsSample(
                timestamp=99.5,  # Exactly at window start
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=40,
                cpu_loads=[15, 10, 0, 0],
                cpu_temp=40.0,
                gpu_temp=50.0,
                tj_temp=51.0,
                power_mw=4000
            ),
            TegraStatsSample(
                timestamp=101.8,  # Exactly at window end
                wall_time=datetime.now(),
                ram_used_mb=2500,
                ram_total_mb=7772,
                gpu_freq_pct=50,
                cpu_loads=[25, 20, 0, 0],
                cpu_temp=50.0,
                gpu_temp=55.0,
                tj_temp=56.0,
                power_mw=5000
            )
        ]

        result = engine.correlate(sample_inference, samples)

        assert result is not None
        assert result.sample_count == 2

    def test_correlate_weighted_averaging(self, engine):
        """Test that weighted averaging works correctly."""
        # Create inference centered in time
        inference = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.2,
            t_last_token=101.0,
            ttft_ms=200.0,
            total_tokens=1,
            tokens_per_sec_ollama=1.0,
            tokens_per_sec_wall=1.0,
            inference_duration_sec=1.0,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        # Create samples with different weights (overlap times)
        samples = [
            TegraStatsSample(
                timestamp=99.6,  # Small overlap (0.1s)
                wall_time=datetime.now(),
                ram_used_mb=2000,
                ram_total_mb=7772,
                gpu_freq_pct=30,
                cpu_loads=[10, 5, 0, 0],
                cpu_temp=40.0,
                gpu_temp=45.0,
                tj_temp=46.0,
                power_mw=3000
            ),
            TegraStatsSample(
                timestamp=100.5,  # Large overlap (0.5s) - should have more weight
                wall_time=datetime.now(),
                ram_used_mb=3000,
                ram_total_mb=7772,
                gpu_freq_pct=60,
                cpu_loads=[30, 25, 0, 0],
                cpu_temp=55.0,
                gpu_temp=60.0,
                tj_temp=61.0,
                power_mw=6000
            )
        ]

        result = engine.correlate(inference, samples)

        assert result is not None
        # Average should be closer to sample 2 due to larger overlap weight
        assert result.avg_gpu_temp > 50.0  # Closer to 60 than 45

    def test_correlate_varying_padding(self, sample_inference, sample_samples):
        """Test correlation with different padding values."""
        # Small padding
        engine_small = CorrelationEngine(padding_sec=0.1)
        result_small = engine_small.correlate(sample_inference, sample_samples)

        # Large padding
        engine_large = CorrelationEngine(padding_sec=2.0)
        result_large = engine_large.correlate(sample_inference, sample_samples)

        if result_small and result_large:
            # Larger padding should include more samples
            assert result_large.sample_count >= result_small.sample_count

    def test_correlate_temperature_tracking(self, engine, sample_inference):
        """Test temperature tracking through samples."""
        # Create samples with thermal ramp-up
        samples = []
        for i in range(20):
            temp_ramp = 45.0 + (i * 1.0)  # Temperature increases by 1°C each sample
            sample = TegraStatsSample(
                timestamp=99.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45 + (i * 1),  # Frequency also increases
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=temp_ramp - 5.0,
                gpu_temp=temp_ramp,
                tj_temp=temp_ramp + 1.0,
                power_mw=4500 + (i * 100)
            )
            samples.append(sample)

        result = engine.correlate(sample_inference, samples)

        assert result is not None
        # Check temperature tracking
        assert result.max_gpu_temp > result.avg_gpu_temp
        assert result.max_tj_temp > result.avg_tj_temp
        # TJ temp should be highest
        assert result.max_tj_temp > result.max_gpu_temp

    def test_correlate_power_tracking(self, engine, sample_inference):
        """Test power consumption tracking."""
        # Create samples with varying power
        samples = []
        for i in range(20):
            power_variance = 4000 + (i * 150)  # Power increases by 150mW each sample
            sample = TegraStatsSample(
                timestamp=99.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45 + (i * 2),
                cpu_loads=[20 + i, 15 + i, 0, 0],
                cpu_temp=45.0 + (i * 0.3),
                gpu_temp=52.0 + (i * 0.3),
                tj_temp=53.0 + (i * 0.3),
                power_mw=power_variance
            )
            samples.append(sample)

        result = engine.correlate(sample_inference, samples)

        assert result is not None
        # Check power tracking
        assert result.peak_power_mw > result.avg_power_mw
        assert result.avg_power_mw > 4000

    def test_correlate_confidence_scoring(self, engine, sample_inference):
        """Test confidence scoring with different sample scenarios."""
        # Low confidence: few samples
        few_samples = [
            TegraStatsSample(
                timestamp=100.5,
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
        ]
        result_low = engine.correlate(sample_inference, few_samples)
        assert result_low is not None
        assert result_low.interpolation_confidence < 0.4

        # High confidence: many samples
        many_samples = []
        for i in range(25):
            sample = TegraStatsSample(
                timestamp=99.0 + (i * 0.1),
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
            many_samples.append(sample)
        result_high = engine.correlate(sample_inference, many_samples)
        assert result_high is not None
        assert result_high.interpolation_confidence > 0.7


class TestConvenienceFunction:
    """Test suite for convenience functions."""

    def test_correlate_inference_convenience(self):
        """Test the convenience correlate_inference function."""
        inference = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=1,
            tokens_per_sec_ollama=1.0,
            tokens_per_sec_wall=1.0,
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        samples = [
            TegraStatsSample(
                timestamp=100.5,
                wall_time=datetime.now(),
                ram_used_mb=2048,
                ram_total_mb=7772,
                gpu_freq_pct=45,
                cpu_loads=[20, 15, 0, 0],
                cpu_temp=45.0,
                gpu_temp=52.0,
                tj_temp=53.0,
                power_mw=4500
            )
        ]

        result = correlate_inference(inference, samples)

        assert result is not None
        assert isinstance(result, CorrelatedResult)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])