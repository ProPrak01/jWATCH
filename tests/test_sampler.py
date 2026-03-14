"""
Comprehensive tests for Tegrastats Sampler.
Tests circular buffer, time window queries, mock subprocess, and graceful degradation.
"""

import asyncio
import pytest
import time

from edgewatch.tegrastats.sampler import TegrastatsSampler, sample_tegrastats
from edgewatch.tegrastats.parser import TegraStatsSample


class TestTegrastatsSampler:
    """Test suite for TegrastatsSampler class."""

    @pytest.fixture
    async def sampler(self):
        """Create a sampler instance for testing."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)
        await sampler.start(mock_mode=True)
        yield sampler
        await sampler.stop()

    @pytest.mark.asyncio
    async def test_start_stop(self):
        """Test starting and stopping the sampler."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)

        assert not sampler.is_running()
        assert sampler.get_sample_count() == 0

        await sampler.start(mock_mode=True)
        assert sampler.is_running()

        await sampler.stop()
        assert not sampler.is_running()

    @pytest.mark.asyncio
    async def test_mock_mode_sampling(self):
        """Test sampling in mock mode generates realistic data."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)
        await sampler.start(mock_mode=True)

        # Wait for some samples to be collected
        await asyncio.sleep(0.5)

        await sampler.stop()

        stats = sampler.get_statistics()
        assert stats["samples_collected"] > 0
        assert stats["samples_in_buffer"] > 0
        assert stats["mock_mode"] is True
        assert stats["parse_success_rate"] > 0.9  # Should have very high success rate

    @pytest.mark.asyncio
    async def test_circular_buffer(self):
        """Test that circular buffer respects max size."""
        buffer_size = 10
        sampler = TegrastatsSampler(buffer_size=buffer_size, interval_ms=10)
        await sampler.start(mock_mode=True)

        # Wait for more samples than buffer size
        await asyncio.sleep(0.3)  # Should generate ~30 samples with 10ms interval

        await sampler.stop()

        # Buffer should not exceed max size
        assert sampler.get_sample_count() <= buffer_size

    @pytest.mark.asyncio
    async def test_get_samples_in_window(self, sampler):
        """Test getting samples within a time window."""
        # Wait for samples to be collected
        await asyncio.sleep(0.3)

        # Get current time range
        all_samples = list(sampler._samples)
        if not all_samples:
            pytest.skip("No samples collected")

        # Test getting samples in the middle of the time range
        t_start = all_samples[len(all_samples) // 3].timestamp
        t_end = all_samples[2 * len(all_samples) // 3].timestamp

        samples_in_window = sampler.get_samples_in_window(t_start, t_end)

        # Verify all returned samples are within the window
        for sample in samples_in_window:
            assert t_start <= sample.timestamp <= t_end

        # Verify samples are sorted
        timestamps = [sample.timestamp for sample in samples_in_window]
        assert timestamps == sorted(timestamps)

    @pytest.mark.asyncio
    async def test_get_latest_sample(self, sampler):
        """Test getting the most recent sample."""
        await asyncio.sleep(0.3)

        latest_sample = sampler.get_latest_sample()
        assert latest_sample is not None

        # Verify it's actually the latest
        all_samples = list(sampler._samples)
        if all_samples:
            assert latest_sample.timestamp == all_samples[-1].timestamp

    @pytest.mark.asyncio
    async def test_empty_buffer_queries(self):
        """Test queries on empty buffer."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)
        await sampler.start(mock_mode=True)

        # Query before any samples collected
        samples_in_window = sampler.get_samples_in_window(0, 1)
        assert samples_in_window == []

        latest_sample = sampler.get_latest_sample()
        assert latest_sample is None

        await sampler.stop()

    @pytest.mark.asyncio
    async def test_statistics(self, sampler):
        """Test sampler statistics."""
        await asyncio.sleep(0.3)

        stats = sampler.get_statistics()

        assert "samples_collected" in stats
        assert "samples_in_buffer" in stats
        assert "parse_errors" in stats
        assert "is_running" in stats
        assert "mock_mode" in stats
        assert "buffer_size" in stats
        assert "interval_ms" in stats
        assert "parse_success_rate" in stats

        assert stats["buffer_size"] == 100
        assert stats["interval_ms"] == 50
        assert stats["mock_mode"] is True
        assert stats["is_running"] is True
        assert stats["samples_collected"] > 0

    @pytest.mark.asyncio
    async def test_clear_buffer(self, sampler):
        """Test clearing the buffer."""
        await asyncio.sleep(0.3)

        # Verify samples exist
        assert sampler.get_sample_count() > 0

        # Clear buffer
        sampler.clear_buffer()

        # Verify buffer is empty
        assert sampler.get_sample_count() == 0
        stats = sampler.get_statistics()
        assert stats["samples_collected"] == 0
        assert stats["parse_errors"] == 0

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with TegrastatsSampler(buffer_size=100, interval_ms=50) as sampler:
            assert sampler.is_running()
            await asyncio.sleep(0.2)
            assert sampler.get_sample_count() > 0

        # Sampler should be stopped after exiting context
        assert not sampler.is_running()

    @pytest.mark.asyncio
    async def test_restart_sampler(self):
        """Test stopping and restarting the sampler."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)

        # First run
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.2)
        first_count = sampler.get_sample_count()
        await sampler.stop()

        # Second run
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.2)
        second_count = sampler.get_sample_count()
        await sampler.stop()

        # Should collect samples in both runs
        assert first_count > 0
        assert second_count > 0

    @pytest.mark.asyncio
    async def test_different_intervals(self):
        """Test sampler with different interval settings."""
        # Fast sampling
        fast_sampler = TegrastatsSampler(buffer_size=100, interval_ms=10)
        await fast_sampler.start(mock_mode=True)
        await asyncio.sleep(0.2)
        fast_count = fast_sampler.get_sample_count()
        await fast_sampler.stop()

        # Slow sampling
        slow_sampler = TegrastatsSampler(buffer_size=100, interval_ms=100)
        await slow_sampler.start(mock_mode=True)
        await asyncio.sleep(0.2)
        slow_count = slow_sampler.get_sample_count()
        await slow_sampler.stop()

        # Fast sampler should have collected more samples
        assert fast_count > slow_count

    @pytest.mark.asyncio
    async def test_partial_time_window(self, sampler):
        """Test time window queries that include partial overlaps."""
        await asyncio.sleep(0.3)

        all_samples = list(sampler._samples)
        if not all_samples:
            pytest.skip("No samples collected")

        # Get middle time range
        middle_idx = len(all_samples) // 2
        t_start = all_samples[middle_idx].timestamp - 0.05  # 50ms before middle sample
        t_end = all_samples[middle_idx].timestamp + 0.05  # 50ms after middle sample

        samples_in_window = sampler.get_samples_in_window(t_start, t_end)

        # Should have at least the middle sample
        assert len(samples_in_window) >= 1

        # Should include samples from around the middle
        timestamps = [sample.timestamp for sample in samples_in_window]
        assert t_start <= all_samples[middle_idx].timestamp <= t_end

    @pytest.mark.asyncio
    async def test_graceful_degradation_no_tegrastats(self):
        """Test graceful degradation when tegrastats is not available."""
        # Create sampler (will try to start real tegrastats first, then fall back to mock)
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=50)

        # Start without explicit mock mode (will detect missing tegrastats)
        await sampler.start(mock_mode=False)  # Should fall back to mock automatically

        # Should still be running (in mock mode)
        assert sampler.is_running()

        await asyncio.sleep(0.2)

        # Should have collected samples
        assert sampler.get_sample_count() > 0

        await sampler.stop()


class TestConvenienceFunction:
    """Test suite for convenience functions."""

    @pytest.mark.asyncio
    async def test_sample_tegrastats_function(self):
        """Test the convenience sample_tegrastats function."""
        samples = await sample_tegrastats(duration_sec=0.3, interval_ms=50, mock_mode=True)

        assert isinstance(samples, list)
        assert len(samples) > 0

        # Verify all samples are TegraStatsSample instances
        for sample in samples:
            assert isinstance(sample, TegraStatsSample)

        # Verify timestamps are reasonable
        if len(samples) >= 2:
            time_diff = samples[-1].timestamp - samples[0].timestamp
            assert 0.2 <= time_diff <= 0.4  # Should be around 0.3 seconds

    @pytest.mark.asyncio
    async def test_sample_tegrastats_different_durations(self):
        """Test sampling for different durations."""
        short_samples = await sample_tegrastats(duration_sec=0.1, interval_ms=50, mock_mode=True)
        long_samples = await sample_tegrastats(duration_sec=0.3, interval_ms=50, mock_mode=True)

        # Longer duration should produce more samples
        assert len(long_samples) > len(short_samples)


class TestEdgeCases:
    """Test suite for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_zero_interval(self):
        """Test sampler with zero interval (should use default)."""
        sampler = TegrastatsSampler(buffer_size=100, interval_ms=0)
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.1)
        await sampler.stop()

        # Should still work (using default or handling gracefully)
        stats = sampler.get_statistics()
        assert stats["is_running"] is False  # Should be stopped

    @pytest.mark.asyncio
    async def test_very_small_buffer(self):
        """Test sampler with very small buffer size."""
        sampler = TegrastatsSampler(buffer_size=2, interval_ms=10)
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.1)
        await sampler.stop()

        # Buffer should not exceed size
        assert sampler.get_sample_count() <= 2

    @pytest.mark.asyncio
    async def test_very_large_buffer(self):
        """Test sampler with very large buffer size."""
        sampler = TegrastatsSampler(buffer_size=10000, interval_ms=10)
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.1)
        await sampler.stop()

        # Should handle large buffer without issues
        assert sampler.get_sample_count() < 10000

    @pytest.mark.asyncio
    async def test_concurrent_time_window_queries(self, sampler):
        """Test multiple concurrent time window queries."""
        await asyncio.sleep(0.3)

        all_samples = list(sampler._samples)
        if not all_samples:
            pytest.skip("No samples collected")

        # Create multiple concurrent queries
        tasks = []
        for i in range(5):
            t_start = all_samples[i].timestamp
            t_end = all_samples[min(i + 3, len(all_samples) - 1)].timestamp
            tasks.append(sampler.get_samples_in_window(t_start, t_end))

        # Execute all queries concurrently
        results = await asyncio.gather(*tasks)

        # All queries should succeed
        for result in results:
            assert isinstance(result, list)

    @pytest.mark.asyncio
    async def test_start_when_already_running(self, sampler):
        """Test starting sampler when already running."""
        # Already running from fixture
        assert sampler.is_running()

        # Try to start again (should be idempotent)
        await sampler.start(mock_mode=True)

        # Should still be running
        assert sampler.is_running()

        # Should have samples
        await asyncio.sleep(0.1)
        assert sampler.get_sample_count() > 0


class TestPerformance:
    """Test suite for performance characteristics."""

    @pytest.mark.asyncio
    async def test_memory_usage_stability(self):
        """Test that memory usage remains stable over time."""
        sampler = TegrastatsSampler(buffer_size=1000, interval_ms=10)

        # Collect initial statistics
        await sampler.start(mock_mode=True)
        await asyncio.sleep(0.2)
        initial_count = sampler.get_sample_count()

        # Collect more data
        await asyncio.sleep(0.2)
        final_count = sampler.get_sample_count()

        await sampler.stop()

        # Memory should be stable (buffer size is fixed)
        assert final_count <= 1000
        assert final_count > initial_count  # Should have more samples

    @pytest.mark.asyncio
    async def test_high_frequency_sampling(self):
        """Test high-frequency sampling performance."""
        sampler = TegrastatsSampler(buffer_size=1000, interval_ms=1)
        await sampler.start(mock_mode=True)

        start_time = time.time()
        await asyncio.sleep(0.1)
        end_time = time.time()

        await sampler.stop()

        stats = sampler.get_statistics()
        actual_duration = end_time - start_time

        # Should have collected approximately 100 samples in 0.1s at 1ms interval
        expected_samples = int(actual_duration / 0.001)
        actual_samples = stats["samples_collected"]

        # Allow for some variance in timing
        assert actual_samples >= expected_samples * 0.5  # At least 50% of expected


if __name__ == "__main__":
    pytest.main([__file__, "-v"])