"""
Async tegrastats sampler that continuously reads hardware metrics.
Uses circular buffer for bounded memory and provides time-window queries.
"""

import asyncio
import time
from collections import deque
from typing import List, Optional, Tuple

from edgewatch.tegrastats.parser import TegraStatsSample, TegrastatsParser


class TegrastatsSampler:
    """
    Continuously samples hardware metrics from tegrastats in background.

    Features:
    - Async subprocess management for non-blocking operation
    - Circular buffer for bounded memory usage
    - Precise monotonic timestamps for all samples
    - Time-window queries for correlation engine
    - Graceful degradation when tegrastats unavailable
    """

    def __init__(self, buffer_size: int = 10000, interval_ms: int = 100):
        """
        Initialize the tegrastats sampler.

        Args:
            buffer_size: Maximum number of samples to keep in circular buffer
            interval_ms: Sampling interval in milliseconds (default 100ms)
        """
        self.buffer_size = buffer_size
        self.interval_ms = interval_ms

        # Thread-safe circular buffer for samples
        self._samples: deque[TegraStatsSample] = deque(maxlen=buffer_size)

        # Parser for processing raw tegrastats output
        self._parser = TegrastatsParser()

        # Async subprocess management
        self._process: Optional[asyncio.subprocess.Process] = None
        self._reader_task: Optional[asyncio.Task] = None
        self._is_running = False

        # Statistics
        self._samples_collected = 0
        self._parse_errors = 0

        # Mock mode for testing
        self._mock_mode = False
        self._mock_generator = None

    async def start(self, mock_mode: bool = False) -> None:
        """
        Start sampling hardware metrics in background.

        Args:
            mock_mode: If True, use mock data generator instead of real tegrastats
        """
        if self._is_running:
            return

        self._is_running = True
        self._mock_mode = mock_mode

        if mock_mode:
            # Start mock data generator
            from edgewatch.utils.mocks import create_tegrastats_mock
            mock = create_tegrastats_mock()
            mock.config.interval_ms = self.interval_ms
            self._mock_generator = mock
            self._reader_task = asyncio.create_task(self._read_mock_data())
        else:
            # Start real tegrastats subprocess
            await self._start_tegrastats_subprocess()

    async def _start_tegrastats_subprocess(self) -> None:
        """Start tegrastats as an async subprocess."""
        try:
            self._process = await asyncio.create_subprocess_exec(
                "tegrastats",
                "--interval",
                str(self.interval_ms),
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            self._reader_task = asyncio.create_task(self._read_tegrastats_data())
        except FileNotFoundError:
            # tegrastats not available, switch to mock mode
            print("Warning: tegrastats not found, using mock data")
            self._mock_mode = True
            from edgewatch.utils.mocks import create_tegrastats_mock
            mock = create_tegrastats_mock()
            mock.config.interval_ms = self.interval_ms
            self._mock_generator = mock
            self._reader_task = asyncio.create_task(self._read_mock_data())
        except Exception as e:
            print(f"Error starting tegrastats: {e}")
            raise

    async def _read_tegrastats_data(self) -> None:
        """Read and parse tegrastats output continuously."""
        if not self._process or not self._process.stdout:
            return

        try:
            while self._is_running:
                # Read a line from tegrastats
                line_bytes = await self._process.stdout.readline()

                if not line_bytes:
                    # End of stream
                    break

                line = line_bytes.decode('utf-8').strip()
                if not line:
                    continue

                # Parse the line and store sample
                await self._process_line(line)

        except Exception as e:
            print(f"Error reading tegrastats data: {e}")
        finally:
            await self._cleanup()

    async def _read_mock_data(self) -> None:
        """Generate and process mock tegrastats data."""
        if not self._mock_generator:
            return

        try:
            while self._is_running:
                # Generate a mock sample
                line = self._mock_generator.generate_sample()

                # Parse and store the sample
                await self._process_line(line)

                # Wait for the specified interval
                await asyncio.sleep(self.interval_ms / 1000.0)

        except Exception as e:
            print(f"Error reading mock data: {e}")

    async def _process_line(self, line: str) -> None:
        """
        Process a single tegrastats line.

        Args:
            line: Raw tegrastats output line
        """
        # Parse the line
        sample = self._parser.parse(line)

        if sample:
            # Store sample in circular buffer
            self._samples.append(sample)
            self._samples_collected += 1
        else:
            self._parse_errors += 1

    async def stop(self) -> None:
        """Stop sampling and cleanup resources."""
        if not self._is_running:
            return

        self._is_running = False

        # Cancel reader task if running
        if self._reader_task and not self._reader_task.done():
            self._reader_task.cancel()
            try:
                await self._reader_task
            except asyncio.CancelledError:
                pass

        await self._cleanup()

    async def _cleanup(self) -> None:
        """Cleanup subprocess resources."""
        if self._process:
            try:
                self._process.terminate()
                await self._process.wait()
            except Exception:
                self._process.kill()
            finally:
                self._process = None

        self._reader_task = None

    def get_samples_in_window(self, t_start: float, t_end: float) -> List[TegraStatsSample]:
        """
        Get all samples within a time window.

        Args:
            t_start: Start time (time.monotonic())
            t_end: End time (time.monotonic())

        Returns:
            List of samples with timestamps in [t_start, t_end]
        """
        samples_in_window = []

        for sample in self._samples:
            if t_start <= sample.timestamp <= t_end:
                samples_in_window.append(sample)

        # Sort by timestamp to ensure chronological order
        samples_in_window.sort(key=lambda s: s.timestamp)

        return samples_in_window

    def get_latest_sample(self) -> Optional[TegraStatsSample]:
        """
        Get the most recent sample.

        Returns:
            Latest sample if available, None otherwise
        """
        if not self._samples:
            return None
        return self._samples[-1]

    def get_sample_count(self) -> int:
        """Get current number of samples in buffer."""
        return len(self._samples)

    def get_statistics(self) -> dict:
        """
        Get sampler statistics.

        Returns:
            Dictionary with sampler statistics
        """
        return {
            "samples_collected": self._samples_collected,
            "samples_in_buffer": len(self._samples),
            "parse_errors": self._parse_errors,
            "is_running": self._is_running,
            "mock_mode": self._mock_mode,
            "buffer_size": self.buffer_size,
            "interval_ms": self.interval_ms,
            "parse_success_rate": (
                self._samples_collected / (self._samples_collected + self._parse_errors)
                if (self._samples_collected + self._parse_errors) > 0 else 0.0
            )
        }

    def clear_buffer(self) -> None:
        """Clear all samples from the buffer."""
        self._samples.clear()
        self._samples_collected = 0
        self._parse_errors = 0
        self._parser.reset_parse_errors()

    def is_running(self) -> bool:
        """Check if sampler is currently running."""
        return self._is_running

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


async def sample_tegrastats(duration_sec: float = 10.0,
                            interval_ms: int = 100,
                            mock_mode: bool = False) -> List[TegraStatsSample]:
    """
    Convenience function to sample tegrastats for a specified duration.

    Args:
        duration_sec: How long to sample in seconds
        interval_ms: Sampling interval in milliseconds
        mock_mode: If True, use mock data

    Returns:
        List of samples collected during the sampling period
    """
    async with TegrastatsSampler(interval_ms=interval_ms) as sampler:
        await asyncio.sleep(duration_sec)

        # Get all samples
        all_samples = list(sampler._samples)
        return all_samples
