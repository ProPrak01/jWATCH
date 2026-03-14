"""
Comprehensive mocking framework for edgewatch components.
Enables extensive testing without constant hardware access.
"""

import asyncio
import json
import random
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime
from typing import AsyncIterator, Generator, List, Optional, Tuple

import httpx


@dataclass
class TegrastatsMockConfig:
    """Configuration for tegrastats mock behavior."""
    interval_ms: int = 100
    base_gpu_temp: float = 45.0
    base_cpu_temp: float = 40.0
    base_tj_temp: float = 50.0
    base_power_mw: int = 4500
    base_ram_mb: int = 2048
    total_ram_mb: int = 7772
    gpu_freq_pct: int = 45
    simulate_throttling: bool = False
    throttle_start_sample: int = 50
    throttle_duration: int = 20


@dataclass
class OllamaMockConfig:
    """Configuration for Ollama mock behavior."""
    model: str = "qwen:4b"
    prompt: str = "Explain gravity in 100 words"
    tokens_per_sec: float = 25.0
    ttft_ms: float = 300.0
    total_tokens: int = 100
    simulate_network_delay: bool = False
    network_delay_ms: Tuple[int, int] = (10, 50)


class MockTegrastatsSubprocess:
    """
    Mock subprocess that simulates tegrastats output.
    Generates realistic hardware metrics with optional throttling simulation.
    """

    def __init__(self, config: TegrastatsMockConfig):
        self.config = config
        self._samples_generated = 0
        self._current_gpu_temp = config.base_gpu_temp
        self._current_cpu_temp = config.base_cpu_temp
        self._current_tj_temp = config.base_tj_temp
        self._current_power_mw = config.base_power_mw
        self._ram_mb = config.base_ram_mb
        self._gpu_freq_pct = config.gpu_freq_pct

    def generate_sample(self) -> str:
        """Generate a single tegrastats sample line."""
        self._samples_generated += 1

        # Simulate thermal behavior
        self._simulate_thermal_dynamics()

        # Simulate throttling if configured
        if (self.config.simulate_throttling and
            self.config.throttle_start_sample <= self._samples_generated <
            self.config.throttle_start_sample + self.config.throttle_duration):
            self._apply_throttling()
        else:
            self._recover_from_throttling()

        # Generate realistic CPU loads for 4 cores
        cpu_loads = [
            random.randint(10, 30),
            random.randint(15, 35),
            random.randint(0, 10),
            random.randint(0, 10)
        ]
        cpu_load_str = ",".join([f"{load}%@{random.randint(1400, 1500)}"
                                if load > 0 else "off"
                                for load in cpu_loads])

        # Build sample line matching tegrastats format
        sample_time = datetime.now().strftime("%m-%d-%Y %H:%M:%S")
        sample = (
            f"{sample_time} RAM {self._ram_mb}/{self.config.total_ram_mb}MB "
            f"(lfb 256x4MB) SWAP 0/3886MB "
            f"CPU [{cpu_load_str}] EMC_FREQ 0% GPC_FREQ {self._gpu_freq_pct}% "
            f"CPU@{self._current_cpu_temp:.1f}C GPU@{self._current_gpu_temp:.1f}C "
            f"tj@{self._current_tj_temp:.1f}C VDD_IN {self._current_power_mw}mW"
        )
        return sample

    def _simulate_thermal_dynamics(self):
        """Simulate gradual temperature changes."""
        # Temperature tends to rise and plateau
        if self._current_gpu_temp < 75.0:
            self._current_gpu_temp += random.uniform(0.1, 0.3)
        if self._current_cpu_temp < 70.0:
            self._current_cpu_temp += random.uniform(0.1, 0.3)
        if self._current_tj_temp < 80.0:
            self._current_tj_temp += random.uniform(0.1, 0.4)

        # Power consumption correlates with temperature
        self._current_power_mw = int(self.config.base_power_mw +
                                     (self._current_tj_temp - self.config.base_tj_temp) * 100)

    def _apply_throttling(self):
        """Apply thermal throttling effects."""
        # Dramatic GPU frequency drop
        self._gpu_freq_pct = random.randint(15, 25)
        # Temperature spikes during throttling
        self._current_tj_temp += random.uniform(1.0, 2.0)
        # Power may spike then drop
        self._current_power_mw = random.randint(6000, 7000)

    def _recover_from_throttling(self):
        """Recover from throttling effects."""
        # Gradual frequency recovery
        if self._gpu_freq_pct < self.config.gpu_freq_pct:
            self._gpu_freq_pct += random.randint(2, 5)

        # Temperature stabilization
        if self._current_tj_temp > self.config.base_tj_temp + 10:
            self._current_tj_temp -= random.uniform(0.5, 1.5)

        # Power returns to normal
        self._current_power_mw = max(self.config.base_power_mw,
                                    self._current_power_mw - 100)

    async def create_subprocess_mock(self) -> "asyncio.subprocess.Process":
        """Create a mock subprocess object."""
        # This would normally be implemented with a proper mock
        # For now, we'll return a simple generator-based simulation
        return self._create_mock_process()

    def _create_mock_process(self):
        """Create a mock process object."""
        class MockProcess:
            def __init__(self, generator: Generator[str, None, None]):
                self._generator = generator
                self.returncode = None
                self.stdin = None

            async def communicate(self) -> Tuple[bytes, bytes]:
                return (b"", b"")

            async def wait(self) -> int:
                return 0

            def kill(self):
                pass

            def terminate(self):
                pass

        return MockProcess(self._generate_lines())

    def _generate_lines(self) -> Generator[str, None, None]:
        """Generate tegrastats output lines."""
        while True:
            yield self.generate_sample()
            time.sleep(self.config.interval_ms / 1000.0)


class AsyncLineGenerator:
    """Async generator that yields tegrastats lines with proper timing."""

    def __init__(self, mock: MockTegrastatsSubprocess):
        self._mock = mock

    async def lines(self) -> AsyncIterator[str]:
        """Yield tegrastats lines asynchronously."""
        while True:
            yield self._mock.generate_sample()
            await asyncio.sleep(self._mock.config.interval_ms / 1000.0)


class MockOllamaResponse:
    """
    Mock Ollama streaming response with realistic timing.
    Generates JSON responses that match Ollama's streaming format.
    """

    def __init__(self, config: OllamaMockConfig):
        self.config = config
        self._tokens_remaining = config.total_tokens
        self._first_token_sent = False
        self._request_start_time = None

    def set_request_start_time(self, timestamp: float):
        """Set the request start time for accurate timing."""
        self._request_start_time = timestamp

    def generate_response_chunks(self) -> Generator[str, None, None]:
        """Generate Ollama response chunks."""
        # Simulate TTFT (Time To First Token)
        if not self._first_token_sent:
            time.sleep(self.config.ttft_ms / 1000.0)
            self._first_token_sent = True

        # Generate token chunks
        while self._tokens_remaining > 0:
            tokens_to_send = min(5, self._tokens_remaining)
            self._tokens_remaining -= tokens_to_send

            # Generate realistic response text
            response_text = " ".join(["token"] * tokens_to_send)
            chunk = {
                "response": response_text,
                "done": False
            }
            yield json.dumps(chunk)

            # Simulate token generation rate
            time.sleep(1.0 / (self.config.tokens_per_sec / tokens_to_send))

            # Simulate network delay if configured
            if self.config.simulate_network_delay:
                delay = random.randint(*self.config.network_delay_ms) / 1000.0
                time.sleep(delay)

        # Send final chunk with metadata
        eval_duration_ns = int(self.config.total_tokens / self.config.tokens_per_sec * 1e9)
        final_chunk = {
            "response": "",
            "done": True,
            "eval_count": self.config.total_tokens,
            "eval_duration": eval_duration_ns,
            "model": self.config.model
        }
        yield json.dumps(final_chunk)


class MockOllamaStreamingResponse:
    """Mock httpx streaming response for Ollama."""

    def __init__(self, mock_response: MockOllamaResponse):
        self._mock_response = mock_response
        self._request_start_time = time.monotonic()
        self._mock_response.set_request_start_time(self._request_start_time)

    def __aiter__(self) -> AsyncIterator[bytes]:
        return self._async_iterator()

    async def _async_iterator(self) -> AsyncIterator[bytes]:
        """Yield response chunks asynchronously."""
        generator = self._mock_response.generate_response_chunks()

        for chunk in generator:
            # Convert to bytes and add newline
            yield (chunk + "\n").encode('utf-8')
            # Simulate async behavior
            await asyncio.sleep(0.01)

    async def aread(self) -> bytes:
        """Read entire response (not used for streaming)."""
        return b""


class MockOllamaClient:
    """
    Mock Ollama HTTP client that simulates realistic streaming responses.
    Can be used to replace httpx.Client in tests.
    """

    def __init__(self, config: Optional[OllamaMockConfig] = None):
        self.config = config or OllamaMockConfig()
        self._requests_made = []

    async def post(self, url: str, json_data: dict, timeout: float = 300.0) -> MockOllamaStreamingResponse:
        """
        Simulate POST request to Ollama API.

        Args:
            url: The API endpoint URL
            json_data: Request body with model, prompt, stream=True
            timeout: Request timeout in seconds
        """
        # Record request for verification
        self._requests_made.append({
            "url": url,
            "data": json_data,
            "timestamp": time.monotonic()
        })

        # Update config from request if provided
        if "model" in json_data:
            self.config.model = json_data["model"]
        if "prompt" in json_data:
            self.config.prompt = json_data["prompt"]

        # Create mock response
        mock_response = MockOllamaResponse(self.config)
        return MockOllamaStreamingResponse(mock_response)

    def get_requests(self) -> List[dict]:
        """Get list of requests made (for test verification)."""
        return self._requests_made

    def clear_requests(self):
        """Clear request history."""
        self._requests_made = []


def create_tegrastats_mock(config: Optional[TegrastatsMockConfig] = None) -> MockTegrastatsSubprocess:
    """Create a tegrastats mock with default or custom configuration."""
    return MockTegrastatsSubprocess(config or TegrastatsMockConfig())


def create_ollama_mock(config: Optional[OllamaMockConfig] = None) -> MockOllamaClient:
    """Create an Ollama mock with default or custom configuration."""
    return MockOllamaClient(config)


# Sample tegrastats outputs for testing
SAMPLE_TEGRASTATS_OUTPUTS = [
    # Normal operation
    "04-01-2025 12:00:00 RAM 2048/7772MB (lfb 512x4MB) SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] EMC_FREQ 0% GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW",
    # High GPU usage
    "04-01-2025 12:00:01 RAM 3072/7772MB (lfb 256x4MB) SWAP 0/3886MB CPU [85%@1475,78%@1465,off,off] EMC_FREQ 15% GPC_FREQ 89% CPU@58.2C GPU@72.1C tj@76.5C VDD_IN 6892mW",
    # Memory pressure
    "04-01-2025 12:00:02 RAM 6500/7772MB (lfb 128x4MB) SWAP 256/3886MB CPU [45%@1435,38%@1425,15%@1415,off] EMC_FREQ 8% GPC_FREQ 67% CPU@51.3C GPU@62.8C tj@65.2C VDD_IN 5432mW",
    # Thermal throttling
    "04-01-2025 12:00:03 RAM 4096/7772MB (lfb 256x4MB) SWAP 128/3886MB CPU [67%@1455,72%@1450,off,off] EMC_FREQ 12% GPC_FREQ 23% CPU@64.8C GPU@78.2C tj@85.1C VDD_IN 6234mW",
    # Missing some fields (different JetPack version)
    "04-01-2025 12:00:04 RAM 2048/7772MB (lfb 512x4MB) CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW",
]


# Sample Ollama response for testing
SAMPLE_OLLAMA_RESPONSES = [
    '{"response": "Hello", "done": false}',
    '{"response": " world", "done": false}',
    '{"response": "!", "done": false}',
    '{"response": "", "done": true, "eval_count": 3, "eval_duration": 120000000, "model": "qwen:4b"}',
]


def get_sample_tegrastats_outputs() -> List[str]:
    """Get sample tegrastats outputs for parser testing."""
    return SAMPLE_TEGRASTATS_OUTPUTS


def get_sample_ollama_responses() -> List[str]:
    """Get sample Ollama responses for stream parser testing."""
    return SAMPLE_OLLAMA_RESPONSES