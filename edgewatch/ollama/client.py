"""
Async Ollama client for streaming LLM inference with precise timing.
Handles HTTP streaming, error management, and performance metrics.
"""

import json
import time
from dataclasses import dataclass, field
from typing import AsyncIterator, Optional, Union

import httpx

from edgewatch.ollama.stream_parser import (
    OllamaStreamParser,
    ParsedOllamaResponse,
    parse_ollama_stream
)


@dataclass
class InferenceRequest:
    """
    Configuration for an inference request to Ollama.
    """
    model: str                     # Model name (e.g., "qwen:4b")
    prompt: str                    # Input prompt
    stream: bool = True            # Enable streaming (always true for benchmarking)
    options: dict = field(default_factory=dict)  # Additional Ollama options


@dataclass
class InferenceResult:
    """
    Complete result from an Ollama inference request.
    """
    model: str                      # Model name
    prompt: str                     # Original prompt
    response_text: str              # Complete response text
    t_request_sent: float           # Request send time (monotonic)
    t_first_token: Optional[float]  # First token arrival time (monotonic)
    t_last_token: Optional[float]   # Last token arrival time (monotonic)
    ttft_ms: Optional[float]       # Time to First Token in milliseconds
    total_tokens: int               # Total token count
    tokens_per_sec_ollama: Optional[float]   # Ollama's TPS measurement
    tokens_per_sec_wall: Optional[float]     # Wall-clock TPS measurement
    inference_duration_sec: Optional[float]   # Total duration in seconds
    eval_duration_ns: Optional[int] # Ollama's eval duration in nanoseconds
    response_chunks: int            # Number of response chunks received
    bytes_received: int            # Total bytes received

    @property
    def ttft_sec(self) -> Optional[float]:
        """Time to First Token in seconds."""
        if self.ttft_ms is None:
            return None
        return self.ttft_ms / 1000.0

    def cross_validate_timing(self, tolerance: float = 0.05) -> bool:
        """
        Cross-validate timing between Ollama's metric and wall-clock measurement.

        Args:
            tolerance: Acceptable relative difference (default 5%)

        Returns:
            True if timing measurements agree within tolerance, False otherwise
        """
        if self.tokens_per_sec_ollama is None or self.tokens_per_sec_wall is None:
            return False

        if self.tokens_per_sec_ollama == 0:
            return False

        relative_diff = abs(self.tokens_per_sec_ollama - self.tokens_per_sec_wall) / self.tokens_per_sec_ollama
        return relative_diff <= tolerance


class OllamaClient:
    """
    Async client for Ollama API with streaming support and precise timing.

    Features:
    - Async HTTP streaming using httpx
    - Precise timestamp recording for performance analysis
    - Comprehensive error handling
    - Support for both real Ollama and mock responses
    """

    def __init__(self,
                 base_url: str = "http://localhost:11434",
                 timeout: float = 300.0,
                 mock_mode: bool = False):
        """
        Initialize the Ollama client.

        Args:
            base_url: Base URL for Ollama API
            timeout: Request timeout in seconds
            mock_mode: If True, use mock responses instead of real Ollama
        """
        self.base_url = base_url
        self.timeout = timeout
        self.mock_mode = mock_mode

        # Real HTTP client
        self._http_client: Optional[httpx.AsyncClient] = None

        # Mock client for testing
        self._mock_client = None

        # Connection status
        self._is_connected = False

    async def __aenter__(self):
        """Async context manager entry."""
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.disconnect()

    async def connect(self) -> bool:
        """
        Establish connection to Ollama API.

        Returns:
            True if connection successful, False otherwise
        """
        if self.mock_mode:
            # Initialize mock client
            from edgewatch.utils.mocks import create_ollama_mock
            self._mock_client = create_ollama_mock()
            self._is_connected = True
            return True

        try:
            self._http_client = httpx.AsyncClient(timeout=self.timeout)
            # Test connection
            await self.check_connection()
            self._is_connected = True
            return True
        except Exception as e:
            print(f"Error connecting to Ollama: {e}")
            return False

    async def disconnect(self) -> None:
        """Close connection to Ollama API."""
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        self._mock_client = None
        self._is_connected = False

    async def check_connection(self) -> bool:
        """
        Check if Ollama API is accessible.

        Returns:
            True if Ollama is accessible, False otherwise
        """
        if self.mock_mode:
            return self._is_connected

        try:
            response = await self._http_client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except Exception:
            return False

    async def list_models(self) -> list[dict]:
        """
        List available models in Ollama.

        Returns:
            List of model information dictionaries
        """
        if self.mock_mode:
            # Return mock model list
            return [
                {"name": "qwen:4b", "size": 2147483648},
                {"name": "qwen:7b", "size": 3758096384},
                {"name": "llama3.2:3b", "size": 1610612736}
            ]

        try:
            response = await self._http_client.get(f"{self.base_url}/api/tags")
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            print(f"Error listing models: {e}")
            return []

    async def infer(self,
                   request: InferenceRequest) -> InferenceResult:
        """
        Perform inference request to Ollama.

        Args:
            request: InferenceRequest with model and prompt

        Returns:
            InferenceResult with timing metrics and response text
        """
        if self.mock_mode:
            return await self._mock_infer(request)
        else:
            return await self._real_infer(request)

    async def _real_infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform real inference via Ollama API."""
        if not self._http_client:
            raise RuntimeError("Not connected to Ollama API")

        # Prepare request body
        request_body = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": request.stream
        }

        # Add options if provided
        if request.options:
            request_body["options"] = request.options

        # Record request send time
        t_request_sent = time.monotonic()

        try:
            # Make POST request with streaming
            url = f"{self.base_url}/api/generate"
            response = await self._http_client.post(
                url,
                json=request_body,
                timeout=self.timeout
            )

            response.raise_for_status()

            # Parse streaming response
            parser = OllamaStreamParser()
            parser.set_request_sent_time(t_request_sent)

            # Process streaming chunks
            async for chunk in response.aiter_bytes():
                chunk_str = chunk.decode('utf-8').strip()
                if chunk_str:
                    # Handle multiple JSON objects in single chunk
                    for json_str in chunk_str.split('\n'):
                        if json_str.strip():
                            result = parser.parse_chunk(json_str)
                            if result is not None:
                                break

            # Get final parsed response
            if not parser.is_complete():
                raise RuntimeError("Incomplete response from Ollama")

            parsed_response = parser._finalize_response({
                "eval_count": parser._metrics.total_tokens,
                "eval_duration": parser._eval_duration_ns or 0,
                "model": request.model
            })

            # Build InferenceResult
            return self._build_inference_result(
                request, parsed_response, t_request_sent
            )

        except httpx.HTTPStatusError as e:
            raise RuntimeError(f"HTTP error from Ollama: {e.response.status_code}") from e
        except httpx.RequestError as e:
            raise RuntimeError(f"Request error: {e}") from e

    async def _mock_infer(self, request: InferenceRequest) -> InferenceResult:
        """Perform mock inference for testing."""
        from edgewatch.utils.mocks import create_ollama_mock

        # Create or reuse mock client
        if self._mock_client is None:
            self._mock_client = create_ollama_mock()

        # Configure mock from request
        self._mock_client.config.model = request.model
        self._mock_client.config.prompt = request.prompt

        # Record request send time
        t_request_sent = time.monotonic()

        # Get mock streaming response
        mock_response = await self._mock_client.post(
            f"{self.base_url}/api/generate",
            json_data={
                "model": request.model,
                "prompt": request.prompt,
                "stream": request.stream
            },
            timeout=self.timeout
        )

        # Parse streaming response
        parser = OllamaStreamParser()
        parser.set_request_sent_time(t_request_sent)

        async for chunk in mock_response:
            chunk_str = chunk.decode('utf-8').strip()
            if chunk_str:
                for json_str in chunk_str.split('\n'):
                    if json_str.strip():
                        result = parser.parse_chunk(json_str)
                        if result is not None:
                            break

        # Build InferenceResult
        if not parser.is_complete():
            raise RuntimeError("Incomplete mock response")

        parsed_response = parser._finalize_response({
            "eval_count": parser._metrics.total_tokens,
            "eval_duration": parser._eval_duration_ns or 0,
            "model": request.model
        })

        return self._build_inference_result(
            request, parsed_response, t_request_sent
        )

    def _build_inference_result(self,
                               request: InferenceRequest,
                               parsed_response: ParsedOllamaResponse,
                               t_request_sent: float) -> InferenceResult:
        """Build InferenceResult from parsed Ollama response."""
        return InferenceResult(
            model=parsed_response.model or request.model,
            prompt=request.prompt,
            response_text=parsed_response.response_text,
            t_request_sent=t_request_sent,
            t_first_token=parsed_response.metrics.t_first_token,
            t_last_token=parsed_response.metrics.t_last_token,
            ttft_ms=parsed_response.metrics.ttft_ms,
            total_tokens=parsed_response.metrics.total_tokens,
            tokens_per_sec_ollama=parsed_response.tokens_per_sec_ollama,
            tokens_per_sec_wall=parsed_response.metrics.tokens_per_sec_wall,
            inference_duration_sec=parsed_response.metrics.total_duration_ms / 1000.0
            if parsed_response.metrics.total_duration_ms is not None else None,
            eval_duration_ns=parsed_response.eval_duration_ns,
            response_chunks=parsed_response.metrics.response_chunks,
            bytes_received=parsed_response.metrics.bytes_received
        )

    async def infer_stream(self,
                         request: InferenceRequest) -> AsyncIterator[str]:
        """
        Stream inference results token by token.

        Args:
            request: InferenceRequest with model and prompt

        Yields:
            Response text tokens as they arrive
        """
        if self.mock_mode:
            async for token in self._mock_infer_stream(request):
                yield token
        else:
            async for token in self._real_infer_stream(request):
                yield token

    async def _real_infer_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream real inference results."""
        if not self._http_client:
            raise RuntimeError("Not connected to Ollama API")

        request_body = {
            "model": request.model,
            "prompt": request.prompt,
            "stream": True
        }

        try:
            url = f"{self.base_url}/api/generate"
            response = await self._http_client.post(url, json=request_body, timeout=self.timeout)
            response.raise_for_status()

            async for chunk in response.aiter_bytes():
                chunk_str = chunk.decode('utf-8').strip()
                if chunk_str:
                    for json_str in chunk_str.split('\n'):
                        if json_str.strip():
                            try:
                                data = json.loads(json_str)
                                if "response" in data and data["response"]:
                                    yield data["response"]
                                if data.get("done", False):
                                    return
                            except json.JSONDecodeError:
                                continue

        except Exception as e:
            raise RuntimeError(f"Error streaming inference: {e}") from e

    async def _mock_infer_stream(self, request: InferenceRequest) -> AsyncIterator[str]:
        """Stream mock inference results."""
        import asyncio
        from edgewatch.utils.mocks import create_ollama_mock

        if self._mock_client is None:
            self._mock_client = create_ollama_mock()

        self._mock_client.config.model = request.model
        self._mock_client.config.prompt = request.prompt

        mock_response = await self._mock_client.post(
            f"{self.base_url}/api/generate",
            json_data={
                "model": request.model,
                "prompt": request.prompt,
                "stream": True
            },
            timeout=self.timeout
        )

        async for chunk in mock_response:
            chunk_str = chunk.decode('utf-8').strip()
            if chunk_str:
                for json_str in chunk_str.split('\n'):
                    if json_str.strip():
                        try:
                            data = json.loads(json_str)
                            if "response" in data and data["response"]:
                                yield data["response"]
                            if data.get("done", False):
                                return
                        except json.JSONDecodeError:
                            continue

    def is_connected(self) -> bool:
        """Check if client is connected."""
        return self._is_connected


async def quick_inference(model: str, prompt: str, timeout: float = 300.0) -> InferenceResult:
    """
    Convenience function for a quick inference call.

    Args:
        model: Model name to use
        prompt: Input prompt
        timeout: Request timeout in seconds

    Returns:
        InferenceResult with timing metrics
    """
    async with OllamaClient(timeout=timeout) as client:
        request = InferenceRequest(model=model, prompt=prompt)
        return await client.infer(request)