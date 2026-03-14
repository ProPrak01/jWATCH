"""
Comprehensive tests for Ollama Client.
Tests async streaming, timestamp accuracy, error handling, and mock responses.
"""

import pytest
from edgewatch.ollama.client import (
    OllamaClient,
    InferenceRequest,
    InferenceResult,
    quick_inference
)


class TestInferenceRequest:
    """Test suite for InferenceRequest dataclass."""

    def test_default_values(self):
        """Test InferenceRequest default values."""
        request = InferenceRequest(model="qwen:4b", prompt="Hello")

        assert request.model == "qwen:4b"
        assert request.prompt == "Hello"
        assert request.stream is True  # Default
        assert request.options == {}  # Default

    def test_custom_values(self):
        """Test InferenceRequest with custom values."""
        options = {"temperature": 0.7, "max_tokens": 100}
        request = InferenceRequest(
            model="llama3:7b",
            prompt="Explain quantum computing",
            stream=True,
            options=options
        )

        assert request.model == "llama3:7b"
        assert request.prompt == "Explain quantum computing"
        assert request.stream is True
        assert request.options == options

    def test_stream_false(self):
        """Test InferenceRequest with stream=False."""
        request = InferenceRequest(
            model="qwen:4b",
            prompt="Hello",
            stream=False
        )

        assert request.stream is False


class TestInferenceResult:
    """Test suite for InferenceResult dataclass."""

    def test_full_result(self):
        """Test InferenceResult with all fields populated."""
        result = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi there!",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=2,
            tokens_per_sec_ollama=2.5,
            tokens_per_sec_wall=2.5,
            inference_duration_sec=1.3,
            eval_duration_ns=800_000_000,
            response_chunks=3,
            bytes_received=150
        )

        assert result.model == "qwen:4b"
        assert result.ttft_ms == 300.0
        assert result.ttft_sec == 0.3

    def test_cross_validate_timing_success(self):
        """Test timing validation with matching measurements."""
        result = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=2,
            tokens_per_sec_ollama=2.0,
            tokens_per_sec_wall=2.0,
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        assert result.cross_validate_timing() is True

    def test_cross_validate_timing_failure(self):
        """Test timing validation with mismatched measurements."""
        result = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=2,
            tokens_per_sec_ollama=2.0,
            tokens_per_sec_wall=3.0,  # 50% difference
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        assert result.cross_validate_timing() is False

    def test_cross_validate_timing_custom_tolerance(self):
        """Test timing validation with custom tolerance."""
        result = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            ttft_ms=300.0,
            total_tokens=2,
            tokens_per_sec_ollama=2.0,
            tokens_per_sec_wall=3.0,  # 50% difference
            inference_duration_sec=1.3,
            eval_duration_ns=1_000_000_000,
            response_chunks=2,
            bytes_received=50
        )

        # Should fail with default 5% tolerance
        assert result.cross_validate_timing() is False

        # Should pass with 60% tolerance
        assert result.cross_validate_timing(tolerance=0.6) is True

    def test_cross_validate_timing_incomplete(self):
        """Test timing validation with incomplete metrics."""
        result = InferenceResult(
            model="qwen:4b",
            prompt="Hello",
            response_text="Hi",
            t_request_sent=100.0,
            t_first_token=None,
            t_last_token=None,
            ttft_ms=None,
            total_tokens=2,
            tokens_per_sec_ollama=None,
            tokens_per_sec_wall=None,
            inference_duration_sec=None,
            eval_duration_ns=None,
            response_chunks=2,
            bytes_received=50
        )

        assert result.cross_validate_timing() is False


class TestOllamaClient:
    """Test suite for OllamaClient class."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing."""
        return OllamaClient(mock_mode=True)

    @pytest.mark.asyncio
    async def test_initialization(self):
        """Test client initialization."""
        client = OllamaClient(mock_mode=True)
        assert client.base_url == "http://localhost:11434"
        assert client.timeout == 300.0
        assert client.mock_mode is True
        assert client.is_connected() is False

    @pytest.mark.asyncio
    async def test_connect_disconnect(self, client):
        """Test connecting and disconnecting."""
        assert not client.is_connected()

        await client.connect()
        assert client.is_connected()

        await client.disconnect()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_context_manager(self):
        """Test async context manager usage."""
        async with OllamaClient(mock_mode=True) as client:
            assert client.is_connected()

        # Should be disconnected after exiting context
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_check_connection_mock(self, client):
        """Test connection check in mock mode."""
        await client.connect()
        assert await client.check_connection() is True

    @pytest.mark.asyncio
    async def test_list_models_mock(self, client):
        """Test listing models in mock mode."""
        await client.connect()
        models = await client.list_models()

        assert isinstance(models, list)
        assert len(models) > 0
        assert any(model["name"] == "qwen:4b" for model in models)

    @pytest.mark.asyncio
    async def test_infer_mock(self, client):
        """Test inference in mock mode."""
        await client.connect()

        request = InferenceRequest(model="qwen:4b", prompt="Hello")
        result = await client.infer(request)

        assert isinstance(result, InferenceResult)
        assert result.model == "qwen:4b"
        assert result.prompt == "Hello"
        assert result.response_text != ""
        assert result.total_tokens > 0
        assert result.ttft_ms is not None
        assert result.t_request_sent is not None

    @pytest.mark.asyncio
    async def test_infer_mock_timing(self, client):
        """Test timing accuracy in mock inference."""
        await client.connect()

        request = InferenceRequest(model="qwen:4b", prompt="Hello")
        result = await client.infer(request)

        # Verify timing metrics are reasonable
        assert result.ttft_ms > 0
        assert result.t_first_token > result.t_request_sent
        assert result.t_last_token >= result.t_first_token
        assert result.total_tokens > 0

        # Verify TPS calculations
        if result.tokens_per_sec_ollama is not None:
            assert result.tokens_per_sec_ollama > 0
        if result.tokens_per_sec_wall is not None:
            assert result.tokens_per_sec_wall > 0

    @pytest.mark.asyncio
    async def test_infer_mock_cross_validate(self, client):
        """Test timing cross-validation in mock mode."""
        await client.connect()

        request = InferenceRequest(model="qwen:4b", prompt="Hello")
        result = await client.infer(request)

        # Cross-validate timing (should pass within tolerance)
        is_valid = result.cross_validate_timing()
        assert is_valid is True

    @pytest.mark.asyncio
    async def test_infer_stream_mock(self, client):
        """Test streaming inference in mock mode."""
        await client.connect()

        request = InferenceRequest(model="qwen:4b", prompt="Hello")
        tokens = []
        async for token in client.infer_stream(request):
            tokens.append(token)

        assert len(tokens) > 0
        full_response = "".join(tokens)
        assert len(full_response) > 0

    @pytest.mark.asyncio
    async def test_infer_with_options(self, client):
        """Test inference with additional options."""
        await client.connect()

        options = {"temperature": 0.7, "max_tokens": 100}
        request = InferenceRequest(
            model="qwen:4b",
            prompt="Hello",
            options=options
        )

        result = await client.infer(request)
        assert result.model == "qwen:4b"
        assert result.response_text != ""

    @pytest.mark.asyncio
    async def test_multiple_inferences(self, client):
        """Test multiple consecutive inferences."""
        await client.connect()

        prompts = ["Hello", "How are you?", "Goodbye"]
        results = []

        for prompt in prompts:
            request = InferenceRequest(model="qwen:4b", prompt=prompt)
            result = await client.infer(request)
            results.append(result)

        # All inferences should succeed
        for result in results:
            assert isinstance(result, InferenceResult)
            assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_infer_not_connected(self):
        """Test inference when not connected."""
        client = OllamaClient(mock_mode=True)

        request = InferenceRequest(model="qwen:4b", prompt="Hello")

        # Should raise RuntimeError when not connected
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.infer(request)

    @pytest.mark.asyncio
    async def test_infer_stream_not_connected(self):
        """Test streaming inference when not connected."""
        client = OllamaClient(mock_mode=True)

        request = InferenceRequest(model="qwen:4b", prompt="Hello")

        # Should raise RuntimeError when not connected
        with pytest.raises(RuntimeError, match="Not connected"):
            async for _ in client.infer_stream(request):
                pass

    @pytest.mark.asyncio
    async def test_custom_base_url(self):
        """Test client with custom base URL."""
        client = OllamaClient(base_url="http://custom:8080", mock_mode=True)

        await client.connect()
        assert client.is_connected()
        assert client.base_url == "http://custom:8080"

        await client.disconnect()

    @pytest.mark.asyncio
    async def test_custom_timeout(self):
        """Test client with custom timeout."""
        client = OllamaClient(timeout=60.0, mock_mode=True)

        await client.connect()
        assert client.timeout == 60.0

        # Perform inference with custom timeout
        request = InferenceRequest(model="qwen:4b", prompt="Hello")
        result = await client.infer(request)
        assert result is not None

    @pytest.mark.asyncio
    async def test_real_mode_connection_failure(self):
        """Test that real mode handles connection failure gracefully."""
        client = OllamaClient(mock_mode=False, timeout=1.0)

        # Try to connect to non-existent server
        connected = await client.connect()
        assert connected is False

        # Should not be connected
        assert client.is_connected() is False


class TestConvenienceFunction:
    """Test suite for convenience functions."""

    @pytest.mark.asyncio
    async def test_quick_inference(self):
        """Test quick_inference convenience function."""
        result = await quick_inference(model="qwen:4b", prompt="Hello", timeout=30.0)

        assert isinstance(result, InferenceResult)
        assert result.model == "qwen:4b"
        assert result.prompt == "Hello"
        assert result.response_text != ""
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_quick_inference_multiple_calls(self):
        """Test multiple quick_inference calls."""
        prompts = ["Hello", "How are you?"]

        results = []
        for prompt in prompts:
            result = await quick_inference(model="qwen:4b", prompt=prompt)
            results.append(result)

        # All calls should succeed
        for result in results:
            assert isinstance(result, InferenceResult)
            assert result.total_tokens > 0


class TestEdgeCases:
    """Test suite for edge cases and unusual scenarios."""

    @pytest.mark.asyncio
    async def test_empty_prompt(self):
        """Test inference with empty prompt."""
        async with OllamaClient(mock_mode=True) as client:
            request = InferenceRequest(model="qwen:4b", prompt="")
            result = await client.infer(request)

            assert result is not None
            assert result.prompt == ""

    @pytest.mark.asyncio
    async def test_very_long_prompt(self):
        """Test inference with very long prompt."""
        long_prompt = "word " * 1000  # 1000 words

        async with OllamaClient(mock_mode=True) as client:
            request = InferenceRequest(model="qwen:4b", prompt=long_prompt)
            result = await client.infer(request)

            assert result is not None
            assert result.prompt == long_prompt

    @pytest.mark.asyncio
    async def test_stream_false_request(self):
        """Test request with stream=False (still works for benchmarking)."""
        async with OllamaClient(mock_mode=True) as client:
            request = InferenceRequest(model="qwen:4b", prompt="Hello", stream=False)
            result = await client.infer(request)

            assert result is not None
            assert result.response_text != ""

    @pytest.mark.asyncio
    async def test_reconnect_client(self):
        """Test reconnecting the client."""
        client = OllamaClient(mock_mode=True)

        # First connection
        await client.connect()
        assert client.is_connected()
        await client.disconnect()
        assert not client.is_connected()

        # Second connection
        await client.connect()
        assert client.is_connected()
        await client.disconnect()
        assert not client.is_connected()

    @pytest.mark.asyncio
    async def test_infer_after_disconnect(self):
        """Test that inference fails after disconnect."""
        client = OllamaClient(mock_mode=True)

        await client.connect()
        await client.disconnect()

        request = InferenceRequest(model="qwen:4b", prompt="Hello")

        # Should fail after disconnect
        with pytest.raises(RuntimeError, match="Not connected"):
            await client.infer(request)

    @pytest.mark.asyncio
    async def test_concurrent_inferences(self):
        """Test multiple concurrent inferences."""
        import asyncio

        async with OllamaClient(mock_mode=True) as client:
            # Create multiple inference tasks
            tasks = [
                client.infer(InferenceRequest(model="qwen:4b", prompt=f"Hello {i}"))
                for i in range(5)
            ]

            # Execute all tasks concurrently
            results = await asyncio.gather(*tasks)

            # All inferences should succeed
            assert len(results) == 5
            for result in results:
                assert isinstance(result, InferenceResult)
                assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_stream_empty_response(self):
        """Test streaming with potential empty response."""
        async with OllamaClient(mock_mode=True) as client:
            request = InferenceRequest(model="qwen:4b", prompt="Generate nothing")

            tokens = []
            async for token in client.infer_stream(request):
                tokens.append(token)

            # Should not crash, even if response is empty
            # Mock will generate some tokens regardless

    @pytest.mark.asyncio
    async def test_infer_different_models(self):
        """Test inference with different model names."""
        models = ["qwen:4b", "qwen:7b", "llama3:2:3b"]

        async with OllamaClient(mock_mode=True) as client:
            for model in models:
                request = InferenceRequest(model=model, prompt="Hello")
                result = await client.infer(request)

                assert result is not None
                assert result.model == model


if __name__ == "__main__":
    pytest.main([__file__, "-v"])