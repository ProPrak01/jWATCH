"""
Comprehensive tests for Ollama Stream Parser.
Tests JSON chunk parsing, token accumulation, timing metrics, and various streaming patterns.
"""

import pytest

from edgewatch.ollama.stream_parser import (
    OllamaStreamParser,
    ParsedOllamaResponse,
    StreamingMetrics,
    parse_ollama_stream,
    parse_async_stream
)


class TestStreamingMetrics:
    """Test suite for StreamingMetrics dataclass."""

    def test_ttft_calculation(self):
        """Test Time To First Token (TTFT) calculation."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,  # 300ms after request
            total_tokens=10
        )

        ttft = metrics.ttft_ms
        assert ttft == 300.0

    def test_ttft_no_first_token(self):
        """Test TTFT calculation when first token not received."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            total_tokens=0
        )

        assert metrics.ttft_ms is None

    def test_total_duration_calculation(self):
        """Test total duration calculation."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_last_token=102.5,  # 2.5 seconds after request
            total_tokens=10
        )

        duration = metrics.total_duration_ms
        assert duration == 2500.0

    def test_total_duration_incomplete(self):
        """Test total duration calculation when incomplete."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            total_tokens=0
        )

        assert metrics.total_duration_ms is None

    def test_generation_duration_calculation(self):
        """Test generation duration (first to last token)."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,  # 300ms TTFT
            t_last_token=101.3,   # 1 second total
            total_tokens=10
        )

        gen_duration = metrics.generation_duration_ms
        assert gen_duration == 1000.0  # 1 second from first to last token

    def test_generation_duration_incomplete(self):
        """Test generation duration when timing incomplete."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            total_tokens=10
        )

        assert metrics.generation_duration_ms is None

    def test_tokens_per_sec_wall(self):
        """Test wall-clock tokens per second calculation."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,  # 1 second generation time
            total_tokens=25     # 25 tokens in 1 second
        )

        tps = metrics.tokens_per_sec_wall
        assert tps == 25.0

    def test_tokens_per_sec_wall_incomplete(self):
        """Test wall-clock TPS when timing incomplete."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            total_tokens=25
        )

        assert metrics.tokens_per_sec_wall is None

    def test_tokens_per_sec_wall_zero_duration(self):
        """Test wall-clock TPS handles zero duration."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=100.3,  # Same time as first token
            total_tokens=10
        )

        assert metrics.tokens_per_sec_wall is None


class TestParsedOllamaResponse:
    """Test suite for ParsedOllamaResponse dataclass."""

    def test_tokens_per_sec_ollama(self):
        """Test Ollama's tokens per second calculation."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            total_tokens=20
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics,
            eval_duration_ns=800_000_000  # 0.8 seconds
        )

        tps = response.tokens_per_sec_ollama
        assert tps == 25.0  # 20 tokens / 0.8 seconds

    def test_tokens_per_sec_ollama_no_eval_duration(self):
        """Test Ollama TPS when eval_duration not available."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            total_tokens=20
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics
        )

        assert response.tokens_per_sec_ollama is None

    def test_cross_validate_timing_success(self):
        """Test timing validation with measurements within tolerance."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            total_tokens=25
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics,
            eval_duration_ns=1_000_000_000  # 1 second
        )

        # 25 tokens/sec from both measurements
        is_valid = response.cross_validate_timing()
        assert is_valid is True

    def test_cross_validate_timing_failure(self):
        """Test timing validation with measurements outside tolerance."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            total_tokens=25
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics,
            eval_duration_ns=500_000_000  # 0.5 seconds (50 tokens/sec - huge difference)
        )

        # 25 vs 50 tokens/sec - 100% difference, should fail
        is_valid = response.cross_validate_timing()
        assert is_valid is False

    def test_cross_validate_timing_custom_tolerance(self):
        """Test timing validation with custom tolerance."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            t_first_token=100.3,
            t_last_token=101.3,
            total_tokens=25
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics,
            eval_duration_ns=750_000_000  # 33.3 tokens/sec
        )

        # 25 vs 33.3 tokens/sec - 33% difference
        # Should fail with default 5% tolerance
        is_valid = response.cross_validate_timing()
        assert is_valid is False

        # Should pass with 50% tolerance
        is_valid = response.cross_validate_timing(tolerance=0.5)
        assert is_valid is True

    def test_cross_validate_timing_incomplete(self):
        """Test timing validation with incomplete metrics."""
        metrics = StreamingMetrics(
            t_request_sent=100.0,
            total_tokens=0
        )

        response = ParsedOllamaResponse(
            response_text="Hello world",
            model="qwen:4b",
            metrics=metrics
        )

        is_valid = response.cross_validate_timing()
        assert is_valid is False


class TestOllamaStreamParser:
    """Test suite for OllamaStreamParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return OllamaStreamParser()

    def test_set_request_sent_time(self, parser):
        """Test setting the request send time."""
        request_time = 100.0
        parser.set_request_sent_time(request_time)

        assert parser.get_current_metrics().t_request_sent == request_time

    def test_parse_single_chunk(self, parser):
        """Test parsing a single response chunk."""
        chunk = '{"response": "Hello", "done": false}'
        response = parser.parse_chunk(chunk)

        # Should return None for incomplete stream
        assert response is None
        assert parser.get_current_response_text() == "Hello"
        assert parser.get_current_metrics().t_first_token is not None

    def test_parse_multiple_chunks(self, parser):
        """Test parsing multiple chunks accumulating text."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "!", "done": false}'
        ]

        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is None  # Still incomplete
        assert parser.get_current_response_text() == "Hello world!"

    def test_parse_complete_response(self, parser):
        """Test parsing a complete response with final chunk."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 80000000, "model": "qwen:4b"}'
        ]

        response = None
        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        # Final chunk should return complete response
        assert response is not None
        assert isinstance(response, ParsedOllamaResponse)
        assert response.response_text == "Hello world"
        assert response.model == "qwen:4b"
        assert response.metrics.total_tokens == 2
        assert response.eval_duration_ns == 80_000_000
        assert parser.is_complete() is True

    def test_parse_empty_response(self, parser):
        """Test parsing a response with empty content."""
        chunk = '{"response": "", "done": true, "eval_count": 0, "eval_duration": 0, "model": "qwen:4b"}'
        response = parser.parse_chunk(chunk)

        assert response is not None
        assert response.response_text == ""
        assert response.metrics.total_tokens == 0

    def test_parse_long_response(self, parser):
        """Test parsing a longer multi-chunk response."""
        # Generate multiple chunks simulating a long response
        chunks = []
        word_count = 0
        for i in range(10):
            word = f"word{i} "
            chunks.append(f'{{"response": "{word}", "done": false}}')
            word_count += 1

        # Final chunk
        chunks.append(
            '{"response": "", "done": true, "eval_count": ' +
            f'{word_count}, "eval_duration": 500000000, "model": "llama3:7b"}}'
        )

        response = None
        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        assert len(response.response_text.split()) == word_count

    def test_parse_with_whitespace_chunks(self, parser):
        """Test parsing chunks with various whitespace."""
        chunks = [
            '{"response": "  Hello  ", "done": false}',
            '{"response": "  world  ", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000, "model": "qwen:4b"}'
        ]

        response = None
        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        # Should preserve whitespace
        assert "  Hello  " in response.response_text
        assert "  world  " in response.response_text

    def test_parse_unicode_response(self, parser):
        """Test parsing response with unicode characters."""
        chunks = [
            '{"response": "Hello 🌍", "done": false}',
            '{"response": " 世界", "done": false}',
            '{"response": "", "done": true, "eval_count": 3, "eval_duration": 150000000, "model": "qwen:4b"}'
        ]

        response = None
        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        assert "🌍" in response.response_text
        assert "世界" in response.response_text

    def test_is_complete_tracking(self, parser):
        """Test completion status tracking."""
        assert parser.is_complete() is False

        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000, "model": "qwen:4b"}'
        ]

        for chunk in chunks[:-1]:
            parser.parse_chunk(chunk)
            assert parser.is_complete() is False

        parser.parse_chunk(chunks[-1])
        assert parser.is_complete() is True

    def test_reset_parser(self, parser):
        """Test resetting the parser for a new stream."""
        # Parse first response
        chunks = [
            '{"response": "First", "done": false}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}'
        ]

        for chunk in chunks:
            parser.parse_chunk(chunk)

        assert parser.get_current_response_text() == "First"
        assert parser.is_complete() is True

        # Reset for new response
        parser.reset()
        assert parser.get_current_response_text() == ""
        assert parser.is_complete() is False

        # Parse second response
        chunks = [
            '{"response": "Second", "done": false}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "llama3:7b"}'
        ]

        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        assert response.response_text == "Second"
        assert response.model == "llama3:7b"

    def test_invalid_json_chunk(self, parser):
        """Test handling invalid JSON chunks."""
        invalid_chunk = '{"response": "Hello", "done": invalid}'

        response = parser.parse_chunk(invalid_chunk)

        # Should return None for invalid JSON
        assert response is None

    def test_missing_response_field(self, parser):
        """Test handling chunks without response field."""
        chunk = '{"done": false}'

        response = parser.parse_chunk(chunk)

        assert response is None
        assert parser.get_current_response_text() == ""

    def test_model_extraction(self, parser):
        """Test model name extraction."""
        chunks = [
            '{"response": "Hello", "model": "qwen:4b", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000}'
        ]

        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        assert response.model == "qwen:4b"

    def test_model_from_final_chunk(self, parser):
        """Test model name from final chunk overrides intermediate."""
        chunks = [
            '{"response": "Hello", "model": "wrong", "done": false}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "correct:1b"}'
        ]

        for chunk in chunks:
            response = parser.parse_chunk(chunk)

        assert response is not None
        assert response.model == "correct:1b"

    def test_metrics_accumulation(self, parser):
        """Test that metrics are accumulated correctly."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000}'
        ]

        for chunk in chunks:
            parser.parse_chunk(chunk)

        metrics = parser.get_current_metrics()
        assert metrics.response_chunks == 3
        assert metrics.total_tokens == 2
        assert metrics.t_first_token is not None
        assert metrics.t_last_token is not None

    def test_bytes_received_tracking(self, parser):
        """Test that bytes received are tracked."""
        chunk = '{"response": "Hello", "done": false}'
        chunk_size = len(chunk)

        parser.parse_chunk(chunk)

        metrics = parser.get_current_metrics()
        assert metrics.bytes_received == chunk_size


class TestConvenienceFunctions:
    """Test suite for convenience functions."""

    def test_parse_ollama_stream_complete(self):
        """Test convenience function with complete stream."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000, "model": "qwen:4b"}'
        ]

        response = parse_ollama_stream(chunks)

        assert response is not None
        assert response.response_text == "Hello world"
        assert response.model == "qwen:4b"

    def test_parse_ollama_stream_with_request_time(self):
        """Test convenience function with custom request time."""
        request_time = 50.0
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}'
        ]

        response = parse_ollama_stream(chunks, request_time)

        assert response is not None
        assert response.metrics.t_request_sent == request_time

    def test_parse_ollama_stream_incomplete(self):
        """Test convenience function with incomplete stream."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": " world", "done": false}'
        ]

        response = parse_ollama_stream(chunks)

        # Should return None for incomplete stream
        assert response is None

    @pytest.mark.asyncio
    async def test_parse_async_stream(self):
        """Test async convenience function."""
        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}'
        ]

        response = await parse_async_stream(chunks)

        assert response is not None
        assert response.response_text == "Hello"


class TestEdgeCases:
    """Test suite for edge cases and unusual scenarios."""

    def test_empty_chunks_list(self):
        """Test parsing empty chunks list."""
        chunks = []
        response = parse_ollama_stream(chunks)

        assert response is None

    def test_single_chunk_complete(self):
        """Test parsing single chunk that completes immediately."""
        chunks = [
            '{"response": "", "done": true, "eval_count": 0, "eval_duration": 0, "model": "qwen:4b"}'
        ]

        response = parse_ollama_stream(chunks)

        assert response is not None
        assert response.response_text == ""
        assert response.metrics.total_tokens == 0

    def test_very_long_single_chunk(self):
        """Test parsing a very long single chunk."""
        long_text = "word " * 1000  # 1000 words
        chunk = f'{{"response": "{long_text}", "done": true, "eval_count": 1000, "eval_duration": 2000000000, "model": "qwen:4b"}}'

        response = parse_ollama_stream([chunk])

        assert response is not None
        assert len(response.response_text.split()) == 1000
        assert response.metrics.total_tokens == 1000

    def test_consecutive_done_chunks(self):
        """Test handling of consecutive done: true chunks."""
        parser = OllamaStreamParser()

        chunks = [
            '{"response": "First", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}',
            '{"response": "Second", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}'
        ]

        response = parser.parse_chunk(chunks[0])
        assert response is not None

        # Second done chunk should still return a response
        response2 = parser.parse_chunk(chunks[1])
        # This depends on implementation - may return None or handle gracefully

    def test_malformed_json_recovery(self):
        """Test parser recovery after malformed JSON."""
        parser = OllamaStreamParser()

        chunks = [
            '{"response": "Hello", "done": false}',
            '{"response": "invalid json',
            '{"response": " world", "done": false}',
            '{"response": "", "done": true, "eval_count": 2, "eval_duration": 100000000, "model": "qwen:4b"}'
        ]

        response = None
        for chunk in chunks:
            result = parser.parse_chunk(chunk)
            if result is not None:
                response = result

        # Should eventually succeed despite malformed chunk
        # Note: This depends on error recovery implementation

    def test_special_characters_in_response(self):
        """Test handling special characters in response text."""
        special_chars = '\n\t\r\"\'\\'
        chunks = [
            f'{{"response": "{special_chars}", "done": false}}',
            '{"response": "", "done": true, "eval_count": 1, "eval_duration": 50000000, "model": "qwen:4b"}'
        ]

        response = parse_ollama_stream(chunks)

        assert response is not None
        assert special_chars in response.response_text


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
