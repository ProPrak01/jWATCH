"""
Parser for Ollama's streaming JSON responses.
Handles chunk-by-chunk token accumulation and timing metrics.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class StreamingMetrics:
    """
    Metrics collected during Ollama streaming response parsing.
    """
    t_request_sent: float        # When request was sent (monotonic)
    t_first_token: Optional[float] = None  # When first token arrived (monotonic)
    t_last_token: Optional[float] = None   # When last token arrived (monotonic)
    total_tokens: int = 0        # Total token count from Ollama
    response_chunks: int = 0      # Number of chunks received
    bytes_received: int = 0       # Total bytes received

    @property
    def ttft_ms(self) -> Optional[float]:
        """
        Time to First Token (TTFT) in milliseconds.

        Returns:
            TTFT in milliseconds, or None if first token not received yet
        """
        if self.t_first_token is None:
            return None
        return (self.t_first_token - self.t_request_sent) * 1000.0

    @property
    def total_duration_ms(self) -> Optional[float]:
        """
        Total inference duration in milliseconds.

        Returns:
            Total duration in milliseconds, or None if incomplete
        """
        if self.t_last_token is None:
            return None
        return (self.t_last_token - self.t_request_sent) * 1000.0

    @property
    def generation_duration_ms(self) -> Optional[float]:
        """
        Token generation duration (time from first to last token) in milliseconds.

        Returns:
            Generation duration in milliseconds, or None if incomplete
        """
        if self.t_first_token is None or self.t_last_token is None:
            return None
        return (self.t_last_token - self.t_first_token) * 1000.0

    @property
    def tokens_per_sec_ollama(self) -> Optional[float]:
        """
        Tokens per second based on Ollama's eval_duration metric.

        Returns:
            Tokens per second, or None if eval_duration not available
        """
        return None  # This will be set from Ollama's response

    @property
    def tokens_per_sec_wall(self) -> Optional[float]:
        """
        Tokens per second based on wall-clock time.

        Returns:
            Tokens per second, or None if incomplete timing data
        """
        if self.total_tokens == 0 or self.t_first_token is None or self.t_last_token is None:
            return None
        duration_sec = (self.t_last_token - self.t_first_token)
        if duration_sec <= 0:
            return None
        return self.total_tokens / duration_sec


@dataclass
class ParsedOllamaResponse:
    """
    Complete parsed response from Ollama streaming API.
    """
    response_text: str              # Complete accumulated response text
    model: str                      # Model name
    metrics: StreamingMetrics      # Timing and performance metrics
    eval_duration_ns: Optional[int] = None  # Ollama's own duration metric

    @property
    def tokens_per_sec_ollama(self) -> Optional[float]:
        """
        Tokens per second based on Ollama's eval_duration.

        Returns:
            Tokens per second, or None if eval_duration not available
        """
        if self.eval_duration_ns is None or self.metrics.total_tokens == 0:
            return None
        duration_sec = self.eval_duration_ns / 1e9
        if duration_sec <= 0:
            return None
        return self.metrics.total_tokens / duration_sec

    def cross_validate_timing(self, tolerance: float = 0.05) -> bool:
        """
        Cross-validate timing between Ollama's metric and wall-clock measurement.

        Args:
            tolerance: Acceptable relative difference (default 5%)

        Returns:
            True if timing measurements agree within tolerance
        """
        ollama_tps = self.tokens_per_sec_ollama
        wall_tps = self.metrics.tokens_per_sec_wall

        if ollama_tps is None or wall_tps is None:
            return False  # Can't validate if one measurement is missing

        if ollama_tps == 0:
            return False  # Division by zero protection

        relative_diff = abs(ollama_tps - wall_tps) / ollama_tps
        return relative_diff <= tolerance


class OllamaStreamParser:
    """
    Parser for Ollama's streaming JSON responses.

    Handles:
    - Line-by-line JSON parsing
    - Token accumulation
    - Precise timestamp recording
    - Metric extraction
    """

    def __init__(self):
        """Initialize the stream parser."""
        self._response_text = ""
        self._model = ""
        self._metrics = StreamingMetrics(t_request_sent=time.monotonic())
        self._is_complete = False
        self._eval_duration_ns = None

    def set_request_sent_time(self, timestamp: float) -> None:
        """
        Set the time when the request was sent.

        Args:
            timestamp: Request send time (time.monotonic())
        """
        self._metrics.t_request_sent = timestamp

    def parse_chunk(self, chunk: str) -> Optional[ParsedOllamaResponse]:
        """
        Parse a single JSON chunk from the Ollama stream.

        Args:
            chunk: JSON string from Ollama stream

        Returns:
            ParsedOllamaResponse if stream is complete, None otherwise
        """
        try:
            # Record first token time if not set
            if self._metrics.t_first_token is None:
                self._metrics.t_first_token = time.monotonic()

            # Parse JSON chunk
            data = json.loads(chunk)
            self._metrics.response_chunks += 1

            # Extract response text if present
            if "response" in data:
                response_chunk = data["response"]
                if response_chunk:
                    self._response_text += response_chunk
                    self._metrics.bytes_received += len(chunk)

            # Extract model name if present
            if "model" in data and data["model"]:
                self._model = data["model"]

            # Check if stream is complete
            if data.get("done", False):
                return self._finalize_response(data)

            return None

        except json.JSONDecodeError as e:
            # Handle malformed JSON
            print(f"Error parsing JSON chunk: {e}")
            return None
        except Exception as e:
            # Handle other errors
            print(f"Error processing chunk: {e}")
            return None

    def _finalize_response(self, final_data: dict) -> ParsedOllamaResponse:
        """
        Finalize the response when stream is complete.

        Args:
            final_data: Final JSON chunk from Ollama

        Returns:
            Complete ParsedOllamaResponse
        """
        # Record last token time
        self._metrics.t_last_token = time.monotonic()
        self._is_complete = True

        # Extract final metrics from Ollama
        if "eval_count" in final_data:
            self._metrics.total_tokens = final_data["eval_count"]
        if "eval_duration" in final_data:
            self._eval_duration_ns = final_data["eval_duration"]
        if "model" in final_data and final_data["model"]:
            self._model = final_data["model"]

        return ParsedOllamaResponse(
            response_text=self._response_text,
            model=self._model,
            metrics=self._metrics,
            eval_duration_ns=self._eval_duration_ns
        )

    def is_complete(self) -> bool:
        """Check if the stream parsing is complete."""
        return self._is_complete

    def get_current_response_text(self) -> str:
        """Get the current accumulated response text (incomplete)."""
        return self._response_text

    def get_current_metrics(self) -> StreamingMetrics:
        """Get the current metrics (incomplete)."""
        return self._metrics

    def reset(self) -> None:
        """Reset the parser for a new stream."""
        self._response_text = ""
        self._model = ""
        self._metrics = StreamingMetrics(t_request_sent=time.monotonic())
        self._is_complete = False
        self._eval_duration_ns = None


def parse_ollama_stream(chunks: list[str],
                        request_time: Optional[float] = None) -> Optional[ParsedOllamaResponse]:
    """
    Convenience function to parse a complete Ollama stream.

    Args:
        chunks: List of JSON chunks from Ollama stream
        request_time: Optional request send time (uses current time if not provided)

    Returns:
        ParsedOllamaResponse if parsing succeeds, None otherwise
    """
    parser = OllamaStreamParser()

    if request_time is not None:
        parser.set_request_sent_time(request_time)

    for chunk in chunks:
        response = parser.parse_chunk(chunk)
        if response is not None:
            return response

    return None


async def parse_async_stream(chunks: list[str],
                            request_time: Optional[float] = None) -> Optional[ParsedOllamaResponse]:
    """
    Async convenience function to parse a complete Ollama stream.

    Args:
        chunks: List of JSON chunks from Ollama stream
        request_time: Optional request send time (uses current time if not provided)

    Returns:
        ParsedOllamaResponse if parsing succeeds, None otherwise
    """
    # For now, this is the same as the synchronous version
    # In a real implementation, this could handle async chunk sources
    return parse_ollama_stream(chunks, request_time)