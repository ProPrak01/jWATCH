"""
Ollama API client module for LLM inference.
"""

from edgewatch.ollama.client import OllamaClient, InferenceRequest, InferenceResult
from edgewatch.ollama.stream_parser import (
    OllamaStreamParser,
    ParsedOllamaResponse,
    StreamingMetrics
)

__all__ = [
    "OllamaClient",
    "InferenceRequest",
    "InferenceResult",
    "OllamaStreamParser",
    "ParsedOllamaResponse",
    "StreamingMetrics",
]