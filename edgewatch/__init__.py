"""
edgewatch — CLI-based LLM benchmarking and hardware profiler for NVIDIA Jetson.

A command-line tool that benchmarks LLM inference on NVIDIA Jetson devices by
simultaneously querying models via the Ollama API and sampling hardware metrics
via tegrastats, correlating inference performance with real-time hardware state.
"""

__version__ = "0.1.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

# Core components
from edgewatch.tegrastats.parser import TegraStatsSample, TegrastatsParser
from edgewatch.tegrastats.sampler import TegrastatsSampler
from edgewatch.ollama.client import OllamaClient, InferenceRequest, InferenceResult
from edgewatch.ollama.stream_parser import (
    OllamaStreamParser,
    ParsedOllamaResponse,
    StreamingMetrics
)

# Phase 2: Correlation components
from edgewatch.correlation.engine import CorrelationEngine, CorrelatedResult, correlate_inference
from edgewatch.correlation.interpolator import (
    Interpolator,
    InterpolatedSample,
    InterpolationMethod,
    interpolate_at_timestamp
)

__all__ = [
    # Version
    "__version__",

    # Tegrastats components
    "TegraStatsSample",
    "TegrastatsParser",
    "TegrastatsSampler",

    # Ollama components
    "OllamaClient",
    "InferenceRequest",
    "InferenceResult",
    "OllamaStreamParser",
    "ParsedOllamaResponse",
    "StreamingMetrics",

    # Correlation components
    "CorrelationEngine",
    "CorrelatedResult",
    "correlate_inference",
    "Interpolator",
    "InterpolatedSample",
    "InterpolationMethod",
    "interpolate_at_timestamp",
]