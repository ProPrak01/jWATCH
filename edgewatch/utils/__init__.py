"""
Utility functions and mocking framework for testing.
"""

from edgewatch.utils.mocks import (
    MockTegrastatsSubprocess,
    MockOllamaClient,
    MockOllamaResponse,
    TegrastatsMockConfig,
    OllamaMockConfig,
    create_tegrastats_mock,
    create_ollama_mock,
    get_sample_tegrastats_outputs,
    get_sample_ollama_responses,
)

__all__ = [
    "MockTegrastatsSubprocess",
    "MockOllamaClient",
    "MockOllamaResponse",
    "TegrastatsMockConfig",
    "OllamaMockConfig",
    "create_tegrastats_mock",
    "create_ollama_mock",
    "get_sample_tegrastats_outputs",
    "get_sample_ollama_responses",
]