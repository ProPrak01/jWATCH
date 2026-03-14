"""
Time-series correlation module for aligning hardware samples with inference windows.
"""

from edgewatch.correlation.engine import CorrelationEngine, CorrelatedResult, correlate_inference
from edgewatch.correlation.interpolator import (
    Interpolator,
    InterpolatedSample,
    InterpolationMethod,
    interpolate_at_timestamp
)

__all__ = [
    "CorrelationEngine",
    "CorrelatedResult",
    "correlate_inference",
    "Interpolator",
    "InterpolatedSample",
    "InterpolationMethod",
    "interpolate_at_timestamp",
]