"""
Analysis module for thermal throttling detection and statistical analysis.
"""

from edgewatch.analysis.throttle import (
    ThrottleEvent,
    ThrottleDetector,
    detect_throttling
)
from edgewatch.analysis.stats import (
    StatisticalResult,
    StatisticalEngine,
    run_statistical_benchmark
)

__all__ = [
    "ThrottleEvent",
    "ThrottleDetector",
    "detect_throttling",
    "StatisticalResult",
    "StatisticalEngine",
    "run_statistical_benchmark",
]