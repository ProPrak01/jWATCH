"""
Tegrastats integration module for hardware monitoring.
"""

from edgewatch.tegrastats.parser import TegraStatsSample, TegrastatsParser
from edgewatch.tegrastats.sampler import TegrastatsSampler

__all__ = [
    "TegraStatsSample",
    "TegrastatsParser",
    "TegrastatsSampler",
]