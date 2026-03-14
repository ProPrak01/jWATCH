"""
Parser for tegrastats hardware monitoring output.
Handles various JetPack versions and missing fields gracefully.
"""

import re
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional


@dataclass
class TegraStatsSample:
    """
    Represents a single hardware metrics sample from tegrastats.

    All timing uses time.monotonic() to avoid NTP clock adjustment issues.
    """
    timestamp: float              # time.monotonic() when sample was received
    wall_time: datetime            # actual wall clock time
    ram_used_mb: int               # RAM usage in MB
    ram_total_mb: int              # Total RAM in MB
    gpu_freq_pct: int              # GPU frequency percentage (GPC_FREQ %)
    cpu_loads: list[int]           # Per-core CPU load percentages
    cpu_temp: float                # CPU temperature in Celsius
    gpu_temp: float                # GPU temperature in Celsius
    tj_temp: float                 # Thermal junction temperature in Celsius
    power_mw: int                  # Power consumption in milliwatts
    swap_used_mb: Optional[int] = None      # Swap usage in MB (may be missing)
    swap_total_mb: Optional[int] = None     # Total swap in MB (may be missing)
    emc_freq_pct: Optional[int] = None       # EMC frequency percentage (may be missing)


class TegrastatsParser:
    """
    Parses raw tegrastats output into structured TegraStatsSample objects.
    Uses regex patterns to handle format variations across JetPack versions.
    """

    # Regex patterns for parsing tegrastats output
    RAM_PATTERN = re.compile(r"RAM (\d+)/(\d+)MB")
    SWAP_PATTERN = re.compile(r"SWAP (\d+)/(\d+)MB")
    CPU_LOAD_PATTERN = re.compile(r"CPU \[([^\]]+)\]")
    GPU_FREQ_PATTERN = re.compile(r"GPC_FREQ (\d+)%")
    EMC_FREQ_PATTERN = re.compile(r"EMC_FREQ (\d+)%")
    CPU_TEMP_PATTERN = re.compile(r"CPU@([\d.]+)C")
    GPU_TEMP_PATTERN = re.compile(r"GPU@([\d.]+)C")
    TJ_TEMP_PATTERN = re.compile(r"tj@([\d.]+)C")
    POWER_PATTERN = re.compile(r"VDD_IN (\d+)mW")
    WALL_TIME_PATTERN = re.compile(r"^(\d{2}-\d{2}-\d{4} \d{2}:\d{2}:\d{2})")

    def __init__(self):
        self._parse_errors = 0

    def parse(self, raw_line: str) -> Optional[TegraStatsSample]:
        """
        Parse a single tegrastats line into a TegraStatsSample.

        Args:
            raw_line: Raw output line from tegrastats

        Returns:
            TegraStatsSample if parsing succeeds, None otherwise
        """
        try:
            # Extract wall time if present
            wall_time = self._extract_wall_time(raw_line)

            # Extract RAM usage
            ram_used_mb, ram_total_mb = self._extract_ram(raw_line)

            # Extract swap usage (may be missing)
            swap_used_mb, swap_total_mb = self._extract_swap(raw_line)

            # Extract CPU loads
            cpu_loads = self._extract_cpu_loads(raw_line)

            # Extract GPU frequency
            gpu_freq_pct = self._extract_gpu_freq(raw_line)

            # Extract EMC frequency (may be missing)
            emc_freq_pct = self._extract_emc_freq(raw_line)

            # Extract temperatures
            cpu_temp = self._extract_cpu_temp(raw_line)
            gpu_temp = self._extract_gpu_temp(raw_line)
            tj_temp = self._extract_tj_temp(raw_line)

            # Extract power consumption
            power_mw = self._extract_power(raw_line)

            # Validate critical fields
            if None in [ram_used_mb, ram_total_mb, cpu_temp, gpu_temp, tj_temp, power_mw]:
                raise ValueError("Missing critical fields in tegrastats output")

            # Use monotonic timestamp (will be set by sampler)
            timestamp = time.monotonic()

            return TegraStatsSample(
                timestamp=timestamp,
                wall_time=wall_time,
                ram_used_mb=ram_used_mb,
                ram_total_mb=ram_total_mb,
                gpu_freq_pct=gpu_freq_pct or 0,
                cpu_loads=cpu_loads or [],
                cpu_temp=cpu_temp,
                gpu_temp=gpu_temp,
                tj_temp=tj_temp,
                power_mw=power_mw,
                swap_used_mb=swap_used_mb,
                swap_total_mb=swap_total_mb,
                emc_freq_pct=emc_freq_pct
            )

        except Exception as e:
            self._parse_errors += 1
            # Log the error but don't crash
            # In production, you'd want proper logging here
            return None

    def _extract_wall_time(self, raw_line: str) -> Optional[datetime]:
        """Extract wall time from tegrastats line if present."""
        match = self.WALL_TIME_PATTERN.search(raw_line)
        if match:
            try:
                # Parse format: MM-DD-YYYY HH:MM:SS
                time_str = match.group(1)
                return datetime.strptime(time_str, "%m-%d-%Y %H:%M:%S")
            except ValueError:
                return None
        return None

    def _extract_ram(self, raw_line: str) -> tuple[int, int]:
        """Extract RAM usage and total from tegrastats line."""
        match = self.RAM_PATTERN.search(raw_line)
        if match:
            try:
                used_mb = int(match.group(1))
                total_mb = int(match.group(2))
                return used_mb, total_mb
            except (ValueError, IndexError):
                pass
        raise ValueError("Could not extract RAM information from tegrastats output")

    def _extract_swap(self, raw_line: str) -> tuple[Optional[int], Optional[int]]:
        """Extract swap usage and total from tegrastats line (may be missing)."""
        match = self.SWAP_PATTERN.search(raw_line)
        if match:
            try:
                used_mb = int(match.group(1))
                total_mb = int(match.group(2))
                return used_mb, total_mb
            except (ValueError, IndexError):
                pass
        return None, None

    def _extract_cpu_loads(self, raw_line: str) -> list[int]:
        """Extract per-core CPU load percentages."""
        match = self.CPU_LOAD_PATTERN.search(raw_line)
        if match:
            loads_str = match.group(1)
            # Parse format: "23%@1420,15%@1420,off,off"
            cpu_loads = []
            for core_str in loads_str.split(','):
                if core_str.strip() == 'off':
                    cpu_loads.append(0)
                else:
                    try:
                        # Extract load percentage before @
                        load_pct = int(core_str.split('@')[0].replace('%', ''))
                        cpu_loads.append(load_pct)
                    except (ValueError, IndexError):
                        cpu_loads.append(0)
            return cpu_loads
        return []

    def _extract_gpu_freq(self, raw_line: str) -> Optional[int]:
        """Extract GPU frequency percentage."""
        match = self.GPU_FREQ_PATTERN.search(raw_line)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_emc_freq(self, raw_line: str) -> Optional[int]:
        """Extract EMC frequency percentage (may be missing)."""
        match = self.EMC_FREQ_PATTERN.search(raw_line)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_cpu_temp(self, raw_line: str) -> Optional[float]:
        """Extract CPU temperature in Celsius."""
        match = self.CPU_TEMP_PATTERN.search(raw_line)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_gpu_temp(self, raw_line: str) -> Optional[float]:
        """Extract GPU temperature in Celsius."""
        match = self.GPU_TEMP_PATTERN.search(raw_line)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_tj_temp(self, raw_line: str) -> Optional[float]:
        """Extract thermal junction temperature in Celsius."""
        match = self.TJ_TEMP_PATTERN.search(raw_line)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def _extract_power(self, raw_line: str) -> Optional[int]:
        """Extract power consumption in milliwatts."""
        match = self.POWER_PATTERN.search(raw_line)
        if match:
            try:
                return int(match.group(1))
            except (ValueError, IndexError):
                pass
        return None

    def get_parse_errors(self) -> int:
        """Get count of parse errors encountered."""
        return self._parse_errors

    def reset_parse_errors(self):
        """Reset parse error counter."""
        self._parse_errors = 0


def parse_tegrastats_line(raw_line: str) -> Optional[TegraStatsSample]:
    """
    Convenience function to parse a single tegrastats line.

    Args:
        raw_line: Raw output line from tegrastats

    Returns:
        TegraStatsSample if parsing succeeds, None otherwise
    """
    parser = TegrastatsParser()
    return parser.parse(raw_line)