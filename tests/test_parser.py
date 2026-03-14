"""
Comprehensive tests for Tegrastats Parser.
Tests various JetPack versions, missing fields, and edge cases.
"""

import pytest
from edgewatch.tegrastats.parser import TegrastatsParser, TegraStatsSample, parse_tegrastats_line


class TestTegrastatsParser:
    """Test suite for TegrastatsParser class."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return TegrastatsParser()

    def test_parse_normal_output(self, parser):
        """Test parsing normal tegrastats output with all fields."""
        raw_line = "04-01-2025 12:00:00 RAM 2048/7772MB (lfb 512x4MB) SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] EMC_FREQ 0% GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert isinstance(sample, TegraStatsSample)
        assert sample.ram_used_mb == 2048
        assert sample.ram_total_mb == 7772
        assert sample.swap_used_mb == 0
        assert sample.swap_total_mb == 3886
        assert sample.cpu_loads == [23, 15, 0, 0]
        assert sample.gpu_freq_pct == 45
        assert sample.emc_freq_pct == 0
        assert sample.cpu_temp == 45.5
        assert sample.gpu_temp == 52.3
        assert sample.tj_temp == 53.0
        assert sample.power_mw == 4521

    def test_parse_high_gpu_usage(self, parser):
        """Test parsing output with high GPU usage."""
        raw_line = "04-01-2025 12:00:01 RAM 3072/7772MB (lfb 256x4MB) SWAP 0/3886MB CPU [85%@1475,78%@1465,off,off] EMC_FREQ 15% GPC_FREQ 89% CPU@58.2C GPU@72.1C tj@76.5C VDD_IN 6892mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.ram_used_mb == 3072
        assert sample.cpu_loads == [85, 78, 0, 0]
        assert sample.gpu_freq_pct == 89
        assert sample.emc_freq_pct == 15
        assert sample.cpu_temp == 58.2
        assert sample.gpu_temp == 72.1
        assert sample.tj_temp == 76.5
        assert sample.power_mw == 6892

    def test_parse_memory_pressure(self, parser):
        """Test parsing output with memory pressure."""
        raw_line = "04-01-2025 12:00:02 RAM 6500/7772MB (lfb 128x4MB) SWAP 256/3886MB CPU [45%@1435,38%@1425,15%@1415,off] EMC_FREQ 8% GPC_FREQ 67% CPU@51.3C GPU@62.8C tj@65.2C VDD_IN 5432mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.ram_used_mb == 6500
        assert sample.swap_used_mb == 256
        assert sample.swap_total_mb == 3886
        assert sample.cpu_loads == [45, 38, 15, 0]
        assert sample.gpu_freq_pct == 67
        assert sample.emc_freq_pct == 8

    def test_parse_thermal_throttling(self, parser):
        """Test parsing output indicating thermal throttling."""
        raw_line = "04-01-2025 12:00:03 RAM 4096/7772MB (lfb 256x4MB) SWAP 128/3886MB CPU [67%@1455,72%@1450,off,off] EMC_FREQ 12% GPC_FREQ 23% CPU@64.8C GPU@78.2C tj@85.1C VDD_IN 6234mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.gpu_freq_pct == 23  # Low GPU frequency indicates throttling
        assert sample.tj_temp == 85.1  # High thermal junction temp
        assert sample.cpu_temp == 64.8
        assert sample.gpu_temp == 78.2

    def test_parse_missing_swap_field(self, parser):
        """Test parsing output with missing swap field (different JetPack version)."""
        raw_line = "04-01-2025 12:00:04 RAM 2048/7772MB (lfb 512x4MB) CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.ram_used_mb == 2048
        assert sample.swap_used_mb is None
        assert sample.swap_total_mb is None
        assert sample.cpu_loads == [23, 15, 0, 0]
        assert sample.gpu_freq_pct == 45

    def test_parse_missing_emc_freq(self, parser):
        """Test parsing output with missing EMC frequency field."""
        raw_line = "04-01-2025 12:00:05 RAM 2048/7772MB (lfb 512x4MB) SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.gpu_freq_pct == 45
        assert sample.emc_freq_pct is None

    def test_parse_all_cores_off(self, parser):
        """Test parsing output with all CPU cores off."""
        raw_line = "04-01-2025 12:00:06 RAM 1024/7772MB (lfb 256x4MB) SWAP 0/3886MB CPU [off,off,off,off] EMC_FREQ 0% GPC_FREQ 10% CPU@35.2C GPU@40.1C tj@42.0C VDD_IN 3210mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.cpu_loads == [0, 0, 0, 0]
        assert sample.cpu_temp == 35.2
        assert sample.gpu_temp == 40.1
        assert sample.tj_temp == 42.0

    def test_parse_single_core_active(self, parser):
        """Test parsing output with only one CPU core active."""
        raw_line = "04-01-2025 12:00:07 RAM 1500/7772MB (lfb 256x4MB) SWAP 0/3886MB CPU [89%@1480,off,off,off] EMC_FREQ 5% GPC_FREQ 35% CPU@55.3C GPU@48.2C tj@58.1C VDD_IN 4876mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.cpu_loads == [89, 0, 0, 0]
        assert sample.gpu_freq_pct == 35

    def test_parse_mixed_cpu_states(self, parser):
        """Test parsing output with mixed CPU core states."""
        raw_line = "04-01-2025 12:00:08 RAM 2500/7772MB (lfb 256x4MB) SWAP 0/3886MB CPU [45%@1435,off,78%@1460,off] EMC_FREQ 3% GPC_FREQ 55% CPU@52.1C GPU@58.4C tj@60.2C VDD_IN 5123mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.cpu_loads == [45, 0, 78, 0]
        assert sample.gpu_freq_pct == 55

    def test_parse_invalid_line(self, parser):
        """Test parsing invalid tegrastats line returns None."""
        invalid_lines = [
            "Invalid tegrastats output",
            "RAM 2048/7772MB",  # Missing critical fields
            "CPU [23%@1420] GPU@52.3C",  # Missing many fields
            "",  # Empty line
            "   ",  # Whitespace only
        ]

        for line in invalid_lines:
            sample = parser.parse(line)
            assert sample is None

    def test_parse_missing_wall_time(self, parser):
        """Test parsing output without wall time (still works)."""
        raw_line = "RAM 2048/7772MB (lfb 512x4MB) SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] EMC_FREQ 0% GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parser.parse(raw_line)

        assert sample is not None
        assert sample.wall_time is None
        assert sample.ram_used_mb == 2048

    def test_parse_error_counting(self, parser):
        """Test that parse errors are counted correctly."""
        valid_line = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"
        invalid_line = "Invalid tegrastats output"

        # Parse valid lines
        parser.parse(valid_line)
        assert parser.get_parse_errors() == 0

        # Parse invalid lines
        parser.parse(invalid_line)
        assert parser.get_parse_errors() == 1

        parser.parse(invalid_line)
        assert parser.get_parse_errors() == 2

        # Reset counter
        parser.reset_parse_errors()
        assert parser.get_parse_errors() == 0

    def test_parse_extreme_values(self, parser):
        """Test parsing output with extreme hardware values."""
        # High temperature
        high_temp_line = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 95% CPU@85.2C GPU@89.5C tj@92.1C VDD_IN 8500mW"
        sample = parser.parse(high_temp_line)
        assert sample is not None
        assert sample.cpu_temp == 85.2
        assert sample.gpu_temp == 89.5
        assert sample.tj_temp == 92.1

        # High power
        high_power_line = "RAM 4096/7772MB SWAP 256/3886MB CPU [85%@1480,78%@1475,off,off] GPC_FREQ 100% CPU@75.5C GPU@82.3C tj@88.0C VDD_IN 12000mW"
        sample = parser.parse(high_power_line)
        assert sample is not None
        assert sample.power_mw == 12000

        # High memory usage
        high_memory_line = "RAM 7500/7772MB SWAP 2048/3886MB CPU [45%@1435,38%@1425,15%@1415,off] GPC_FREQ 67% CPU@60.2C GPU@68.5C tj@72.3C VDD_IN 7100mW"
        sample = parser.parse(high_memory_line)
        assert sample is not None
        assert sample.ram_used_mb == 7500
        assert sample.swap_used_mb == 2048


class TestConvenienceFunction:
    """Test suite for the convenience parse_tegrastats_line function."""

    def test_convenience_function(self):
        """Test the convenience function works correctly."""
        raw_line = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parse_tegrastats_line(raw_line)

        assert sample is not None
        assert isinstance(sample, TegraStatsSample)
        assert sample.ram_used_mb == 2048

    def test_convenience_function_invalid(self):
        """Test the convenience function handles invalid input."""
        invalid_line = "Invalid tegrastats output"

        sample = parse_tegrastats_line(invalid_line)

        assert sample is None


class TestEdgeCases:
    """Test suite for edge cases and unusual inputs."""

    @pytest.fixture
    def parser(self):
        """Create a parser instance for testing."""
        return TegrastatsParser()

    def test_partial_field_match(self, parser):
        """Test handling lines with partial field matches."""
        # Line with some fields but missing critical ones
        partial_line = "RAM 2048/7772MB CPU [23%@1420,15%@1420] EMC_FREQ 5%"

        sample = parser.parse(partial_line)

        # Should return None because critical fields are missing
        assert sample is None

    def test_duplicate_fields(self, parser):
        """Test handling lines with duplicate field occurrences."""
        # Line with potentially ambiguous duplicates
        duplicate_line = "RAM 2048/7772MB RAM 1024/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        # Parser should handle this (first match wins)
        sample = parser.parse(duplicate_line)
        assert sample is not None
        assert sample.ram_used_mb == 2048  # First occurrence

    def test_whitespace_variations(self, parser):
        """Test handling various whitespace patterns."""
        # Extra whitespace
        extra_space_line = "RAM   2048/7772MB   SWAP   0/3886MB   CPU   [23%@1420,15%@1420,off,off]   GPC_FREQ   45%   CPU@45.5C   GPU@52.3C   tj@53.0C   VDD_IN   4521mW"

        sample = parser.parse(extra_space_line)
        assert sample is not None
        assert sample.ram_used_mb == 2048

    def test_missing_lfb_field(self, parser):
        """Test parsing output missing the LFB (Large Frame Buffer) field."""
        no_lfb_line = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] EMC_FREQ 0% GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"

        sample = parser.parse(no_lfb_line)
        assert sample is not None

    def test_temperature_decimal_precision(self, parser):
        """Test handling various temperature decimal precisions."""
        # Single decimal
        single_decimal = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.5C GPU@52.3C tj@53.0C VDD_IN 4521mW"
        sample = parser.parse(single_decimal)
        assert sample.cpu_temp == 45.5

        # Two decimals
        two_decimal = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45.52C GPU@52.34C tj@53.01C VDD_IN 4521mW"
        sample = parser.parse(two_decimal)
        assert sample.cpu_temp == 45.52
        assert sample.gpu_temp == 52.34
        assert sample.tj_temp == 53.01

        # No decimal (integer)
        no_decimal = "RAM 2048/7772MB SWAP 0/3886MB CPU [23%@1420,15%@1420,off,off] GPC_FREQ 45% CPU@45C GPU@52C tj@53C VDD_IN 4521mW"
        sample = parser.parse(no_decimal)
        assert sample.cpu_temp == 45.0
        assert sample.gpu_temp == 52.0
        assert sample.tj_temp == 53.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])