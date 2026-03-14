"""Rich output helpers used by jWATCH CLI."""

from dataclasses import dataclass
from typing import Optional

from rich.console import Console
from rich.panel import Panel
from rich.table import Table


@dataclass
class LiveMetrics:
    tokens_per_sec: Optional[float] = None
    ttft_ms: Optional[float] = None
    gpu_temp: Optional[float] = None
    tj_temp: Optional[float] = None
    gpu_freq_pct: Optional[int] = None
    power_mw: Optional[int] = None
    ram_used_mb: Optional[int] = None
    ram_total_mb: Optional[int] = None
    throttle_status: str = "None"


class RichRenderer:
    """Small wrapper for consistent Rich rendering."""

    def __init__(self, console: Optional[Console] = None):
        self.console = console or Console()

    def show_branding(self) -> None:
        logo = "\n".join(
            [
                "      _ _    _   _  ___  _____ ___ _   _ ",
                "     (_) |  | | | |/ _ \\|_   _|_ _| \\ | |",
                "      _| |  | | | | | | | | |  | ||  \\| |",
                "     | | |__| |_| | |_| | | |  | || |\\  |",
                "     | |_____\\___/ \\___/  |_| |___|_| \\_|",
                "    _/ |                                   ",
                "   |__/ WATCH                              ",
            ]
        )
        self.console.print(
            Panel(
                f"[bold green]{logo}[/bold green]",
                border_style="green",
            )
        )

    def show_metrics_table(self, metrics: LiveMetrics) -> None:
        table = Table(show_header=True, header_style="bold cyan")
        table.add_column("Metric")
        table.add_column("Value")

        table.add_row("Tokens/sec", _fmt(metrics.tokens_per_sec, "{:.2f}"))
        table.add_row("TTFT (ms)", _fmt(metrics.ttft_ms, "{:.2f}"))
        table.add_row("GPU Temp", _fmt(metrics.gpu_temp, "{:.1f}C"))
        table.add_row("TJ Temp", _fmt(metrics.tj_temp, "{:.1f}C"))
        table.add_row("GPU Freq", _fmt(metrics.gpu_freq_pct, "{}%"))
        table.add_row("Power", _fmt(metrics.power_mw, "{} mW"))
        table.add_row("RAM", _fmt(metrics.ram_used_mb, "{} MB"))
        table.add_row("Throttle", metrics.throttle_status)

        self.console.print(table)


def _fmt(value, fmt: str) -> str:
    if value is None:
        return "N/A"
    return fmt.format(value)
