"""
edgewatch CLI - Main entry point for LLM benchmarking and hardware profiling.
Integrates tegrastats monitoring, Ollama inference, and correlation engine.
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table
from rich.text import Text

from edgewatch.correlation.engine import CorrelationEngine, CorrelatedResult
from edgewatch.analysis.stats import StatisticalEngine
from edgewatch.ollama.client import InferenceRequest, InferenceResult, OllamaClient
from edgewatch.report.json_exporter import JSONExporter
from edgewatch.storage.db import DatabaseManager
from edgewatch.tegrastats.sampler import TegrastatsSampler

app = typer.Typer(
    name="jwatch",
    help="🟢 jWATCH - Jetson Workload Analysis & Thermal Hardware Correlation - LLM benchmarking for NVIDIA Jetson",
    add_completion=False,
    no_args_is_help=True
)

console = Console()


def show_branding() -> None:
    """Display the jWATCH branding in the terminal."""
    logo = Text(
        "\n".join(
            [
                "      _ _    _   _  ___  _____ ___ _   _ ",
                "     (_) |  | | | |/ _ \\|_   _|_ _| \\ | |",
                "      _| |  | | | | | | | | |  | ||  \\| |",
                "     | | |__| |_| | |_| | | |  | || |\\  |",
                "     | |_____\\___/ \\___/  |_| |___|_| \\_|",
                "    _/ |                                   ",
                "   |__/ WATCH                              ",
            ]
        ),
        style="bold green",
    )

    panel = Panel(
        logo,
        border_style="green",
        padding=(1, 2),
    )

    console.print()
    console.print(panel)
    console.print()


def show_command_header(command_name: str) -> None:
    """Display header for a specific command."""
    header = Text()
    header.append("jWATCH", style="bold green")
    header.append(f" > {command_name}", style="bold white")

    console.print()
    console.print(header)
    console.print()


@app.command()
def bench(
    model: str = typer.Option(..., "--model", "-m", help="Ollama model name (e.g., 'qwen:4b')"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt for inference"),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of benchmark runs", min=1),
    interval_ms: int = typer.Option(100, "--interval", "-i", help="Tegrastats sampling interval in ms", min=50),
    mock_mode: bool = typer.Option(False, "--mock", help="Use mock mode for testing"),
    output_format: str = typer.Option("rich", "--format", "-f", help="Output format (rich/json)"),
) -> None:
    """
    Benchmark a single LLM model with hardware correlation.

    Runs multiple inference passes while monitoring hardware metrics,
    then correlates the data to show performance characteristics.
    """
    show_branding()
    show_command_header("Benchmark")

    try:
        asyncio.run(run_benchmark(
            model=model,
            prompt=prompt,
            runs=runs,
            interval_ms=interval_ms,
            mock_mode=mock_mode,
            output_format=output_format
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Benchmark interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error during benchmark: {e}[/red]")
        sys.exit(1)


@app.command()
def compare(
    models: list[str] = typer.Option(..., "--model", "-m", help="Models to compare (can specify multiple)"),
    prompt: str = typer.Option(..., "--prompt", "-p", help="Prompt for inference"),
    runs: int = typer.Option(5, "--runs", "-r", help="Number of benchmark runs per model", min=1),
    interval_ms: int = typer.Option(100, "--interval", "-i", help="Tegrastats sampling interval in ms", min=50),
    mock_mode: bool = typer.Option(False, "--mock", help="Use mock mode for testing"),
) -> None:
    """
    Compare multiple LLM models under identical hardware conditions.

    Benchmarks each model and displays a comparison table with
    performance metrics and thermal characteristics.
    """
    show_branding()
    show_command_header("Compare Models")

    try:
        asyncio.run(run_comparison(
            models=models,
            prompt=prompt,
            runs=runs,
            interval_ms=interval_ms,
            mock_mode=mock_mode
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Comparison interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error during comparison: {e}[/red]")
        sys.exit(1)


@app.command()
def monitor(
    interval_ms: int = typer.Option(500, "--interval", "-i", help="Sampling interval in ms", min=100),
    mock_mode: bool = typer.Option(False, "--mock", help="Use mock mode for testing"),
    duration: Optional[int] = typer.Option(None, "--duration", "-d", help="Monitor duration in seconds"),
) -> None:
    """
    Live hardware monitoring without inference.

    Displays real-time hardware metrics from tegrastats.
    Useful for monitoring system state and identifying thermal issues.
    """
    show_branding()
    show_command_header("Hardware Monitor")

    try:
        asyncio.run(run_monitor(
            interval_ms=interval_ms,
            mock_mode=mock_mode,
            duration=duration
        ))
    except KeyboardInterrupt:
        console.print("\n[yellow]Monitor stopped by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]Error during monitoring: {e}[/red]")
        sys.exit(1)


@app.command()
def check() -> None:
    """
    Check Ollama connection and available models.

    Verifies that Ollama is accessible and lists available models.
    Also checks tegrastats availability.
    """
    show_branding()
    show_command_header("System Check")

    try:
        asyncio.run(run_checks())
    except Exception as e:
        console.print(f"\n[red]Error during checks: {e}[/red]")
        sys.exit(1)


@app.command()
def history(
    last: int = typer.Option(10, "--last", help="Show last N runs"),
    model: Optional[str] = typer.Option(None, "--model", help="Filter by model"),
    device_type: Optional[str] = typer.Option(None, "--device-type", help="Filter by device type"),
) -> None:
    """Show historical benchmark runs from SQLite."""
    show_branding()
    show_command_header("History")
    db = DatabaseManager()
    try:
        runs = db.get_runs(model=model, last_n=last, device_type=device_type)
    finally:
        db.close()

    if not runs:
        console.print("[yellow]No benchmark history found.[/yellow]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Run ID", style="cyan")
    table.add_column("Timestamp")
    table.add_column("Model")
    table.add_column("TPS", style="green")
    table.add_column("TTFT(ms)", style="yellow")
    table.add_column("Device")

    for row in runs:
        table.add_row(
            str(row.get("run_id", ""))[:8],
            str(row.get("timestamp", ""))[:19],
            str(row.get("model", "N/A")),
            f"{row.get('tokens_per_sec_mean', 0.0):.2f}",
            f"{row.get('ttft_mean_ms', 0.0):.1f}",
            str(row.get("device_type", "N/A")),
        )
    console.print(table)


@app.command()
def diff(
    run_id_a: str = typer.Argument(..., help="First run ID"),
    run_id_b: str = typer.Argument(..., help="Second run ID"),
) -> None:
    """Compare two historical runs."""
    show_branding()
    show_command_header("Diff")
    db = DatabaseManager()
    try:
        comparison = db.compare_runs(run_id_a, run_id_b)
    finally:
        db.close()

    if "error" in comparison:
        console.print(f"[red]{comparison['error']}[/red]")
        return

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Metric", style="cyan")
    table.add_column("Delta", style="yellow")

    diffs = comparison.get("metrics_difference", {})
    for key, value in diffs.items():
        table.add_row(key, "N/A" if value is None else f"{float(value):.3f}")

    console.print(
        Panel(
            f"Run A: {comparison['run_id_a']}\nRun B: {comparison['run_id_b']}\n"
            f"Model A: {comparison.get('model_a', 'N/A')}\n"
            f"Model B: {comparison.get('model_b', 'N/A')}",
            title="Run Comparison",
            border_style="cyan",
        )
    )
    console.print(table)


@app.command()
def export(
    output: Path = typer.Option(Path("report.json"), "--output", "-o", help="Output JSON path"),
    run_id: Optional[str] = typer.Option(None, "--run-id", help="Export only one run"),
    last: int = typer.Option(10, "--last", help="Export last N runs when --run-id is not used"),
) -> None:
    """Export benchmark history to JSON."""
    show_branding()
    show_command_header("Export")

    db = DatabaseManager()
    try:
        if run_id:
            row = db.get_run(run_id)
            runs = [row] if row else []
        else:
            runs = db.get_runs(last_n=last)
    finally:
        db.close()

    if not runs:
        console.print("[yellow]No runs found to export.[/yellow]")
        return

    exporter = JSONExporter()
    result = exporter.export_benchmark_runs(runs, output, last_n=last, show_details=True)

    if result.success:
        console.print(
            f"[green]Exported {result.records_exported} run(s) to {result.filepath}[/green]"
        )
    else:
        console.print(f"[red]Export failed: {result.error_message}[/red]")


async def run_benchmark(model: str,
                       prompt: str,
                       runs: int,
                       interval_ms: int,
                       mock_mode: bool,
                       output_format: str) -> None:
    """Run a single model benchmark."""
    console.print(Panel(
        f"[bold cyan]edgewatch Benchmark[/bold cyan]\n\n"
        f"Model: {model}\n"
        f"Prompt: {prompt}\n"
        f"Runs: {runs}\n"
        f"Interval: {interval_ms}ms\n"
        f"Mode: {'Mock' if mock_mode else 'Real'}",
        title="Configuration",
        border_style="cyan"
    ))

    correlation_engine = CorrelationEngine(padding_sec=0.5)

    results = []
    correlated_results = []

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console
    ) as progress:

        for run_num in range(1, runs + 1):
            task = progress.add_task(f"Run {run_num}/{runs}...", total=None)

            async with TegrastatsSampler(interval_ms=interval_ms) as sampler:
                await asyncio.sleep(0.2)  # Warm up sampling

                async with OllamaClient(mock_mode=mock_mode) as ollama:
                    request = InferenceRequest(model=model, prompt=prompt)
                    inference_result = await ollama.infer(request)
                    results.append(inference_result)

                all_samples = list(sampler._samples)

                if all_samples and inference_result.t_request_sent and inference_result.t_last_token:
                    correlated = correlation_engine.correlate(inference_result, all_samples)
                    if correlated:
                        correlated_results.append(correlated)

            progress.update(task, description=f"Run {run_num}/{runs} - Complete")

    if output_format == "json":
        display_json_results(results, correlated_results)
    else:
        display_rich_results(results, correlated_results, model)

    stats_engine = StatisticalEngine(warmup_runs=1 if runs > 1 else 0)
    summary = stats_engine.summarize(
        model=model,
        inference_results=results,
        correlated_results=correlated_results,
    )

    db = DatabaseManager()
    try:
        run_id = db.save_run(summary, prompt=prompt)
    finally:
        db.close()

    console.print(f"[green]Saved benchmark run: {run_id}[/green]")


async def run_comparison(models: list[str],
                         prompt: str,
                         runs: int,
                         interval_ms: int,
                         mock_mode: bool) -> None:
    """Compare multiple models."""
    console.print(Panel(
        f"[bold cyan]edgewatch Model Comparison[/bold cyan]\n\n"
        f"Models: {', '.join(models)}\n"
        f"Prompt: {prompt}\n"
        f"Runs per model: {runs}\n"
        f"Mode: {'Mock' if mock_mode else 'Real'}",
        title="Configuration",
        border_style="cyan"
    ))

    all_model_results = {}

    for model in models:
        console.print(f"\n[yellow]Benchmarking {model}...[/yellow]")

        correlation_engine = CorrelationEngine(padding_sec=0.5)
        results = []
        correlated_results = []

        for run_num in range(1, runs + 1):
            console.print(f"  Run {run_num}/{runs}...", end=" ")

            async with TegrastatsSampler(interval_ms=interval_ms) as sampler:
                await asyncio.sleep(0.2)

                async with OllamaClient(mock_mode=mock_mode) as ollama:
                    request = InferenceRequest(model=model, prompt=prompt)
                    inference_result = await ollama.infer(request)
                    results.append(inference_result)

                all_samples = list(sampler._samples)
                if all_samples and inference_result.t_request_sent:
                    correlated = correlation_engine.correlate(inference_result, all_samples)
                    if correlated:
                        correlated_results.append(correlated)

            console.print("[green]✓[/green]")

        all_model_results[model] = {
            "inference_results": results,
            "correlated_results": correlated_results
        }

    display_comparison_table(all_model_results)


async def run_monitor(interval_ms: int,
                      mock_mode: bool,
                      duration: Optional[int]) -> None:
    """Run hardware monitor."""
    console.print(Panel(
        f"[bold cyan]edgewatch Hardware Monitor[/bold cyan]\n\n"
        f"Interval: {interval_ms}ms\n"
        f"Mode: {'Mock' if mock_mode else 'Real'}\n"
        f"Duration: {duration}s (Ctrl+C to stop)" if duration else "Duration: Continuous (Ctrl+C to stop)",
        title="Configuration",
        border_style="cyan"
    ))

    console.print("\n[green]Starting hardware monitoring...[/green]\n")

    try:
        async with TegrastatsSampler(interval_ms=interval_ms) as sampler:
            start_time = asyncio.get_event_loop().time()

            while True:
                if duration:
                    elapsed = asyncio.get_event_loop().time() - start_time
                    if elapsed >= duration:
                        break

                latest_sample = sampler.get_latest_sample()

                if latest_sample:
                    display_sample(latest_sample)

                await asyncio.sleep(interval_ms / 1000.0)

    except KeyboardInterrupt:
        pass


async def run_checks() -> None:
    """Run system checks."""
    console.print(Panel(
        "[bold cyan]edgewatch System Checks[/bold cyan]",
        title="edgewatch",
        border_style="cyan"
    ))

    console.print("\n[bold]Checking Ollama connection...[/bold]")

    try:
        async with OllamaClient(mock_mode=False) as ollama:
            connected = await ollama.check_connection()

            if connected:
                console.print("[green]✓[/green] Ollama is accessible")

                console.print("\n[bold]Available models:[/bold]")
                models = await ollama.list_models()

                if models:
                    table = Table(show_header=True, header_style="bold magenta")
                    table.add_column("Model", style="cyan")
                    table.add_column("Size", style="green")

                    for model_info in models:
                        name = model_info.get("name", "Unknown")
                        size_bytes = model_info.get("size", 0)
                        size_gb = size_bytes / (1024**3)
                        table.add_row(name, f"{size_gb:.2f} GB")

                    console.print(table)
                else:
                    console.print("[yellow]No models found. Pull a model with: ollama pull <model>[/yellow]")
            else:
                console.print("[red]✗[/red] Could not connect to Ollama")
                console.print("[yellow]Make sure Ollama is running: ollama serve[/yellow]")

    except Exception as e:
        console.print(f"[red]✗[/red] Error checking Ollama: {e}")
        console.print("[yellow]Falling back to mock mode for testing[/yellow]")

    console.print("\n[bold]Checking tegrastats...[/bold]")

    try:
        import asyncio.subprocess
        proc = await asyncio.create_subprocess_exec(
            "tegrastats",
            "--interval",
            "100",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )

        try:
            line = await asyncio.wait_for(proc.stdout.readline(), timeout=1.0)
            if line:
                console.print("[green]✓[/green] tegrastats is available")
            else:
                console.print("[yellow]⚠[/yellow] tegrastats ran but produced no output")
        except asyncio.TimeoutError:
            console.print("[yellow]⚠[/yellow] tegrastats timeout (may still work)")

        proc.terminate()
        await proc.wait()

    except FileNotFoundError:
        console.print("[red]✗[/red] tegrastats not found")
        console.print("[yellow]edgewatch will use mock mode for hardware monitoring[/yellow]")
    except Exception as e:
        console.print(f"[yellow]⚠[/yellow] Could not verify tegrastats: {e}")
        console.print("[yellow]edgewatch will use mock mode for hardware monitoring[/yellow]")

    console.print("\n[green]System checks complete![/green]")


def display_rich_results(inference_results: list[InferenceResult],
                        correlated_results: list[CorrelatedResult],
                        model: str) -> None:
    """Display benchmark results in rich format."""
    console.print("\n[bold cyan]Benchmark Results[/bold cyan]\n")

    if inference_results:
        tokens_per_sec_values = [r.tokens_per_sec_wall for r in inference_results if r.tokens_per_sec_wall]
        ttft_values = [r.ttft_ms for r in inference_results if r.ttft_ms is not None]

        if tokens_per_sec_values:
            avg_tps = sum(tokens_per_sec_values) / len(tokens_per_sec_values)
            min_tps = min(tokens_per_sec_values)
            max_tps = max(tokens_per_sec_values)

            console.print(f"Model: [bold]{model}[/bold]")
            console.print(f"Runs: {len(inference_results)}\n")

            console.print("[bold]Performance Metrics:[/bold]")
            console.print(f"  Average tokens/sec: {avg_tps:.2f}")
            console.print(f"  Range: {min_tps:.2f} - {max_tps:.2f} tokens/sec")

            if ttft_values:
                avg_ttft = sum(ttft_values) / len(ttft_values)
                console.print(f"  Average TTFT: {avg_ttft:.2f} ms")

        if correlated_results:
            avg_gpu_temp = sum(c.avg_gpu_temp for c in correlated_results) / len(correlated_results)
            max_gpu_temp = max(c.max_gpu_temp for c in correlated_results)
            avg_tj_temp = sum(c.avg_tj_temp for c in correlated_results) / len(correlated_results)
            max_tj_temp = max(c.max_tj_temp for c in correlated_results)
            avg_confidence = sum(c.interpolation_confidence for c in correlated_results) / len(correlated_results)

            console.print(f"\n[bold]Thermal Characteristics:[/bold]")
            console.print(f"  Average GPU temp: {avg_gpu_temp:.1f}°C (max: {max_gpu_temp:.1f}°C)")
            console.print(f"  Average TJ temp: {avg_tj_temp:.1f}°C (max: {max_tj_temp:.1f}°C)")
            console.print(f"  Correlation confidence: {avg_confidence:.2f}")

    console.print("\n[green]Benchmark complete![/green]")


def display_comparison_table(all_model_results: dict) -> None:
    """Display model comparison results."""
    console.print("\n[bold cyan]Model Comparison[/bold cyan]\n")

    table = Table(show_header=True, header_style="bold magenta")
    table.add_column("Model", style="cyan")
    table.add_column("Tokens/sec", style="green")
    table.add_column("TTFT (ms)", style="yellow")
    table.add_column("GPU Temp", style="blue")
    table.add_column("TJ Temp", style="blue")
    table.add_column("Confidence", style="magenta")

    for model, results in all_model_results.items():
        inference_results = results["inference_results"]
        correlated_results = results["correlated_results"]

        tokens_per_sec_values = [r.tokens_per_sec_wall for r in inference_results if r.tokens_per_sec_wall]
        ttft_values = [r.ttft_ms for r in inference_results if r.ttft_ms is not None]

        if tokens_per_sec_values:
            avg_tps = sum(tokens_per_sec_values) / len(tokens_per_sec_values)
            std_tps = (sum((x - avg_tps) ** 2 for x in tokens_per_sec_values) / len(tokens_per_sec_values)) ** 0.5
            tps_str = f"{avg_tps:.1f}±{std_tps:.1f}"
        else:
            tps_str = "N/A"

        if ttft_values:
            avg_ttft = sum(ttft_values) / len(ttft_values)
            ttft_str = f"{avg_ttft:.0f}"
        else:
            ttft_str = "N/A"

        if correlated_results:
            avg_gpu_temp = sum(c.avg_gpu_temp for c in correlated_results) / len(correlated_results)
            avg_tj_temp = sum(c.avg_tj_temp for c in correlated_results) / len(correlated_results)
            avg_confidence = sum(c.interpolation_confidence for c in correlated_results) / len(correlated_results)

            gpu_temp_str = f"{avg_gpu_temp:.1f}°C"
            tj_temp_str = f"{avg_tj_temp:.1f}°C"
            confidence_str = f"{avg_confidence:.2f}"
        else:
            gpu_temp_str = "N/A"
            tj_temp_str = "N/A"
            confidence_str = "N/A"

        table.add_row(model, tps_str, ttft_str, gpu_temp_str, tj_temp_str, confidence_str)

    console.print(table)
    console.print("\n[green]Comparison complete![/green]")


def display_sample(sample) -> None:
    """Display a single hardware sample."""
    console.print(f"[bold cyan]Hardware Sample[/bold cyan]  ", end="")
    console.print(f"CPU: {sample.cpu_temp:.1f}°C  ", end="")
    console.print(f"GPU: {sample.gpu_temp:.1f}°C  ", end="")
    console.print(f"TJ: {sample.tj_temp:.1f}°C  ", end="")
    console.print(f"Freq: {sample.gpu_freq_pct}%  ", end="")
    console.print(f"Power: {sample.power_mw}mW  ", end="")
    console.print(f"RAM: {sample.ram_used_mb}MB")


def display_json_results(inference_results: list[InferenceResult],
                        correlated_results: list[CorrelatedResult]) -> None:
    """Display benchmark results in JSON format."""
    import json

    output = {
        "inference_results": [],
        "correlated_results": []
    }

    for result in inference_results:
        output["inference_results"].append({
            "model": result.model,
            "prompt": result.prompt,
            "response_text": result.response_text,
            "t_request_sent": result.t_request_sent,
            "t_first_token": result.t_first_token,
            "t_last_token": result.t_last_token,
            "ttft_ms": result.ttft_ms,
            "total_tokens": result.total_tokens,
            "tokens_per_sec_ollama": result.tokens_per_sec_ollama,
            "tokens_per_sec_wall": result.tokens_per_sec_wall,
            "inference_duration_sec": result.inference_duration_sec,
            "eval_duration_ns": result.eval_duration_ns,
            "response_chunks": result.response_chunks,
            "bytes_received": result.bytes_received
        })

    for result in correlated_results:
        output["correlated_results"].append({
            "model": result.inference.model,
            "avg_gpu_temp": result.avg_gpu_temp,
            "max_gpu_temp": result.max_gpu_temp,
            "avg_tj_temp": result.avg_tj_temp,
            "max_tj_temp": result.max_tj_temp,
            "avg_gpu_freq_pct": result.avg_gpu_freq_pct,
            "avg_power_mw": result.avg_power_mw,
            "peak_power_mw": result.peak_power_mw,
            "avg_ram_used_mb": result.avg_ram_used_mb,
            "peak_ram_used_mb": result.peak_ram_used_mb,
            "sample_count": result.sample_count,
            "interpolation_confidence": result.interpolation_confidence,
            "window_start": result.window_start,
            "window_end": result.window_end,
            "samples_interpolated": result.samples_interpolated,
            "samples_measured": result.samples_measured
        })

    console.print(json.dumps(output, indent=2))


if __name__ == "__main__":
    app()
