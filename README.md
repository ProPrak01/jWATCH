<p align="center">
  <img src="assets/logo.png" alt="jWATCH logo" width="560" />
</p>

# jWATCH

`jWATCH` is a CLI tool for benchmarking LLM inference on NVIDIA Jetson and correlating model performance with hardware telemetry.

It runs inference through Ollama and samples device metrics (from `tegrastats`) in parallel, then reports throughput, latency, and thermal context for each run.

## Why this project exists

When you benchmark on edge hardware, raw tokens/sec is not enough. Performance drift is often caused by thermal state, power limits, or memory pressure. jWATCH captures both sides (inference + hardware) so you can compare models under realistic conditions.

## Current capabilities

- Single-model benchmarking (`bench`)
- Multi-model comparison (`compare`)
- Live hardware monitor (`monitor`)
- Environment checks (`check`)
- Local run history in SQLite (`history`)
- Run-to-run diff (`diff`)
- JSON export (`export`)
- Mock mode for non-Jetson development

## Requirements

- Python 3.9+
- Ollama (for real inference)
- NVIDIA Jetson + `tegrastats` (for real hardware telemetry)

For Mac/Linux development without Jetson, use `--mock`.

## Installation

```bash
python3 -m pip install -U pip
python3 -m pip install .
```

If you want test/dev tools:

```bash
python3 -m pip install pytest pytest-asyncio pytest-cov black mypy
```

## Quick start

### 1. Check setup

```bash
python3 -m edgewatch check
```

### 2. Dummy run (works on Mac with mock telemetry)

```bash
python3 -m edgewatch bench \
  --model qwen3.5:0.8b \
  --prompt "Hello" \
  --runs 2 \
  --mock
```

### 3. Compare models

```bash
python3 -m edgewatch compare \
  --model qwen3.5:0.8b \
  --model qwen:4b \
  --prompt "Write a short summary" \
  --runs 3 \
  --mock
```

### 4. History / diff / export

```bash
python3 -m edgewatch history --last 10
python3 -m edgewatch diff <run_id_a> <run_id_b>
python3 -m edgewatch export --output report.json --last 10
```

## Real Jetson run

On Jetson (with Ollama and `tegrastats` available), remove `--mock`:

```bash
python3 -m edgewatch bench --model qwen:4b --prompt "Explain gravity" --runs 5
```

## Output

jWATCH reports:

- tokens/sec (wall-time based)
- TTFT (time to first token)
- correlated thermal/power/memory metrics
- summary table for comparison runs
- persistent run records (SQLite)

## Project layout

```text
edgewatch/
  analysis/
  correlation/
  ollama/
  report/
  storage/
  tegrastats/
  edgewatch/cli.py
tests/
```

## Notes

- `tegrastats` is Jetson-only.
- On non-Jetson machines, use `--mock` for telemetry.
- If `python3 -m edgewatch` fails due to missing dependencies, install with `pip install .` and retry.

## License

MIT
