"""JSON export helpers for jWATCH benchmark results."""

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExportConfig:
    schema_version: str = "1.0"
    indent: int = 2
    include_metadata: bool = True


@dataclass
class ExportResult:
    success: bool
    filepath: Path
    records_exported: int
    error_message: Optional[str] = None
    export_duration_ms: float = 0.0
    file_size_bytes: int = 0


class JSONExporter:
    """Export run dictionaries into a consistent JSON schema."""

    def __init__(self, config: Optional[ExportConfig] = None):
        self.config = config or ExportConfig()

    def export_benchmark_runs(
        self,
        results: List[Dict[str, Any]],
        filepath: Path,
        model: Optional[str] = None,
        device_type: Optional[str] = None,
        last_n: int = 10,
        show_details: bool = False,
    ) -> ExportResult:
        start = time.time()
        filtered = self._filter_results(results, model, device_type, last_n)
        payload = self._prepare_export_data(
            filtered,
            show_details=show_details,
            filters={"model": model, "device_type": device_type, "last_n": last_n},
        )

        try:
            filepath.parent.mkdir(parents=True, exist_ok=True)
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=self.config.indent)

            size = filepath.stat().st_size if filepath.exists() else 0
            return ExportResult(
                success=True,
                filepath=filepath,
                records_exported=len(filtered),
                export_duration_ms=(time.time() - start) * 1000.0,
                file_size_bytes=size,
            )
        except Exception as exc:
            return ExportResult(
                success=False,
                filepath=filepath,
                records_exported=0,
                error_message=str(exc),
                export_duration_ms=(time.time() - start) * 1000.0,
            )

    @staticmethod
    def _filter_results(
        results: List[Dict[str, Any]],
        model: Optional[str],
        device_type: Optional[str],
        last_n: int,
    ) -> List[Dict[str, Any]]:
        filtered = list(results)
        if model:
            filtered = [r for r in filtered if model.lower() in str(r.get("model", "")).lower()]
        if device_type:
            filtered = [r for r in filtered if str(r.get("device_type", "")).lower() == device_type.lower()]

        filtered.sort(key=lambda r: str(r.get("timestamp", "")), reverse=True)
        if last_n > 0:
            filtered = filtered[:last_n]
        return filtered

    def _prepare_export_data(
        self,
        results: List[Dict[str, Any]],
        show_details: bool,
        filters: Dict[str, Any],
    ) -> Dict[str, Any]:
        data: Dict[str, Any] = {
            "schema_version": self.config.schema_version,
            "export_timestamp": datetime.now().isoformat(),
            "tool": "jwatch",
            "runs": [self._prepare_run_data(r, show_details) for r in results],
        }

        if self.config.include_metadata:
            data["metadata"] = {
                "filters_applied": filters,
                "records": len(results),
            }
        return data

    @staticmethod
    def _prepare_run_data(result: Dict[str, Any], show_details: bool) -> Dict[str, Any]:
        run_data: Dict[str, Any] = {
            "run_id": result.get("run_id"),
            "timestamp": result.get("timestamp"),
            "model": result.get("model"),
            "device_type": result.get("device_type"),
            "jetpack_version": result.get("jetpack_version"),
            "performance": {
                "tokens_per_sec_mean": result.get("tokens_per_sec_mean"),
                "tokens_per_sec_ci95_low": result.get("tokens_per_sec_ci95_low"),
                "tokens_per_sec_ci95_high": result.get("tokens_per_sec_ci95_high"),
                "ttft_mean_ms": result.get("ttft_mean_ms"),
            },
        }

        if show_details and result.get("raw_json"):
            try:
                run_data["raw"] = json.loads(result["raw_json"])
            except Exception:
                run_data["raw"] = result["raw_json"]

        return run_data
