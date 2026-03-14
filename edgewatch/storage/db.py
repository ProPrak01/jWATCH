"""SQLite persistence for benchmark results and history queries."""

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from edgewatch.analysis.stats import StatisticalResult


class DatabaseManager:
    """Store and query benchmark run summaries."""

    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or self._get_default_db_path()
        self._connection: Optional[sqlite3.Connection] = None
        self._ensure_database_exists()

    def _get_default_db_path(self) -> str:
        home_dir = Path.home()
        jwatch_dir = home_dir / ".jwatch"
        return str(jwatch_dir / "benchmark_runs.db")

    @property
    def connection(self) -> sqlite3.Connection:
        if self._connection is None:
            self._connection = sqlite3.connect(self.db_path)
            self._connection.row_factory = sqlite3.Row
        return self._connection

    def _ensure_database_exists(self) -> None:
        db_file = Path(self.db_path)
        db_file.parent.mkdir(parents=True, exist_ok=True)
        _ = self.connection
        self._create_schema()

    def _create_schema(self) -> None:
        cursor = self.connection.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id TEXT UNIQUE NOT NULL,
                timestamp TEXT NOT NULL,
                device_type TEXT,
                jetpack_version TEXT,
                model TEXT NOT NULL,
                quantization TEXT,
                prompt_hash TEXT,
                tokens_per_sec_mean REAL NOT NULL,
                tokens_per_sec_ci95_low REAL NOT NULL,
                tokens_per_sec_ci95_high REAL NOT NULL,
                ttft_mean_ms REAL NOT NULL,
                avg_gpu_temp REAL,
                max_gpu_temp REAL,
                avg_tj_temp REAL,
                max_tj_temp REAL,
                avg_gpu_freq_pct INTEGER,
                avg_power_mw INTEGER,
                peak_power_mw INTEGER,
                avg_ram_used_mb INTEGER,
                peak_ram_used_mb INTEGER,
                throttle_events_count INTEGER DEFAULT 0,
                raw_json TEXT NOT NULL
            )
            """
        )
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_model ON benchmark_runs(model)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON benchmark_runs(timestamp DESC)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_device ON benchmark_runs(device_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_run_id ON benchmark_runs(run_id)")
        self.connection.commit()

    def close(self) -> None:
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def save_run(
        self,
        run_result: StatisticalResult,
        prompt: str = "",
        device_type: str = "default",
        jetpack_version: str = "unknown",
    ) -> str:
        run_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        prompt_hash = hashlib.sha256(prompt.encode("utf-8")).hexdigest() if prompt else ""
        quantization = self._extract_quantization(run_result.model)

        thermal = self._aggregate_thermal(run_result)
        raw_json = self._serialize_result(run_result)

        self.connection.execute(
            """
            INSERT INTO benchmark_runs (
                run_id, timestamp, device_type, jetpack_version,
                model, quantization, prompt_hash,
                tokens_per_sec_mean, tokens_per_sec_ci95_low, tokens_per_sec_ci95_high,
                ttft_mean_ms,
                avg_gpu_temp, max_gpu_temp,
                avg_tj_temp, max_tj_temp,
                avg_gpu_freq_pct, avg_power_mw, peak_power_mw,
                avg_ram_used_mb, peak_ram_used_mb,
                throttle_events_count, raw_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                run_id,
                timestamp,
                device_type,
                jetpack_version,
                run_result.model,
                quantization,
                prompt_hash,
                run_result.tokens_per_sec_mean,
                run_result.tokens_per_sec_ci95_low,
                run_result.tokens_per_sec_ci95_high,
                run_result.ttft_mean_ms,
                thermal["avg_gpu_temp"],
                thermal["max_gpu_temp"],
                thermal["avg_tj_temp"],
                thermal["max_tj_temp"],
                thermal["avg_gpu_freq_pct"],
                thermal["avg_power_mw"],
                thermal["peak_power_mw"],
                thermal["avg_ram_used_mb"],
                thermal["peak_ram_used_mb"],
                0,
                raw_json,
            ),
        )
        self.connection.commit()
        return run_id

    def get_run(self, run_id: str) -> Optional[Dict[str, Any]]:
        row = self.connection.execute(
            "SELECT * FROM benchmark_runs WHERE run_id = ?",
            (run_id,),
        ).fetchone()
        return dict(row) if row else None

    def get_runs(
        self,
        model: Optional[str] = None,
        last_n: int = 10,
        device_type: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        query = "SELECT * FROM benchmark_runs"
        params: List[Any] = []
        where: List[str] = []

        if model:
            where.append("model LIKE ?")
            params.append(f"%{model}%")
        if device_type:
            where.append("device_type = ?")
            params.append(device_type)

        if where:
            query += " WHERE " + " AND ".join(where)

        query += " ORDER BY timestamp DESC"
        if last_n > 0:
            query += " LIMIT ?"
            params.append(last_n)

        rows = self.connection.execute(query, tuple(params)).fetchall()
        return [dict(row) for row in rows]

    def get_recent_runs(self, last_n: int = 10) -> List[Dict[str, Any]]:
        return self.get_runs(last_n=last_n)

    def compare_runs(self, run_id_a: str, run_id_b: str) -> Dict[str, Any]:
        run_a = self.get_run(run_id_a)
        run_b = self.get_run(run_id_b)

        if not run_a or not run_b:
            return {
                "error": "One or both runs not found",
                "run_a": run_a,
                "run_b": run_b,
            }

        return {
            "run_id_a": run_id_a,
            "run_id_b": run_id_b,
            "model_a": run_a["model"],
            "model_b": run_b["model"],
            "timestamp_a": run_a["timestamp"],
            "timestamp_b": run_b["timestamp"],
            "metrics_difference": {
                "tokens_per_sec_mean_diff": run_b["tokens_per_sec_mean"] - run_a["tokens_per_sec_mean"],
                "ttft_mean_ms_diff": run_b["ttft_mean_ms"] - run_a["ttft_mean_ms"],
                "avg_gpu_temp_diff": _safe_sub(run_b.get("avg_gpu_temp"), run_a.get("avg_gpu_temp")),
                "avg_tj_temp_diff": _safe_sub(run_b.get("avg_tj_temp"), run_a.get("avg_tj_temp")),
                "avg_power_mw_diff": _safe_sub(run_b.get("avg_power_mw"), run_a.get("avg_power_mw")),
            },
        }

    def delete_run(self, run_id: str) -> bool:
        cursor = self.connection.execute("DELETE FROM benchmark_runs WHERE run_id = ?", (run_id,))
        self.connection.commit()
        return cursor.rowcount > 0

    def export_runs_json(
        self,
        output_path: str,
        model: Optional[str] = None,
        last_n: int = 10,
        device_type: Optional[str] = None,
    ) -> None:
        runs = self.get_runs(model=model, last_n=last_n, device_type=device_type)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(runs, f, indent=2)

    def get_database_info(self) -> Dict[str, Any]:
        count = self.connection.execute("SELECT COUNT(*) FROM benchmark_runs").fetchone()[0]
        size_bytes = Path(self.db_path).stat().st_size if Path(self.db_path).exists() else 0
        return {
            "db_path": self.db_path,
            "runs_count": count,
            "size_bytes": size_bytes,
        }

    def _serialize_result(self, run_result: StatisticalResult) -> str:
        payload = {
            "model": run_result.model,
            "n_runs": run_result.n_runs,
            "warmup_runs": run_result.warmup_runs,
            "effective_runs": run_result.effective_runs,
            "tokens_per_sec_mean": run_result.tokens_per_sec_mean,
            "tokens_per_sec_median": run_result.tokens_per_sec_median,
            "tokens_per_sec_std": run_result.tokens_per_sec_std,
            "tokens_per_sec_ci95_low": run_result.tokens_per_sec_ci95_low,
            "tokens_per_sec_ci95_high": run_result.tokens_per_sec_ci95_high,
            "ttft_mean_ms": run_result.ttft_mean_ms,
            "ttft_std_ms": run_result.ttft_std_ms,
            "inference_duration_mean_sec": run_result.inference_duration_mean_sec,
            "outliers_detected": run_result.outliers_detected,
            "is_stable": run_result.is_stable,
            "total_tokens": run_result.total_tokens,
            "total_duration_sec": run_result.total_duration_sec,
            "correlated_results": [
                {
                    "avg_gpu_temp": c.avg_gpu_temp,
                    "max_gpu_temp": c.max_gpu_temp,
                    "avg_tj_temp": c.avg_tj_temp,
                    "max_tj_temp": c.max_tj_temp,
                    "avg_gpu_freq_pct": c.avg_gpu_freq_pct,
                    "avg_power_mw": c.avg_power_mw,
                    "peak_power_mw": c.peak_power_mw,
                    "avg_ram_used_mb": c.avg_ram_used_mb,
                    "peak_ram_used_mb": c.peak_ram_used_mb,
                    "sample_count": c.sample_count,
                    "interpolation_confidence": c.interpolation_confidence,
                }
                for c in run_result.correlated_results
            ],
        }
        return json.dumps(payload)

    def _aggregate_thermal(self, run_result: StatisticalResult) -> Dict[str, Any]:
        corr = run_result.correlated_results
        if not corr:
            return {
                "avg_gpu_temp": None,
                "max_gpu_temp": None,
                "avg_tj_temp": None,
                "max_tj_temp": None,
                "avg_gpu_freq_pct": None,
                "avg_power_mw": None,
                "peak_power_mw": None,
                "avg_ram_used_mb": None,
                "peak_ram_used_mb": None,
            }

        return {
            "avg_gpu_temp": sum(c.avg_gpu_temp for c in corr) / len(corr),
            "max_gpu_temp": max(c.max_gpu_temp for c in corr),
            "avg_tj_temp": sum(c.avg_tj_temp for c in corr) / len(corr),
            "max_tj_temp": max(c.max_tj_temp for c in corr),
            "avg_gpu_freq_pct": int(round(sum(c.avg_gpu_freq_pct for c in corr) / len(corr))),
            "avg_power_mw": int(round(sum(c.avg_power_mw for c in corr) / len(corr))),
            "peak_power_mw": max(c.peak_power_mw for c in corr),
            "avg_ram_used_mb": int(round(sum(c.avg_ram_used_mb for c in corr) / len(corr))),
            "peak_ram_used_mb": max(c.peak_ram_used_mb for c in corr),
        }

    @staticmethod
    def _extract_quantization(model: str) -> str:
        return model.split(":", 1)[-1] if ":" in model else "unknown"


def _safe_sub(a: Any, b: Any) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(a) - float(b)


def save_run(
    run_result: StatisticalResult,
    prompt: str = "",
    device_type: str = "default",
    jetpack_version: str = "unknown",
    db_path: Optional[str] = None,
) -> str:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.save_run(run_result, prompt=prompt, device_type=device_type, jetpack_version=jetpack_version)
    finally:
        db.close()


def get_run(run_id: str, db_path: Optional[str] = None) -> Optional[Dict[str, Any]]:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.get_run(run_id)
    finally:
        db.close()


def get_runs(
    model: Optional[str] = None,
    last_n: int = 10,
    device_type: Optional[str] = None,
    db_path: Optional[str] = None,
) -> List[Dict[str, Any]]:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.get_runs(model=model, last_n=last_n, device_type=device_type)
    finally:
        db.close()


def get_recent_runs(last_n: int = 10, db_path: Optional[str] = None) -> List[Dict[str, Any]]:
    return get_runs(last_n=last_n, db_path=db_path)


def compare_runs(run_id_a: str, run_id_b: str, db_path: Optional[str] = None) -> Dict[str, Any]:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.compare_runs(run_id_a, run_id_b)
    finally:
        db.close()


def delete_run(run_id: str, db_path: Optional[str] = None) -> bool:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.delete_run(run_id)
    finally:
        db.close()


def export_runs_json(
    output_path: str,
    model: Optional[str] = None,
    last_n: int = 10,
    device_type: Optional[str] = None,
    db_path: Optional[str] = None,
) -> None:
    db = DatabaseManager(db_path=db_path)
    try:
        db.export_runs_json(output_path, model=model, last_n=last_n, device_type=device_type)
    finally:
        db.close()


def get_database_info(db_path: Optional[str] = None) -> Dict[str, Any]:
    db = DatabaseManager(db_path=db_path)
    try:
        return db.get_database_info()
    finally:
        db.close()
