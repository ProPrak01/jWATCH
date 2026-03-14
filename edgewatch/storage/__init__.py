"""
Storage module for SQLite persistence of benchmark results.
"""

from edgewatch.storage.db import (
    DatabaseManager,
    get_run,
    save_run,
    get_runs,
    get_recent_runs,
    compare_runs,
    delete_run,
    export_runs_json,
    get_database_info
)

__all__ = [
    "DatabaseManager",
    "get_run",
    "save_run",
    "get_runs",
    "get_recent_runs",
    "compare_runs",
    "delete_run",
    "export_runs_json",
    "get_database_info",
]