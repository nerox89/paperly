"""SQLite storage for suggestion cache, processing history, and settings."""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from dataclasses import asdict
from pathlib import Path

from paperly.classifier import ClassificationResult

logger = logging.getLogger(__name__)

_DEFAULT_DB = Path(os.environ.get("PAPERLY_DATA_DIR", ".")) / "paperly.db"


class Database:
    """Lightweight SQLite wrapper for persistent app state."""

    def __init__(self, path: Path | str | None = None) -> None:
        self._path = str(path or _DEFAULT_DB)
        self._conn: sqlite3.Connection | None = None

    def open(self) -> None:
        logger.info("Opening database at %s", self._path)
        self._conn = sqlite3.connect(self._path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._migrate()

    def close(self) -> None:
        if self._conn:
            self._conn.close()
            self._conn = None

    # ------------------------------------------------------------------
    # Schema
    # ------------------------------------------------------------------

    def _migrate(self) -> None:
        assert self._conn
        self._conn.executescript("""
            CREATE TABLE IF NOT EXISTS suggestions (
                doc_id INTEGER PRIMARY KEY,
                result_json TEXT NOT NULL,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL,
                action TEXT NOT NULL,
                old_values_json TEXT,
                new_values_json TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
        """)

    # ------------------------------------------------------------------
    # Suggestion cache
    # ------------------------------------------------------------------

    def get_suggestion(self, doc_id: int) -> ClassificationResult | None:
        assert self._conn
        row = self._conn.execute(
            "SELECT result_json FROM suggestions WHERE doc_id = ?", (doc_id,)
        ).fetchone()
        if not row:
            return None
        data = json.loads(row["result_json"])
        return ClassificationResult(**data)

    def set_suggestion(self, doc_id: int, result: ClassificationResult) -> None:
        assert self._conn
        self._conn.execute(
            "INSERT OR REPLACE INTO suggestions (doc_id, result_json, created_at) "
            "VALUES (?, ?, datetime('now'))",
            (doc_id, json.dumps(asdict(result), ensure_ascii=False)),
        )
        self._conn.commit()

    def clear_suggestion(self, doc_id: int) -> None:
        assert self._conn
        self._conn.execute("DELETE FROM suggestions WHERE doc_id = ?", (doc_id,))
        self._conn.commit()

    def clear_all_suggestions(self) -> None:
        assert self._conn
        self._conn.execute("DELETE FROM suggestions")
        self._conn.commit()

    def suggestion_count(self) -> int:
        assert self._conn
        row = self._conn.execute("SELECT COUNT(*) FROM suggestions").fetchone()
        return row[0] if row else 0

    # ------------------------------------------------------------------
    # History / audit log
    # ------------------------------------------------------------------

    def log_action(
        self,
        doc_id: int,
        action: str,
        *,
        old_values: dict | None = None,
        new_values: dict | None = None,
    ) -> None:
        assert self._conn
        self._conn.execute(
            "INSERT INTO history (doc_id, action, old_values_json, new_values_json) "
            "VALUES (?, ?, ?, ?)",
            (
                doc_id,
                action,
                json.dumps(old_values, ensure_ascii=False) if old_values else None,
                json.dumps(new_values, ensure_ascii=False) if new_values else None,
            ),
        )
        self._conn.commit()

    def get_history(self, limit: int = 50) -> list[dict]:
        assert self._conn
        rows = self._conn.execute(
            "SELECT id, doc_id, action, old_values_json, new_values_json, created_at "
            "FROM history ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [
            {
                "id": r["id"],
                "doc_id": r["doc_id"],
                "action": r["action"],
                "old_values": json.loads(r["old_values_json"]) if r["old_values_json"] else None,
                "new_values": json.loads(r["new_values_json"]) if r["new_values_json"] else None,
                "created_at": r["created_at"],
            }
            for r in rows
        ]
