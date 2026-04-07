"""SQLite storage for suggestion cache, processing history, and settings."""

from __future__ import annotations

import json
import logging
import os
import re
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
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                doc_id INTEGER NOT NULL UNIQUE,
                action TEXT NOT NULL,
                suggested_title TEXT,
                suggested_correspondent_id INTEGER,
                suggested_document_type_id INTEGER,
                suggested_storage_path_id INTEGER,
                suggested_tag_ids TEXT,
                suggested_confidence REAL,
                suggested_reasoning TEXT,
                final_title TEXT,
                final_correspondent_id INTEGER,
                final_document_type_id INTEGER,
                final_storage_path_id INTEGER,
                final_tag_ids TEXT,
                title_changed BOOLEAN DEFAULT 0,
                correspondent_changed BOOLEAN DEFAULT 0,
                document_type_changed BOOLEAN DEFAULT 0,
                storage_path_changed BOOLEAN DEFAULT 0,
                tags_changed BOOLEAN DEFAULT 0,
                accepted_as_is BOOLEAN DEFAULT 0,
                content_preview TEXT,
                provider_name TEXT,
                provider_model TEXT,
                created_at TEXT DEFAULT (datetime('now'))
            );
            CREATE TABLE IF NOT EXISTS correction_rules (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                rule_type TEXT NOT NULL,
                description TEXT NOT NULL,
                prompt_text TEXT NOT NULL,
                source_pattern TEXT,
                auto_generated BOOLEAN DEFAULT 1,
                active BOOLEAN DEFAULT 1,
                hit_count INTEGER DEFAULT 0,
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

    def next_suggestion_doc_id(self, exclude_doc_id: int | None = None) -> int | None:
        """Return the doc_id of the next cached suggestion, or None."""
        assert self._conn
        if exclude_doc_id is not None:
            row = self._conn.execute(
                "SELECT doc_id FROM suggestions WHERE doc_id != ? ORDER BY created_at LIMIT 1",
                (exclude_doc_id,),
            ).fetchone()
        else:
            row = self._conn.execute(
                "SELECT doc_id FROM suggestions ORDER BY created_at LIMIT 1"
            ).fetchone()
        return row[0] if row else None

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

    # ------------------------------------------------------------------
    # Settings (key-value store)
    # ------------------------------------------------------------------

    def get_setting(self, key: str, default: str | None = None) -> str | None:
        assert self._conn
        row = self._conn.execute(
            "SELECT value FROM settings WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else default

    def set_setting(self, key: str, value: str) -> None:
        assert self._conn
        self._conn.execute(
            "INSERT OR REPLACE INTO settings (key, value) VALUES (?, ?)",
            (key, value),
        )
        self._conn.commit()

    def get_all_settings(self) -> dict[str, str]:
        assert self._conn
        rows = self._conn.execute("SELECT key, value FROM settings").fetchall()
        return {r["key"]: r["value"] for r in rows}

    # ------------------------------------------------------------------
    # Feedback (self-learning)
    # ------------------------------------------------------------------

    def record_feedback(
        self,
        doc_id: int,
        action: str,
        *,
        suggestion: ClassificationResult | None = None,
        final_title: str | None = None,
        final_correspondent_id: int | None = None,
        final_document_type_id: int | None = None,
        final_storage_path_id: int | None = None,
        final_tag_ids: list[int] | None = None,
        content_preview: str = "",
    ) -> None:
        """Record what the AI suggested vs what the user chose."""
        assert self._conn

        s_title = suggestion.title if suggestion else None
        s_corr = suggestion.correspondent_id if suggestion else None
        s_dt = suggestion.document_type_id if suggestion else None
        s_sp = suggestion.storage_path_id if suggestion else None
        s_tags = json.dumps(suggestion.tag_ids) if suggestion else None
        s_conf = suggestion.confidence if suggestion else None
        s_reason = suggestion.reasoning if suggestion else None
        s_provider = suggestion.provider_name if suggestion else None
        s_model = suggestion.provider_model if suggestion else None

        f_tags_json = json.dumps(final_tag_ids) if final_tag_ids is not None else None

        # Compute diff flags (for apply/accept/modify actions)
        is_apply = action in ("apply", "accept", "modify")
        title_changed = (s_title or "") != (final_title or "") if is_apply else False
        corr_changed = s_corr != final_correspondent_id if is_apply else False
        dt_changed = s_dt != final_document_type_id if is_apply else False
        sp_changed = s_sp != final_storage_path_id if is_apply else False
        tags_changed = set(suggestion.tag_ids if suggestion else []) != set(final_tag_ids or []) if is_apply else False
        accepted_as_is = is_apply and not any([title_changed, corr_changed, dt_changed, sp_changed, tags_changed])

        # Determine real action based on diffs
        if action == "apply":
            action = "accept" if accepted_as_is else "modify"

        self._conn.execute(
            """INSERT OR REPLACE INTO feedback (
                doc_id, action,
                suggested_title, suggested_correspondent_id, suggested_document_type_id,
                suggested_storage_path_id, suggested_tag_ids, suggested_confidence, suggested_reasoning,
                final_title, final_correspondent_id, final_document_type_id,
                final_storage_path_id, final_tag_ids,
                title_changed, correspondent_changed, document_type_changed,
                storage_path_changed, tags_changed, accepted_as_is,
                content_preview, provider_name, provider_model
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                doc_id, action,
                s_title, s_corr, s_dt, s_sp, s_tags, s_conf, s_reason,
                final_title, final_correspondent_id, final_document_type_id,
                final_storage_path_id, f_tags_json,
                title_changed, corr_changed, dt_changed, sp_changed, tags_changed, accepted_as_is,
                content_preview[:500] if content_preview else "",
                s_provider, s_model,
            ),
        )
        self._conn.commit()

    def get_feedback_stats(self) -> dict:
        """Return aggregate feedback statistics for the learning dashboard."""
        assert self._conn
        total = self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]
        if total == 0:
            return {
                "total": 0, "accepted": 0, "modified": 0, "skipped": 0, "deleted": 0,
                "accepted_pct": 0, "modified_pct": 0,
                "title_accuracy": 0, "correspondent_accuracy": 0,
                "document_type_accuracy": 0, "storage_path_accuracy": 0,
                "tags_accuracy": 0, "overall_accuracy": 0,
            }

        counts = {}
        for row in self._conn.execute(
            "SELECT action, COUNT(*) as cnt FROM feedback GROUP BY action"
        ).fetchall():
            counts[row["action"]] = row["cnt"]

        applied = counts.get("accept", 0) + counts.get("modify", 0)
        if applied == 0:
            field_accuracy = {"title": 0, "correspondent": 0, "document_type": 0, "storage_path": 0, "tags": 0}
        else:
            field_accuracy = {}
            for field in ("title", "correspondent", "document_type", "storage_path", "tags"):
                unchanged = self._conn.execute(
                    f"SELECT COUNT(*) FROM feedback WHERE action IN ('accept','modify') AND {field}_changed = 0"
                ).fetchone()[0]
                field_accuracy[field] = round(unchanged / applied * 100, 1)

        return {
            "total": total,
            "accepted": counts.get("accept", 0),
            "modified": counts.get("modify", 0),
            "skipped": counts.get("skip", 0),
            "deleted": counts.get("delete", 0),
            "accepted_pct": round(counts.get("accept", 0) / total * 100, 1) if total else 0,
            "modified_pct": round(counts.get("modify", 0) / total * 100, 1) if total else 0,
            "title_accuracy": field_accuracy.get("title", 0),
            "correspondent_accuracy": field_accuracy.get("correspondent", 0),
            "document_type_accuracy": field_accuracy.get("document_type", 0),
            "storage_path_accuracy": field_accuracy.get("storage_path", 0),
            "tags_accuracy": field_accuracy.get("tags", 0),
            "overall_accuracy": round(
                self._conn.execute(
                    "SELECT COUNT(*) FROM feedback WHERE accepted_as_is = 1"
                ).fetchone()[0] / applied * 100, 1
            ) if applied else 0,
        }

    def get_similar_examples(self, content_preview: str, *, limit: int = 3) -> list[dict]:
        """Find confirmed examples most similar to the given content for few-shot prompting."""
        assert self._conn
        rows = self._conn.execute(
            """SELECT doc_id, action, final_title, final_correspondent_id,
                      final_document_type_id, final_storage_path_id, final_tag_ids,
                      content_preview, suggested_confidence, created_at
               FROM feedback
               WHERE action IN ('accept', 'modify')
                 AND final_title IS NOT NULL
                 AND content_preview IS NOT NULL AND content_preview != ''
               ORDER BY created_at DESC
               LIMIT 200"""
        ).fetchall()

        if not rows:
            return []

        # Extract keywords from target content (simple TF approach)
        target_words = _extract_keywords(content_preview)
        if not target_words:
            # No keywords — just return most recent examples
            return [_row_to_example(r) for r in rows[:limit]]

        scored: list[tuple[float, dict]] = []
        for r in rows:
            example = _row_to_example(r)
            example_words = _extract_keywords(r["content_preview"] or "")
            overlap = len(target_words & example_words)
            # Bonus for accepted-as-is (higher quality example)
            quality_bonus = 1.0 if r["action"] == "accept" else 0.5
            score = overlap + quality_bonus
            if score > 0:
                scored.append((score, example))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [ex for _, ex in scored[:limit]]

    def get_top_corrections(self, limit: int = 10) -> list[dict]:
        """Return the most frequent correction patterns."""
        assert self._conn
        corrections = []

        # Correspondent corrections
        rows = self._conn.execute(
            """SELECT suggested_correspondent_id as from_id, final_correspondent_id as to_id, COUNT(*) as cnt
               FROM feedback
               WHERE correspondent_changed = 1
                 AND suggested_correspondent_id IS NOT NULL
                 AND final_correspondent_id IS NOT NULL
               GROUP BY suggested_correspondent_id, final_correspondent_id
               ORDER BY cnt DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        for r in rows:
            corrections.append({
                "type": "correspondent",
                "from_id": r["from_id"], "to_id": r["to_id"],
                "count": r["cnt"],
            })

        # Document type corrections
        rows = self._conn.execute(
            """SELECT suggested_document_type_id as from_id, final_document_type_id as to_id, COUNT(*) as cnt
               FROM feedback
               WHERE document_type_changed = 1
                 AND suggested_document_type_id IS NOT NULL
                 AND final_document_type_id IS NOT NULL
               GROUP BY suggested_document_type_id, final_document_type_id
               ORDER BY cnt DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        for r in rows:
            corrections.append({
                "type": "document_type",
                "from_id": r["from_id"], "to_id": r["to_id"],
                "count": r["cnt"],
            })

        corrections.sort(key=lambda x: x["count"], reverse=True)
        return corrections[:limit]

    # ------------------------------------------------------------------
    # Correction rules
    # ------------------------------------------------------------------

    def get_active_rules(self) -> list[dict]:
        """Return all active correction rules."""
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM correction_rules WHERE active = 1 ORDER BY hit_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def get_all_rules(self) -> list[dict]:
        """Return all correction rules (active and inactive)."""
        assert self._conn
        rows = self._conn.execute(
            "SELECT * FROM correction_rules ORDER BY active DESC, hit_count DESC"
        ).fetchall()
        return [dict(r) for r in rows]

    def add_rule(self, rule_type: str, description: str, prompt_text: str,
                 source_pattern: str = "", auto_generated: bool = False) -> int:
        assert self._conn
        cursor = self._conn.execute(
            """INSERT INTO correction_rules (rule_type, description, prompt_text, source_pattern, auto_generated)
               VALUES (?, ?, ?, ?, ?)""",
            (rule_type, description, prompt_text, source_pattern, auto_generated),
        )
        self._conn.commit()
        return cursor.lastrowid  # type: ignore[return-value]

    def toggle_rule(self, rule_id: int) -> None:
        assert self._conn
        self._conn.execute(
            "UPDATE correction_rules SET active = NOT active WHERE id = ?", (rule_id,)
        )
        self._conn.commit()

    def update_rule(self, rule_id: int, description: str, prompt_text: str) -> None:
        """Update description and prompt_text of a rule."""
        assert self._conn
        self._conn.execute(
            "UPDATE correction_rules SET description = ?, prompt_text = ? WHERE id = ?",
            (description, prompt_text, rule_id),
        )
        self._conn.commit()

    def delete_rule(self, rule_id: int) -> None:
        assert self._conn
        self._conn.execute("DELETE FROM correction_rules WHERE id = ?", (rule_id,))
        self._conn.commit()

    def increment_rule_hits(self, rule_ids: list[int]) -> None:
        assert self._conn
        if not rule_ids:
            return
        placeholders = ",".join("?" * len(rule_ids))
        self._conn.execute(
            f"UPDATE correction_rules SET hit_count = hit_count + 1 WHERE id IN ({placeholders})",
            rule_ids,
        )
        self._conn.commit()

    def detect_correction_patterns(self, min_occurrences: int = 3) -> list[dict]:
        """Analyze feedback to find recurring corrections that could become rules.
        
        Includes example document titles for context.
        """
        assert self._conn
        existing_rules = {r["source_pattern"] for r in self.get_all_rules()}
        new_patterns = []

        # Correspondent swaps
        rows = self._conn.execute(
            """SELECT suggested_correspondent_id, final_correspondent_id, COUNT(*) as cnt
               FROM feedback
               WHERE correspondent_changed = 1
                 AND suggested_correspondent_id IS NOT NULL
                 AND final_correspondent_id IS NOT NULL
               GROUP BY suggested_correspondent_id, final_correspondent_id
               HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (min_occurrences,),
        ).fetchall()
        for r in rows:
            pattern = json.dumps({"type": "correspondent", "from": r[0], "to": r[1]})
            if pattern not in existing_rules:
                examples = self._conn.execute(
                    """SELECT COALESCE(final_title, suggested_title, '') as title
                       FROM feedback
                       WHERE correspondent_changed = 1
                         AND suggested_correspondent_id = ?
                         AND final_correspondent_id = ?
                       ORDER BY created_at DESC LIMIT 5""",
                    (r[0], r[1]),
                ).fetchall()
                new_patterns.append({
                    "rule_type": "correspondent",
                    "from_id": r[0], "to_id": r[1], "count": r[2],
                    "source_pattern": pattern,
                    "example_titles": [e[0] for e in examples if e[0]],
                })

        # Document type swaps
        rows = self._conn.execute(
            """SELECT suggested_document_type_id, final_document_type_id, COUNT(*) as cnt
               FROM feedback
               WHERE document_type_changed = 1
                 AND suggested_document_type_id IS NOT NULL
                 AND final_document_type_id IS NOT NULL
               GROUP BY suggested_document_type_id, final_document_type_id
               HAVING cnt >= ?
               ORDER BY cnt DESC""",
            (min_occurrences,),
        ).fetchall()
        for r in rows:
            pattern = json.dumps({"type": "document_type", "from": r[0], "to": r[1]})
            if pattern not in existing_rules:
                examples = self._conn.execute(
                    """SELECT COALESCE(final_title, suggested_title, '') as title
                       FROM feedback
                       WHERE document_type_changed = 1
                         AND suggested_document_type_id = ?
                         AND final_document_type_id = ?
                       ORDER BY created_at DESC LIMIT 5""",
                    (r[0], r[1]),
                ).fetchall()
                new_patterns.append({
                    "rule_type": "document_type",
                    "from_id": r[0], "to_id": r[1], "count": r[2],
                    "source_pattern": pattern,
                    "example_titles": [e[0] for e in examples if e[0]],
                })

        return new_patterns

    def feedback_count(self) -> int:
        assert self._conn
        return self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0]

    def get_confirmed_doc_ids(self, doc_ids: list[int]) -> set[int]:
        """Return which of the given doc IDs have feedback recorded."""
        assert self._conn
        if not doc_ids:
            return set()
        placeholders = ",".join("?" * len(doc_ids))
        rows = self._conn.execute(
            f"SELECT DISTINCT doc_id FROM feedback WHERE doc_id IN ({placeholders})",
            doc_ids,
        ).fetchall()
        return {r[0] for r in rows}

    def get_recent_feedback(self, limit: int = 20) -> list[dict]:
        """Return recent feedback entries for the dashboard."""
        assert self._conn
        rows = self._conn.execute(
            """SELECT id, doc_id, action, accepted_as_is,
                      suggested_title, final_title, title_changed,
                      correspondent_changed, document_type_changed,
                      storage_path_changed, tags_changed,
                      suggested_confidence, provider_name, created_at
               FROM feedback ORDER BY created_at DESC LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]

    def delete_feedback(self, feedback_id: int) -> None:
        """Delete a single feedback record."""
        assert self._conn
        self._conn.execute("DELETE FROM feedback WHERE id = ?", (feedback_id,))
        self._conn.commit()

    def clear_all_feedback(self) -> None:
        """Delete all feedback data and reset learning."""
        assert self._conn
        self._conn.execute("DELETE FROM feedback")
        self._conn.execute("DELETE FROM correction_rules WHERE auto_generated = 1")
        self._conn.commit()


# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = frozenset(
    "der die das ein eine einer eines einem den dem des und oder aber in im "
    "an am auf aus bei für mit nach über um von vor zu zur zum ist sind war "
    "hat haben wird werden kann können ich du er sie es wir ihr sie nicht "
    "auch noch sehr viel nur schon wenn dass ob wie was wer wo als so da "
    "nr tel fax str the and for with from this that".split()
)


def _extract_keywords(text: str, max_keywords: int = 20) -> set[str]:
    """Extract meaningful keywords from text for similarity matching."""
    words = re.findall(r"[a-zäöüß]{3,}", text.lower())
    filtered = [w for w in words if w not in _STOP_WORDS]
    # Simple frequency-based selection
    freq: dict[str, int] = {}
    for w in filtered:
        freq[w] = freq.get(w, 0) + 1
    top = sorted(freq, key=freq.get, reverse=True)[:max_keywords]  # type: ignore[arg-type]
    return set(top)


def _row_to_example(row: sqlite3.Row) -> dict:
    return {
        "doc_id": row["doc_id"],
        "title": row["final_title"],
        "correspondent_id": row["final_correspondent_id"],
        "document_type_id": row["final_document_type_id"],
        "storage_path_id": row["final_storage_path_id"],
        "tag_ids": json.loads(row["final_tag_ids"]) if row["final_tag_ids"] else [],
        "content_preview": row["content_preview"],
    }
