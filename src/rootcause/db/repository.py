"""SQLite persistence layer for RootCause AI investigations."""

import json
from pathlib import Path
import sqlite3
from typing import Any, Dict, List, Optional, Union

from rootcause.core.models import InvestigationState


class InvestigationRepository:
    """Manages SQLite storage for investigation history, state, and audit trails."""

    def __init__(self, db_path: Optional[Union[Path, str]] = None):
        self.db_path = str(db_path) if db_path else ":memory:"
        if self.db_path == ":memory:":
            self._shared_conn: Optional[sqlite3.Connection] = sqlite3.connect(":memory:", check_same_thread=False)
            self._shared_conn.row_factory = sqlite3.Row
        else:
            self._shared_conn = None
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._shared_conn is not None:
            return self._shared_conn
        conn = sqlite3.connect(self.db_path, check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS investigations (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                repo_path TEXT NOT NULL,
                bug_description TEXT NOT NULL,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                state_json TEXT NOT NULL
            )
        """)
        conn.commit()
        if self._shared_conn is None:
            conn.close()

    def save(self, state: InvestigationState) -> None:
        """Persist or update an investigation state snapshot."""
        state_dict = state.model_dump(mode="json")
        state_json = json.dumps(state_dict)

        conn = self._get_connection()
        try:
            conn.execute("""
                INSERT INTO investigations (id, status, repo_path, bug_description, created_at, updated_at, state_json)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    status=excluded.status,
                    updated_at=excluded.updated_at,
                    state_json=excluded.state_json
            """, (
                state.investigation_id,
                state.status.value,
                state.request.repo_path,
                state.request.bug_description,
                state.created_at.isoformat(),
                state.updated_at.isoformat(),
                state_json,
            ))
            conn.commit()
        finally:
            if self._shared_conn is None:
                conn.close()

    def get(self, investigation_id: str) -> Optional[InvestigationState]:
        """Retrieve an investigation by its ID."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT state_json FROM investigations WHERE id = ?",
                (investigation_id,),
            )
            row = cursor.fetchone()
            if not row:
                return None
            data = json.loads(row["state_json"])
            return InvestigationState.model_validate(data)
        finally:
            if self._shared_conn is None:
                conn.close()

    def list_all(self, limit: int = 50) -> List[Dict[str, Any]]:
        """List summary records of stored investigations."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "SELECT id, status, repo_path, bug_description, created_at, updated_at FROM investigations ORDER BY created_at DESC LIMIT ?",
                (limit,),
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            if self._shared_conn is None:
                conn.close()

    def delete(self, investigation_id: str) -> bool:
        """Delete an investigation record."""
        conn = self._get_connection()
        try:
            cursor = conn.execute(
                "DELETE FROM investigations WHERE id = ?",
                (investigation_id,),
            )
            conn.commit()
            return cursor.rowcount > 0
        finally:
            if self._shared_conn is None:
                conn.close()
