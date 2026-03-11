"""
Job storage adapter implementations.
Provides SQLite and Redis backends for job queue.
"""

import sqlite3
import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from .ports import JobStoragePort


class SQLiteJobStorageAdapter(JobStoragePort):
    """SQLite-based job storage for development."""

    def __init__(self, db_path: str | Path = ":memory:"):
        self.db_path = str(db_path)
        self._is_memory = self.db_path == ":memory:"
        self._conn = None  # Persistent connection for in-memory databases
        self._init_db()

    def _get_connection(self):
        """Get database connection."""
        if self._is_memory:
            if self._conn is None:
                self._conn = sqlite3.connect(":memory:", check_same_thread=False)
            return self._conn
        return sqlite3.connect(self.db_path)

    def _close_connection(self, conn):
        """Close connection if not using in-memory database."""
        if not self._is_memory:
            conn.close()

    def _init_db(self):
        """Initialize database with jobs table."""
        conn = self._get_connection()
        conn.execute("""
            CREATE TABLE IF NOT EXISTS jobs (
                id TEXT PRIMARY KEY,
                status TEXT NOT NULL,
                job_type TEXT NOT NULL,
                engine TEXT NOT NULL,
                data TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                results TEXT,
                error TEXT,
                webhook_url TEXT,
                progress_current INTEGER DEFAULT 0,
                progress_total INTEGER DEFAULT 0
            )
        """)
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_status ON jobs(status)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_jobs_created ON jobs(created_at)")
        conn.commit()
        self._close_connection(conn)

    def _row_to_dict(self, row) -> dict:
        """Convert database row to dictionary."""
        return {
            "job_id": row[0],
            "status": row[1],
            "job_type": row[2],
            "engine": row[3],
            "data": json.loads(row[4]) if row[4] else {},
            "created_at": row[5],
            "updated_at": row[6],
            "started_at": row[7],
            "completed_at": row[8],
            "results": json.loads(row[9]) if row[9] else None,
            "error": row[10],
            "webhook_url": row[11],
            "progress": {
                "current": row[12] or 0,
                "total": row[13] or 0
            }
        }

    async def create_job(self, job_data: dict) -> str:
        """Create a new job."""
        job_id = str(uuid.uuid4())
        conn = self._get_connection()
        conn.execute(
            """INSERT INTO jobs (id, status, job_type, engine, data, webhook_url)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                job_id,
                "queued",
                job_data.get("job_type", "batch"),
                job_data.get("engine", "openvino"),
                json.dumps(job_data.get("data", {})),
                job_data.get("webhook_url")
            )
        )
        conn.commit()
        self._close_connection(conn)
        return job_id

    async def get_job(self, job_id: str) -> dict | None:
        """Get job by ID."""
        conn = self._get_connection()
        cursor = conn.execute("SELECT * FROM jobs WHERE id = ?", (job_id,))
        row = cursor.fetchone()
        self._close_connection(conn)
        return self._row_to_dict(row) if row else None

    async def update_job(self, job_id: str, updates: dict) -> bool:
        """Update job fields."""
        allowed_fields = ['status', 'started_at', 'completed_at', 'results', 'error', 'updated_at', 'progress_current', 'progress_total']
        set_clauses = []
        values = []

        for key, value in updates.items():
            if key in allowed_fields:
                set_clauses.append(f"{key} = ?")
                if isinstance(value, (dict, list)):
                    values.append(json.dumps(value))
                else:
                    values.append(value)

        if not set_clauses:
            return False

        values.append(job_id)
        conn = self._get_connection()
        cursor = conn.execute(
            f"UPDATE jobs SET {', '.join(set_clauses)} WHERE id = ?",
            values
        )
        conn.commit()
        self._close_connection(conn)
        return cursor.rowcount > 0

    async def get_next_pending_job(self) -> dict | None:
        """Get oldest pending job."""
        conn = self._get_connection()
        cursor = conn.execute(
            """SELECT * FROM jobs
               WHERE status = 'queued'
               ORDER BY created_at ASC
               LIMIT 1"""
        )
        row = cursor.fetchone()
        self._close_connection(conn)
        return self._row_to_dict(row) if row else None

    async def list_jobs(self, status: str | None = None, job_type: str | None = None, limit: int = 100, offset: int = 0) -> list[dict]:
        """List jobs with optional status filter."""
        conn = self._get_connection()
        if status:
            cursor = conn.execute(
                """SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (status, limit, offset)
            )
        else:
            cursor = conn.execute(
                """SELECT * FROM jobs ORDER BY created_at DESC LIMIT ? OFFSET ?""",
                (limit, offset)
            )
        rows = cursor.fetchall()
        self._close_connection(conn)
        return [self._row_to_dict(row) for row in rows]

    async def delete_job(self, job_id: str) -> bool:
        """Delete job by ID."""
        conn = self._get_connection()
        cursor = conn.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
        conn.commit()
        self._close_connection(conn)
        return cursor.rowcount > 0

    async def cleanup_old_jobs(self, days: int = 7) -> int:
        """Delete jobs older than specified days."""
        conn = self._get_connection()
        # SQLite datetime function doesn't accept parameters directly
        cursor = conn.execute(
            f"""DELETE FROM jobs WHERE created_at < datetime('now', '-{days} days')"""
        )
        deleted_count = cursor.rowcount
        conn.commit()
        self._close_connection(conn)
        return deleted_count
