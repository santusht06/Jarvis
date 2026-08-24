import sqlite3
import json
from datetime import datetime, timezone
from typing import List, Dict, Any, Optional
from config import settings

class Database:
    def __init__(self, db_path: str = settings.SQLITE_DB_PATH):
        self.db_path = db_path
        self.init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def init_db(self):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            
            # Projects Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS projects (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                type TEXT NOT NULL, -- 'local' or 'github'
                path_or_url TEXT NOT NULL,
                description TEXT,
                has_readme INTEGER DEFAULT 0,
                last_indexed_at TEXT,
                last_maintained_at TEXT,
                status TEXT DEFAULT 'pending', -- 'pending', 'active', 'completed', 'failed'
                created_at TEXT NOT NULL
            )
            """)

            # Maintenance Logs Table
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS maintenance_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                project_id TEXT NOT NULL,
                project_name TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                patch_summary TEXT,
                diff_content TEXT,
                status TEXT NOT NULL, -- 'success', 'dry_run', 'failed', 'skipped'
                commit_sha TEXT,
                pr_url TEXT,
                error_message TEXT,
                FOREIGN KEY (project_id) REFERENCES projects (id)
            )
            """)

            # Daily Execution State
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS daily_schedule (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date_str TEXT UNIQUE NOT NULL, -- YYYY-MM-DD
                project_id TEXT NOT NULL,
                status TEXT NOT NULL, -- 'scheduled', 'running', 'completed', 'failed'
                updated_at TEXT NOT NULL
            )
            """)

            conn.commit()

    def upsert_project(self, proj_id: str, name: str, p_type: str, path_or_url: str, description: str = "", has_readme: bool = False):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
            INSERT INTO projects (id, name, type, path_or_url, description, has_readme, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                name = excluded.name,
                type = excluded.type,
                path_or_url = excluded.path_or_url,
                description = excluded.description,
                has_readme = excluded.has_readme
            """, (proj_id, name, p_type, path_or_url, description, 1 if has_readme else 0, now))
            conn.commit()

    def list_projects(self) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects ORDER BY last_maintained_at ASC NULLS FIRST, name ASC")
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_project_by_id(self, proj_id: str) -> Optional[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM projects WHERE id = ?", (proj_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def mark_project_maintained(self, proj_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("UPDATE projects SET last_maintained_at = ?, status = 'completed' WHERE id = ?", (now, proj_id))
            conn.commit()

    def mark_project_indexed(self, proj_id: str):
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("UPDATE projects SET last_indexed_at = ? WHERE id = ?", (now, proj_id))
            conn.commit()

    def add_log(self, project_id: str, project_name: str, patch_summary: str, diff_content: str, status: str, commit_sha: str = "", pr_url: str = "", error_message: str = "") -> int:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            now = datetime.now(timezone.utc).isoformat()
            cursor.execute("""
            INSERT INTO maintenance_logs (project_id, project_name, timestamp, patch_summary, diff_content, status, commit_sha, pr_url, error_message)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (project_id, project_name, now, patch_summary, diff_content, status, commit_sha, pr_url, error_message))
            conn.commit()
            return cursor.lastrowid

    def list_logs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM maintenance_logs ORDER BY id DESC LIMIT ?", (limit,))
            rows = cursor.fetchall()
            return [dict(r) for r in rows]

    def get_daily_project_executed_today(self) -> Optional[Dict[str, Any]]:
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM daily_schedule WHERE date_str = ?", (today_str,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def record_daily_run(self, project_id: str, status: str):
        today_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        now = datetime.now(timezone.utc).isoformat()
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
            INSERT INTO daily_schedule (date_str, project_id, status, updated_at)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(date_str) DO UPDATE SET
                project_id = excluded.project_id,
                status = excluded.status,
                updated_at = excluded.updated_at
            """, (today_str, project_id, status, now))
            conn.commit()

db = Database()
