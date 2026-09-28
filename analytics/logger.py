"""
Lightweight SQLite event logger.
"""
import json
import os
import sqlite3
from datetime import datetime


DB_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "analytics", "events.db",
)


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts TEXT NOT NULL,
            session_id TEXT NOT NULL,
            event_type TEXT NOT NULL,
            payload TEXT
        )
    """)
    return conn


def log_event(session_id: str, event_type: str, payload: dict = None):
    try:
        conn = _conn()
        conn.execute(
            "INSERT INTO events (ts, session_id, event_type, payload) "
            "VALUES (?, ?, ?, ?)",
            (datetime.utcnow().isoformat(), session_id, event_type,
             json.dumps(payload or {})),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


def fetch_events(session_id: str = None, limit: int = 500):
    conn = _conn()
    if session_id:
        rows = conn.execute(
            "SELECT ts, event_type, payload FROM events "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
    else:
        rows = conn.execute(
            "SELECT ts, session_id, event_type, payload FROM events "
            "ORDER BY id DESC LIMIT ?", (limit,),
        ).fetchall()
    conn.close()
    return rows


def session_summary(session_id: str) -> dict:
    conn = _conn()
    rows = conn.execute(
        "SELECT event_type, COUNT(*) FROM events "
        "WHERE session_id = ? GROUP BY event_type",
        (session_id,),
    ).fetchall()
    conn.close()
    return dict(rows)