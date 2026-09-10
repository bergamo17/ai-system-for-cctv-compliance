import sqlite3
from config import DB_PATH
from typing import Any

def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def get_summary_by_id(summary_id: int) -> dict[str, Any] | None:
    conn = _get_connection()

    try:
        curr = conn.execute(
            """
            SELECT id, video_name, session_start, ops_score,
                   active_pct, idle_pct, violations_pct,
                   zone_in_pct, zone_out_pct, summary_text, pdf_path
            FROM summaries
            WHERE id = ?
            """,
            (summary_id,),
        )
        row = curr.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()

def get_list_of_summaries(
        date_filter: str | None = None,
        limit: int = 50,
        offset: int = 0,
) -> list[dict[str, Any]]:
    conn = _get_connection()
    try:
        if date_filter:
            curr = conn.execute(
                """
                SELECT id, video_name, session_start, ops_score,
                       active_pct, idle_pct, violations_pct,
                       zone_in_pct, zone_out_pct, pdf_path
                FROM summaries
                WHERE date(session_start) = date(?)
                ORDER BY session_start DESC
                LIMIT ? OFFSET ?
                """,
                (date_filter, limit, offset),
            )
        else:
            curr = conn.execute(
                """
                SELECT id, video_name, session_start, ops_score,
                        active_pct, idle_pct, violations_pct,
                        zone_in_pct, zone_out_pct, pdf_path
                FROM summaries
                ORDER BY session_start DESC
                LIMIT ? OFFSET ?
                """,
                (limit, offset),
            )
        return [dict(row) for row in curr.fetchall()]
    finally:
        conn.close()

def get_pdf_path(summary_id: int) -> str | None:
    conn = _get_connection()
    try:
        curr = conn.execute(
            """
            SELECT pdf_path
            FROM summaries
            WHERE id = ?
            """,
            (summary_id,)
        )
        row = curr.fetchone()
        return row['pdf_path'] if row is not None else None
    finally:
        conn.close()

def get_latest_status() -> dict[str, Any] | None:
    conn = _get_connection()
    try:
        curr = conn.execute(
            """
            SELECT id, video_name, session_start, ops_score
            FROM summaries
            ORDER BY session_start DESC
            LIMIT 1
            """
        )
        row = curr.fetchone()
        return dict(row) if row is not None else None
    finally:
        conn.close()