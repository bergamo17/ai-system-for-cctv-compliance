CREATE TABLE summaries (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_name TEXT NOT NULL,
    session_start TEXT NOT NULL,
    ops_score INTEGER,
    active_pct REAL,
    idle_pct REAL,
    violations_pct REAL,
    zone_in_pct REAL,
    zone_out_pct REAL,
    summary_text TEXT,
    pdf_path TEXT
);