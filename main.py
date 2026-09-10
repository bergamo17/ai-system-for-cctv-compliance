import os
import sqlite3
import uvicorn
from datetime import datetime
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Query, Request, HTTPException
from fastapi.responses import JSONResponse, FileResponse
from config import (
    DB_PATH, SUMMARY_OUTPUT_DIR
)
from pipeline.db import (
    get_summary_by_id,
    get_list_of_summaries,
    get_pdf_path,
    get_latest_status
)

os.makedirs(SUMMARY_OUTPUT_DIR, exist_ok=True)

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute("SELECT 1 FROM summaries LIMIT 1")
        conn.close()
    except sqlite3.OperationalError as e:
        raise RuntimeError(
            f"Cannot read the 'summaries' table from {DB_PATH}. "
            f"Already run the migration? {e}"
        )

    yield

app = FastAPI(lifespan=lifespan)

@app.get("/summaries")
async def get_list_summaries(
    date_filter: str | None = Query(None, description="Format: YYYY-MM-DD"),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    if date_filter:
        try:
            datetime.strftime(date_filter, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=400,
                detail="Invalid date format, use YYYY-MM-DD"
            )

        summaries = get_list_of_summaries(date_filter, limit, offset)
        return {"date": date_filter, "count": len(summaries), "summaries": summaries}

@app.get("/summaries/{summary_id}")
async def get_summaries(summary_id: int):
    summary = get_summaries(summary_id=summary_id)
    if summary is None:
        raise HTTPException(status_code=404, detail="Summary not found")
    return summary

@app.get("/summaries/{summary_id}/pdf")
async def get_summaries_file(summary_id: int):
    summary_file = get_pdf_path(summary_id=summary_id)
    if summary_file is None:
        raise HTTPException(status_code=404, detail="Summary file not found")
    return FileResponse(
        summary_file,
        media_type="application/pdf",
        filename=summary_file.split("/")[-1],
    )


@app.get("/status")
async def status():
    latest = get_latest_status()
    if latest is None:
        return {"status": "no_data", "latest_session": None}
    return {"status": "ok", "latest_session": latest}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8080, reload=False)