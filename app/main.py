"""
GoalCut — FastAPI Backend (Supabase Storage)
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

from supabase import create_client, Client
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from processor import process_video

# ─── CONFIG ───────────────────────────────────────────────────────────────────

SUPABASE_URL    = os.environ["SUPABASE_URL"]
SUPABASE_KEY    = os.environ["SUPABASE_KEY"]
STORAGE_BUCKET  = os.environ.get("STORAGE_BUCKET", "videos")

TEMP_DIR = Path("/tmp/goalcut")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ─── SUPABASE CLIENT ──────────────────────────────────────────────────────────

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# ─── JOB STORE ────────────────────────────────────────────────────────────────

jobs: dict = {}

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="GoalCut API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ───────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str
    step: Optional[str] = None
    progress: int = 0
    error: Optional[str] = None
    clips: Optional[list] = None

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-storage")
def test_storage():
    """Проверяет подключение к Supabase Storage."""
    try:
        buckets = supabase.storage.list_buckets()
        return {"status": "ok", "buckets": [b.name for b in buckets]}
    except Exception as e:
        return {"status": "error", "error": str(e)}


@app.post("/upload", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clip_count:    int = Form(5),
    clip_duration: int = Form(30),
    music_style:   str = Form("epic"),
    color_grade:   str = Form("cinema"),
):
    job_id   = str(uuid.uuid4())
    job_dir  = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    suffix     = Path(file.filename).suffix or ".mp4"
    input_path = job_dir / f"input{suffix}"

    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    jobs[job_id] = {
        "status": "queued", "step": "Видео загружено...",
        "progress": 0, "error": None, "clips": None,
    }

    background_tasks.add_task(
        run_processing,
        job_id=job_id,
        input_path=str(input_path),
        job_dir=str(job_dir),
        clip_count=clip_count,
        clip_duration=clip_duration,
        music_style=music_style,
        color_grade=color_grade,
    )

    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/status/{job_id}", response_model=JobStatus)
def get_status(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/download/{job_id}")
def download_zip(job_id: str):
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(400, "Обработка ещё не завершена")

    zip_key = f"{job_id}/highlights.zip"
    try:
        # Получаем подписанную ссылку на 1 час
        res = supabase.storage.from_(STORAGE_BUCKET).create_signed_url(
            zip_key, 3600
        )
        return {"download_url": res["signedURL"], "expires_in": 3600}
    except Exception as e:
        raise HTTPException(500, f"Не удалось создать ссылку: {e}")


# ─── BACKGROUND TASK ──────────────────────────────────────────────────────────

async def run_processing(
    job_id: str,
    input_path: str,
    job_dir: str,
    clip_count: int,
    clip_duration: int,
    music_style: str,
    color_grade: str,
):
    def update(step: str, progress: int):
        jobs[job_id]["step"]     = step
        jobs[job_id]["progress"] = progress
        jobs[job_id]["status"]   = "processing"

    try:
        jobs[job_id]["status"] = "processing"
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            lambda: process_video(
                job_id=job_id,
                input_path=input_path,
                job_dir=job_dir,
                clip_count=clip_count,
                clip_duration=clip_duration,
                music_style=music_style,
                color_grade=color_grade,
                progress_callback=update,
                storage_client=supabase.storage.from_(STORAGE_BUCKET),
            )
        )

        jobs[job_id].update({
            "status": "done", "progress": 100,
            "step": "Готово!", "clips": result["clips"],
        })

    except Exception as e:
        jobs[job_id].update({
            "status": "error", "error": str(e), "step": "Ошибка",
        })
