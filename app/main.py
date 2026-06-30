"""
GoalCut — FastAPI Backend
Эндпоинты: загрузка видео, статус задачи, скачивание результата
"""

import os
import uuid
import asyncio
from pathlib import Path
from typing import Optional

import boto3
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from botocore.config import Config
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from processor import process_video

# ─── CONFIG ───────────────────────────────────────────────────────────────────

B2_KEY_ID        = os.environ["B2_KEY_ID"]
B2_APP_KEY       = os.environ["B2_APP_KEY"]
B2_BUCKET        = os.environ.get("B2_BUCKET", "goalcut-videos")
B2_ENDPOINT      = os.environ.get("B2_ENDPOINT", "https://s3.eu-central-003.backblaze2.com")

TEMP_DIR = Path("/tmp/goalcut")
TEMP_DIR.mkdir(parents=True, exist_ok=True)

# ─── S3 CLIENT (Backblaze B2 совместим с S3 API) ──────────────────────────────

s3 = boto3.client(
    "s3",
    endpoint_url=B2_ENDPOINT,
    aws_access_key_id=B2_KEY_ID,
    aws_secret_access_key=B2_APP_KEY,
    config=Config(
        signature_version="s3v4",
        connect_timeout=60,
        read_timeout=300,
    ),
    verify=False,  # Backblaze иногда возвращает истёкший SSL cert
)

# ─── JOB STORE (in-memory, для MVP) ───────────────────────────────────────────
# В продакшне заменить на Redis / базу данных

jobs: dict[str, dict] = {}

# ─── APP ──────────────────────────────────────────────────────────────────────

app = FastAPI(title="GoalCut API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # В продакшне заменить на домен фронтенда
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── MODELS ───────────────────────────────────────────────────────────────────

class JobStatus(BaseModel):
    job_id: str
    status: str          # queued | processing | done | error
    step: Optional[str]  # текущий шаг обработки
    progress: int        # 0-100
    error: Optional[str]
    clips: Optional[list[dict]]  # [{label, url, duration}]

# ─── ROUTES ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/test-b2")
def test_b2():
    """Проверяет подключение к Backblaze B2."""
    try:
        response = s3.list_objects_v2(Bucket=B2_BUCKET, MaxKeys=1)
        return {
            "status": "ok",
            "bucket": B2_BUCKET,
            "endpoint": B2_ENDPOINT,
            "objects": response.get("KeyCount", 0),
        }
    except Exception as e:
        return {"status": "error", "error": str(e), "bucket": B2_BUCKET, "endpoint": B2_ENDPOINT}


@app.post("/upload", response_model=JobStatus)
async def upload_video(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    clip_count:    int = Form(5),
    clip_duration: int = Form(30),
    music_style:   str = Form("epic"),   # epic | hiphop | edm | rock | none | custom
    color_grade:   str = Form("cinema"), # cinema | vivid | bw | none
):
    """
    Принимает видеофайл, сохраняет во временную папку,
    запускает обработку в фоне, возвращает job_id.
    """
    # Валидация
    allowed = {"video/mp4", "video/quicktime", "video/x-msvideo",
               "video/x-matroska", "video/webm", "application/octet-stream"}
    if file.content_type not in allowed:
        raise HTTPException(400, f"Неподдерживаемый тип файла: {file.content_type}")

    job_id = str(uuid.uuid4())
    job_dir = TEMP_DIR / job_id
    job_dir.mkdir(parents=True, exist_ok=True)

    # Сохраняем входной файл
    suffix = Path(file.filename).suffix or ".mp4"
    input_path = job_dir / f"input{suffix}"
    with open(input_path, "wb") as f:
        content = await file.read()
        f.write(content)

    # Инициализируем запись о задаче
    jobs[job_id] = {
        "status":   "queued",
        "step":     "Видео загружено, ожидаем очереди...",
        "progress": 0,
        "error":    None,
        "clips":    None,
    }

    # Запускаем обработку в фоне
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
    """Возвращает текущий статус задачи."""
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    return JobStatus(job_id=job_id, **jobs[job_id])


@app.get("/download/{job_id}")
def download_zip(job_id: str):
    """
    Генерирует временную подписанную ссылку на ZIP-архив с клипами.
    Ссылка живёт 1 час.
    """
    if job_id not in jobs:
        raise HTTPException(404, "Задача не найдена")
    if jobs[job_id]["status"] != "done":
        raise HTTPException(400, "Обработка ещё не завершена")

    zip_key = f"{job_id}/highlights.zip"
    try:
        url = s3.generate_presigned_url(
            "get_object",
            Params={"Bucket": B2_BUCKET, "Key": zip_key},
            ExpiresIn=3600,
        )
    except Exception as e:
        raise HTTPException(500, f"Не удалось создать ссылку: {e}")

    return {"download_url": url, "expires_in": 3600}


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
    """Оборачивает синхронный processor в asyncio, обновляет jobs[job_id]."""

    def update(step: str, progress: int):
        jobs[job_id]["step"]     = step
        jobs[job_id]["progress"] = progress
        jobs[job_id]["status"]   = "processing"

    try:
        jobs[job_id]["status"] = "processing"

        # Запускаем тяжёлую работу в отдельном потоке чтобы не блокировать event loop
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
                s3_client=s3,
                bucket=B2_BUCKET,
            )
        )

        jobs[job_id]["status"]   = "done"
        jobs[job_id]["progress"] = 100
        jobs[job_id]["step"]     = "Готово!"
        jobs[job_id]["clips"]    = result["clips"]

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["error"]  = str(e)
        jobs[job_id]["step"]   = "Ошибка обработки"
