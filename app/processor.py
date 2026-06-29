"""
GoalCut — Video Processor
Нарезка по пикам громкости, наложение музыки, цветокоррекция FFmpeg
"""

import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ─── ЦВЕТОВЫЕ ПРЕСЕТЫ (FFmpeg фильтры) ───────────────────────────────────────

COLOR_GRADES = {
    "cinema": (
        "eq=contrast=1.15:brightness=-0.03:saturation=1.1,"
        "curves=r='0/0 0.1/0.08 0.9/0.95 1/1':"
              "g='0/0 0.1/0.09 0.9/0.93 1/1':"
              "b='0/0 0.1/0.12 0.9/0.88 1/0.97',"
        "vignette=PI/6"
    ),
    "vivid": (
        "eq=contrast=1.2:brightness=0.02:saturation=1.5:gamma=0.95"
    ),
    "bw": (
        "hue=s=0,"
        "eq=contrast=1.3:brightness=-0.02"
    ),
    "none": None,
}

# ─── МУЗЫКАЛЬНЫЕ ТРЕКИ (встроенный пак) ──────────────────────────────────────
# Файлы лежат рядом с processor.py в папке music/
# Названия совпадают с ключами стилей

MUSIC_PACK = {
    "epic":   "music/epic.mp3",
    "hiphop": "music/hiphop.mp3",
    "edm":    "music/edm.mp3",
    "rock":   "music/rock.mp3",
}

# ─── УТИЛИТЫ ─────────────────────────────────────────────────────────────────

def run(cmd: list[str], check=True) -> subprocess.CompletedProcess:
    """Запускает FFmpeg-команду, пробрасывает ошибки с текстом stderr."""
    result = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if check and result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-2000:]}")
    return result


def get_duration(path: str) -> float:
    """Возвращает длительность видео в секундах."""
    result = run([
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "json", path
    ])
    data = json.loads(result.stdout)
    return float(data["format"]["duration"])


def extract_audio_pcm(video_path: str, out_path: str, sr: int = 8000):
    """Извлекает аудио в сырой PCM-формат для анализа громкости."""
    run([
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1",          # моно
        "-ar", str(sr),      # частота дискретизации
        "-f", "s16le",       # сырые 16-bit signed ints
        out_path,
    ])


def detect_loud_moments(
    video_path: str,
    clip_duration: int,
    n_clips: int,
    min_gap: float = 5.0,
) -> list[float]:
    """
    Находит временны́е метки пиков громкости (крик фанатов = гол/момент).

    Алгоритм:
    1. Извлекаем аудио в сырой PCM
    2. Считаем RMS-энергию по окнам (0.5 сек)
    3. Применяем скользящее среднее для сглаживания
    4. Берём N наибольших пиков с минимальным расстоянием между ними
    5. Сдвигаем метку назад на 2 сек — чтобы захватить момент до крика
    """
    sr = 8000
    pcm_path = video_path + ".pcm"

    extract_audio_pcm(video_path, pcm_path, sr)

    # Читаем как numpy array
    raw = np.frombuffer(open(pcm_path, "rb").read(), dtype=np.int16).astype(np.float32)
    os.remove(pcm_path)

    # RMS по окнам 0.5 сек
    win = sr // 2
    n_frames = len(raw) // win
    rms = np.array([
        np.sqrt(np.mean(raw[i*win:(i+1)*win]**2))
        for i in range(n_frames)
    ])

    # Сглаживаем (окно 3 фрейма = 1.5 сек)
    kernel = np.ones(3) / 3
    rms_smooth = np.convolve(rms, kernel, mode="same")

    video_duration = get_duration(video_path)

    # Ищем пики с минимальным расстоянием min_gap сек
    min_gap_frames = int(min_gap / 0.5)
    peaks = []
    used = np.zeros(len(rms_smooth), dtype=bool)

    # Итерируем по убыванию энергии
    sorted_idx = np.argsort(rms_smooth)[::-1]
    for idx in sorted_idx:
        if len(peaks) >= n_clips * 2:  # берём с запасом
            break
        # Проверяем что вокруг нет уже выбранного пика
        lo = max(0, idx - min_gap_frames)
        hi = min(len(used), idx + min_gap_frames)
        if not used[lo:hi].any():
            t = idx * 0.5 - 2.0  # сдвиг назад на 2 сек
            t = max(0.0, min(t, video_duration - clip_duration))
            peaks.append(t)
            used[lo:hi] = True

    # Сортируем по времени и берём нужное количество
    peaks = sorted(peaks)[:n_clips]

    # Если пиков меньше чем нужно — добавляем равномерно
    if len(peaks) < n_clips:
        step = video_duration / (n_clips + 1)
        for i in range(1, n_clips + 1):
            t = step * i
            if not any(abs(t - p) < min_gap for p in peaks):
                peaks.append(t)
        peaks = sorted(peaks)[:n_clips]

    return peaks


def cut_clip(
    video_path: str,
    start: float,
    duration: int,
    out_path: str,
    color_grade: Optional[str] = None,
):
    """Вырезает клип с цветокоррекцией."""
    vf = COLOR_GRADES.get(color_grade or "none")

    cmd = [
        "ffmpeg", "-y",
        "-ss", str(start),
        "-i", video_path,
        "-t", str(duration),
        "-c:v", "libx264",
        "-preset", "fast",
        "-crf", "22",
        "-c:a", "aac",
        "-b:a", "192k",
    ]

    if vf:
        cmd += ["-vf", vf]

    cmd += ["-movflags", "+faststart", out_path]
    run(cmd)


def mix_music(
    clip_path: str,
    music_path: str,
    out_path: str,
    music_volume: float = 0.35,
):
    """
    Накладывает музыкальный трек на клип.
    Оригинальный звук сохраняется (volume 1.0), музыка добавляется тихо (0.35).
    """
    run([
        "ffmpeg", "-y",
        "-i", clip_path,
        "-stream_loop", "-1",   # музыка зациклена если короче клипа
        "-i", music_path,
        "-filter_complex",
        f"[0:a]volume=1.0[orig];"
        f"[1:a]volume={music_volume}[music];"
        f"[orig][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
        "-map", "0:v",
        "-map", "[aout]",
        "-c:v", "copy",
        "-c:a", "aac",
        "-b:a", "192k",
        "-shortest",
        out_path,
    ])


def make_zip(clip_paths: list[str], zip_path: str):
    """Упаковывает клипы в ZIP."""
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in clip_paths:
            zf.write(p, os.path.basename(p))


def upload_to_b2(s3_client, bucket: str, local_path: str, key: str) -> str:
    """Загружает файл в Backblaze B2, возвращает ключ объекта."""
    s3_client.upload_file(local_path, bucket, key)
    return key


# ─── ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────────

def process_video(
    job_id: str,
    input_path: str,
    job_dir: str,
    clip_count: int,
    clip_duration: int,
    music_style: str,
    color_grade: str,
    progress_callback: Callable[[str, int], None],
    s3_client,
    bucket: str,
    custom_music_path: Optional[str] = None,
) -> dict:
    """
    Полный пайплайн обработки видео.
    Возвращает {"clips": [{"label": ..., "key": ..., "duration": ...}]}
    """

    job_path   = Path(job_dir)
    clips_dir  = job_path / "clips"
    clips_dir.mkdir(exist_ok=True)

    # ── 1. Анализ видео ───────────────────────────────────────────────────────
    progress_callback("Анализирую аудио, ищу крутые моменты...", 5)

    timestamps = detect_loud_moments(
        video_path=input_path,
        clip_duration=clip_duration,
        n_clips=clip_count,
    )

    progress_callback(f"Найдено {len(timestamps)} моментов, нарезаю клипы...", 25)

    # ── 2. Нарезка клипов ─────────────────────────────────────────────────────
    raw_clips = []
    for i, t in enumerate(timestamps):
        raw_path = str(clips_dir / f"raw_{i+1:02d}.mp4")
        cut_clip(
            video_path=input_path,
            start=t,
            duration=clip_duration,
            out_path=raw_path,
            color_grade=color_grade,
        )
        raw_clips.append(raw_path)
        pct = 25 + int(30 * (i + 1) / len(timestamps))
        progress_callback(f"Нарезка: клип {i+1}/{len(timestamps)}", pct)

    # ── 3. Наложение музыки ───────────────────────────────────────────────────
    mixed_clips = []

    # Определяем источник музыки
    music_path = None
    if music_style == "custom" and custom_music_path:
        music_path = custom_music_path
    elif music_style != "none":
        # Берём из встроенного пака (файлы рядом с processor.py)
        base_dir = Path(__file__).parent
        candidate = base_dir / MUSIC_PACK.get(music_style, "music/epic.mp3")
        if candidate.exists():
            music_path = str(candidate)

    for i, raw_path in enumerate(raw_clips):
        mixed_path = str(clips_dir / f"clip_{i+1:02d}.mp4")
        if music_path:
            progress_callback(f"Накладываю музыку: клип {i+1}/{len(raw_clips)}", 55 + int(15 * (i+1) / len(raw_clips)))
            mix_music(raw_path, music_path, mixed_path)
        else:
            # Без музыки — просто переименовываем
            shutil.copy(raw_path, mixed_path)
        mixed_clips.append(mixed_path)
        os.remove(raw_path)

    # ── 4. Упаковка в ZIP ─────────────────────────────────────────────────────
    progress_callback("Упаковываю архив...", 72)
    zip_path = str(job_path / "highlights.zip")
    make_zip(mixed_clips, zip_path)

    # ── 5. Загрузка в Backblaze B2 ────────────────────────────────────────────
    progress_callback("Загружаю в облако...", 80)

    clip_results = []
    for i, clip_path in enumerate(mixed_clips):
        key = f"{job_id}/clips/clip_{i+1:02d}.mp4"
        progress_callback(f"Загружаю клип {i+1}/{len(mixed_clips)}...", 80 + int(15 * (i+1) / len(mixed_clips)))
        upload_to_b2(s3_client, bucket, clip_path, key)
        clip_results.append({
            "label":    f"Момент {i+1} ({int(timestamps[i]//60):02d}:{int(timestamps[i]%60):02d})",
            "key":      key,
            "duration": clip_duration,
            "start_sec": round(timestamps[i], 1),
        })

    # Загружаем ZIP
    zip_key = f"{job_id}/highlights.zip"
    upload_to_b2(s3_client, bucket, zip_path, zip_key)

    # ── 6. Очистка временных файлов ───────────────────────────────────────────
    progress_callback("Очистка...", 98)
    shutil.rmtree(str(clips_dir), ignore_errors=True)
    os.remove(zip_path)
    os.remove(input_path)

    return {"clips": clip_results}
