"""
GoalCut — Video Processor (Supabase Storage)
"""

import os
import json
import shutil
import subprocess
import zipfile
from pathlib import Path
from typing import Callable, Optional

import numpy as np

# ─── ЦВЕТОВЫЕ ПРЕСЕТЫ ─────────────────────────────────────────────────────────

COLOR_GRADES = {
    "cinema": (
        "eq=contrast=1.15:brightness=-0.03:saturation=1.1,"
        "curves=r='0/0 0.1/0.08 0.9/0.95 1/1':"
              "g='0/0 0.1/0.09 0.9/0.93 1/1':"
              "b='0/0 0.1/0.12 0.9/0.88 1/0.97',"
        "vignette=PI/6"
    ),
    "vivid": "eq=contrast=1.2:brightness=0.02:saturation=1.5:gamma=0.95",
    "bw":    "hue=s=0,eq=contrast=1.3:brightness=-0.02",
    "none":  None,
}

MUSIC_PACK = {
    "epic":   "music/epic.mp3",
    "hiphop": "music/hiphop.mp3",
    "edm":    "music/edm.mp3",
    "rock":   "music/rock.mp3",
}

# ─── УТИЛИТЫ ──────────────────────────────────────────────────────────────────

def run(cmd):
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"FFmpeg error:\n{result.stderr[-2000:]}")
    return result


def get_duration(path):
    result = run(["ffprobe","-v","error","-show_entries","format=duration","-of","json", path])
    return float(json.loads(result.stdout)["format"]["duration"])


def extract_audio_pcm(video_path, out_path, sr=8000):
    run(["ffmpeg","-y","-i",video_path,"-ac","1","-ar",str(sr),"-f","s16le", out_path])


def detect_loud_moments(video_path, clip_duration, n_clips, min_gap=5.0):
    sr = 8000
    pcm_path = video_path + ".pcm"
    extract_audio_pcm(video_path, pcm_path, sr)

    raw = np.frombuffer(open(pcm_path,"rb").read(), dtype=np.int16).astype(np.float32)
    os.remove(pcm_path)

    win = sr // 2
    n_frames = len(raw) // win
    rms = np.array([np.sqrt(np.mean(raw[i*win:(i+1)*win]**2)) for i in range(n_frames)])
    rms_smooth = np.convolve(rms, np.ones(3)/3, mode="same")

    video_duration = get_duration(video_path)
    min_gap_frames = int(min_gap / 0.5)
    peaks = []
    used  = np.zeros(len(rms_smooth), dtype=bool)

    for idx in np.argsort(rms_smooth)[::-1]:
        if len(peaks) >= n_clips * 2:
            break
        lo = max(0, idx - min_gap_frames)
        hi = min(len(used), idx + min_gap_frames)
        if not used[lo:hi].any():
            t = max(0.0, min(idx * 0.5 - 2.0, video_duration - clip_duration))
            peaks.append(t)
            used[lo:hi] = True

    peaks = sorted(peaks)[:n_clips]

    if len(peaks) < n_clips:
        step = video_duration / (n_clips + 1)
        for i in range(1, n_clips + 1):
            t = step * i
            if not any(abs(t - p) < min_gap for p in peaks):
                peaks.append(t)
        peaks = sorted(peaks)[:n_clips]

    return peaks


def cut_clip(video_path, start, duration, out_path, color_grade=None):
    vf = COLOR_GRADES.get(color_grade or "none")
    cmd = ["ffmpeg","-y","-ss",str(start),"-i",video_path,"-t",str(duration),
           "-c:v","libx264","-preset","fast","-crf","22","-c:a","aac","-b:a","192k"]
    if vf:
        cmd += ["-vf", vf]
    cmd += ["-movflags","+faststart", out_path]
    run(cmd)


def mix_music(clip_path, music_path, out_path, music_volume=0.35):
    run(["ffmpeg","-y","-i",clip_path,"-stream_loop","-1","-i",music_path,
         "-filter_complex",
         f"[0:a]volume=1.0[orig];[1:a]volume={music_volume}[music];[orig][music]amix=inputs=2:duration=first:dropout_transition=2[aout]",
         "-map","0:v","-map","[aout]","-c:v","copy","-c:a","aac","-b:a","192k","-shortest", out_path])


def make_zip(clip_paths, zip_path):
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in clip_paths:
            zf.write(p, os.path.basename(p))


def upload_to_supabase(storage_client, local_path: str, key: str):
    """Загружает файл в Supabase Storage."""
    with open(local_path, "rb") as f:
        data = f.read()

    # Определяем content-type
    ext = Path(local_path).suffix.lower()
    content_type = "video/mp4" if ext == ".mp4" else "application/zip"

    storage_client.upload(
        path=key,
        file=data,
        file_options={"content-type": content_type, "upsert": "true"},
    )
    return key


# ─── ГЛАВНАЯ ФУНКЦИЯ ──────────────────────────────────────────────────────────

def process_video(
    job_id, input_path, job_dir,
    clip_count, clip_duration,
    music_style, color_grade,
    progress_callback, storage_client,
    custom_music_path=None,
):
    job_path  = Path(job_dir)
    clips_dir = job_path / "clips"
    clips_dir.mkdir(exist_ok=True)

    # 1. Анализ
    progress_callback("Анализирую аудио, ищу крутые моменты...", 5)
    timestamps = detect_loud_moments(input_path, clip_duration, clip_count)
    progress_callback(f"Найдено {len(timestamps)} моментов, нарезаю клипы...", 25)

    # 2. Нарезка
    raw_clips = []
    for i, t in enumerate(timestamps):
        raw_path = str(clips_dir / f"raw_{i+1:02d}.mp4")
        cut_clip(input_path, t, clip_duration, raw_path, color_grade)
        raw_clips.append(raw_path)
        progress_callback(f"Нарезка: клип {i+1}/{len(timestamps)}", 25 + int(30*(i+1)/len(timestamps)))

    # 3. Музыка
    base_dir    = Path(__file__).parent
    music_path  = None
    if music_style == "custom" and custom_music_path:
        music_path = custom_music_path
    elif music_style != "none":
        candidate = base_dir / MUSIC_PACK.get(music_style, "music/epic.mp3")
        if candidate.exists():
            music_path = str(candidate)

    mixed_clips = []
    for i, raw_path in enumerate(raw_clips):
        mixed_path = str(clips_dir / f"clip_{i+1:02d}.mp4")
        if music_path:
            progress_callback(f"Накладываю музыку: клип {i+1}/{len(raw_clips)}", 55 + int(15*(i+1)/len(raw_clips)))
            mix_music(raw_path, music_path, mixed_path)
        else:
            shutil.copy(raw_path, mixed_path)
        mixed_clips.append(mixed_path)
        os.remove(raw_path)

    # 4. ZIP
    progress_callback("Упаковываю архив...", 72)
    zip_path = str(job_path / "highlights.zip")
    make_zip(mixed_clips, zip_path)

    # 5. Загрузка в Supabase
    clip_results = []
    for i, clip_path in enumerate(mixed_clips):
        key = f"{job_id}/clips/clip_{i+1:02d}.mp4"
        progress_callback(f"Загружаю клип {i+1}/{len(mixed_clips)}...", 80 + int(15*(i+1)/len(mixed_clips)))
        upload_to_supabase(storage_client, clip_path, key)
        clip_results.append({
            "label":     f"Момент {i+1} ({int(timestamps[i]//60):02d}:{int(timestamps[i]%60):02d})",
            "key":       key,
            "duration":  clip_duration,
            "start_sec": round(timestamps[i], 1),
        })

    # ZIP в Supabase
    progress_callback("Загружаю архив...", 96)
    upload_to_supabase(storage_client, zip_path, f"{job_id}/highlights.zip")

    # Очистка
    progress_callback("Очистка...", 98)
    shutil.rmtree(str(clips_dir), ignore_errors=True)
    os.remove(zip_path)
    os.remove(input_path)

    return {"clips": clip_results}
