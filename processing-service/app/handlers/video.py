from __future__ import annotations

import asyncio
import logging
import os
import subprocess
import tempfile
from pathlib import Path

import httpx

from app.config import GEMINI_API_KEYS, GEMINI_VISION_MODEL, WHISPER_MODEL_SIZE
from app.handlers.base import BaseHandler, TaskError
from app.handlers.ocr import run_ocr

logger = logging.getLogger("processing.video")

MAX_DURATION_SECONDS = 600
OCR_CONFIDENCE_THRESHOLD = 50.0
OCR_MIN_TEXT_LENGTH = 20
MAX_CONTEXT_CHARS = 20000
MAX_TRANSCRIPT_CHARS = 30000


def _get_frame_interval(duration: float) -> float:
    if duration < 60:
        return 3.0
    if duration <= 300:
        return 5.0
    return 10.0


def _format_timestamp(seconds: float) -> str:
    m = int(seconds) // 60
    s = int(seconds) % 60
    return f"{m}:{s:02d}"


async def _download_video(url: str, dest: str) -> None:
    async with httpx.AsyncClient(timeout=90.0) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=65536):
                    f.write(chunk)


def _probe_duration(video_path: str) -> float:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    if result.returncode != 0:
        raise TaskError(f"ffprobe failed: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError:
        raise TaskError("Cannot parse video duration")


def _has_audio_stream(video_path: str) -> bool:
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "a",
        "-show_entries", "stream=codec_type",
        "-of", "default=noprint_wrappers=1:nokey=1",
        video_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    return result.returncode == 0 and "audio" in result.stdout.lower()


def _extract_audio(video_path: str, audio_path: str) -> bool:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-f", "wav",
        audio_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    return result.returncode == 0 and os.path.exists(audio_path) and os.path.getsize(audio_path) > 1000


def _transcribe_audio(audio_path: str) -> str:
    from faster_whisper import WhisperModel

    model = WhisperModel(WHISPER_MODEL_SIZE, device="cpu", compute_type="int8")
    segments, _ = model.transcribe(audio_path, language=None)
    parts = []
    for segment in segments:
        text = segment.text.strip()
        if text:
            parts.append(text)
    return " ".join(parts)


def _extract_frames(video_path: str, output_dir: str, interval: float, duration: float) -> list[dict]:
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "3",
        os.path.join(output_dir, "frame_%04d.jpg"),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=180)
    if result.returncode != 0:
        raise TaskError(f"Frame extraction failed: {result.stderr.strip()[:200]}")

    frames = []
    frame_files = sorted(Path(output_dir).glob("frame_*.jpg"))
    for i, frame_path in enumerate(frame_files):
        timestamp = (i + 1) * interval
        if timestamp > duration:
            break
        frames.append({"path": str(frame_path), "timestamp": timestamp})
    return frames


def _analyze_frame_ocr(frame_path: str) -> dict:
    result = run_ocr(frame_path)
    text = result.get("text", "").strip()
    blocks = result.get("blocks", [])
    if blocks:
        avg_conf = sum(b.get("confidence", 0) for b in blocks) / len(blocks)
    else:
        avg_conf = 0.0
    return {"text": text, "confidence": avg_conf}


def _should_use_vision(ocr_result: dict) -> bool:
    conf = ocr_result.get("confidence", 0)
    text_len = len(ocr_result.get("text", ""))
    return conf < OCR_CONFIDENCE_THRESHOLD or text_len < OCR_MIN_TEXT_LENGTH


def _call_gemini_vision(frame_path: str) -> str:
    if not GEMINI_API_KEYS:
        return "(Vision API not configured)"

    import google.generativeai as genai
    from PIL import Image

    genai.configure(api_key=GEMINI_API_KEYS[0])
    model = genai.GenerativeModel(GEMINI_VISION_MODEL)
    image = Image.open(frame_path)
    response = model.generate_content(
        [
            "Describe this video frame in Vietnamese. Focus on: visual content, diagrams, animations, "
            "charts, or any educational/informational elements. Be concise (1-2 sentences).",
            image,
        ],
        generation_config={"max_output_tokens": 200, "temperature": 0.2},
    )
    image.close()
    return response.text.strip() if response.text else ""


def _assemble_context(
    has_audio: bool,
    transcript: str,
    frames: list[dict],
    original_name: str,
    duration: float,
) -> str:
    parts = []
    parts.append(f"Video: {original_name} ({_format_timestamp(duration)})")
    parts.append("")

    if has_audio and transcript:
        parts.append("[TRANSCRIPT]")
        parts.append(transcript[:MAX_TRANSCRIPT_CHARS])
        parts.append("")
    elif not has_audio:
        parts.append("(Video không có âm thanh, phân tích dựa trên nội dung hình ảnh)")
        parts.append("")

    parts.append("[FRAME DESCRIPTIONS]")
    for frame in frames:
        ts = _format_timestamp(frame["timestamp"])
        if frame["analysis_type"] == "ocr":
            parts.append(f"[{ts}] {frame['ocr_text']}")
        else:
            parts.append(f"[{ts}] (Hình ảnh) {frame['vision_description']}")
    parts.append("")

    context = "\n".join(parts)
    if len(context) > MAX_CONTEXT_CHARS:
        context = context[:MAX_CONTEXT_CHARS]
    return context


class VideoHandler(BaseHandler):
    async def process(self, payload: dict) -> dict:
        video_url = payload.get("video_url")
        if not video_url:
            raise TaskError("Missing video_url in payload")

        duration_hint = payload.get("duration_seconds", 0)
        original_name = payload.get("original_name", "video.mp4")

        tmp_dir = tempfile.mkdtemp(prefix="mindex_video_")
        video_path = os.path.join(tmp_dir, "input_video")
        audio_path = os.path.join(tmp_dir, "audio.wav")
        frames_dir = os.path.join(tmp_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        loop = asyncio.get_running_loop()

        try:
            logger.info("Downloading video from %s", video_url[:80])
            await _download_video(video_url, video_path)

            duration = await loop.run_in_executor(None, _probe_duration, video_path)
            if duration > MAX_DURATION_SECONDS:
                raise TaskError(f"Video duration {duration:.0f}s exceeds {MAX_DURATION_SECONDS}s limit")

            has_audio = await loop.run_in_executor(None, _has_audio_stream, video_path)
            transcript = ""

            if has_audio:
                logger.info("Extracting audio...")
                audio_ok = await loop.run_in_executor(None, _extract_audio, video_path, audio_path)
                if audio_ok:
                    logger.info("Transcribing audio (whisper %s)...", WHISPER_MODEL_SIZE)
                    transcript = await loop.run_in_executor(None, _transcribe_audio, audio_path)
                else:
                    has_audio = False

            interval = _get_frame_interval(duration)
            logger.info("Extracting frames every %.0fs...", interval)
            raw_frames = await loop.run_in_executor(
                None, _extract_frames, video_path, frames_dir, interval, duration
            )

            logger.info("Analyzing %d frames...", len(raw_frames))
            analyzed_frames = []
            last_vision_call = 0.0
            for frame_info in raw_frames:
                ocr_result = await loop.run_in_executor(None, _analyze_frame_ocr, frame_info["path"])

                if _should_use_vision(ocr_result):
                    import time as _time
                    elapsed = _time.monotonic() - last_vision_call
                    if elapsed < 0.2 and last_vision_call > 0:
                        await asyncio.sleep(0.2 - elapsed)
                    try:
                        vision_desc = await loop.run_in_executor(None, _call_gemini_vision, frame_info["path"])
                        last_vision_call = _time.monotonic()
                    except Exception as e:
                        logger.warning("Vision API failed for frame at %.1fs: %s", frame_info["timestamp"], e)
                        vision_desc = ocr_result.get("text", "") or "(Không thể phân tích frame này)"

                    analyzed_frames.append({
                        "timestamp": frame_info["timestamp"],
                        "analysis_type": "vision",
                        "ocr_text": ocr_result.get("text", ""),
                        "ocr_confidence": round(ocr_result.get("confidence", 0), 1),
                        "vision_description": vision_desc,
                    })
                else:
                    analyzed_frames.append({
                        "timestamp": frame_info["timestamp"],
                        "analysis_type": "ocr",
                        "ocr_text": ocr_result.get("text", ""),
                        "ocr_confidence": round(ocr_result.get("confidence", 0), 1),
                        "vision_description": None,
                    })

            context_text = _assemble_context(
                has_audio=has_audio,
                transcript=transcript,
                frames=analyzed_frames,
                original_name=original_name,
                duration=duration,
            )

            logger.info(
                "Video processing complete: %d frames, has_audio=%s, context=%d chars",
                len(analyzed_frames), has_audio, len(context_text),
            )

            return {
                "has_audio": has_audio,
                "audio_transcript": transcript[:MAX_TRANSCRIPT_CHARS],
                "frame_count": len(analyzed_frames),
                "frames": analyzed_frames,
                "context_text": context_text,
            }

        finally:
            import shutil
            try:
                shutil.rmtree(tmp_dir, ignore_errors=True)
            except Exception:
                pass
