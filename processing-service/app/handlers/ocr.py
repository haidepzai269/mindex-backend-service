from __future__ import annotations

import hashlib
import logging
import os
import tempfile
from pathlib import Path
from urllib.parse import urlparse

import httpx
from PIL import Image, ImageOps
import pytesseract
from pytesseract import Output

from app import cache
from app.handlers.base import BaseHandler, TaskError

logger = logging.getLogger("processing.ocr")

MAX_DIMENSION = 3200
TARGET_TEXT_WIDTH = 1400
MIN_CONFIDENCE = 35.0
PREVIEW_CHARS = 1200


def _configure_tesseract():
    candidates = [
        os.environ.get("TESSERACT_CMD", ""),
        r"C:\Program Files\Tesseract-OCR\tesseract.exe",
        r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
        r"C:\Users\admin\AppData\Local\Programs\Tesseract-OCR\tesseract.exe",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            pytesseract.pytesseract.tesseract_cmd = candidate
            return candidate
    return "tesseract"


def _local_tessdata_config():
    tessdata_dir = Path(__file__).resolve().parent.parent.parent / "tessdata"
    if not tessdata_dir.exists():
        return ""
    if not any(tessdata_dir.glob("*.traineddata")):
        return ""
    return f"--tessdata-dir {tessdata_dir}"


def _select_ocr_languages():
    tessdata_dir = Path(__file__).resolve().parent.parent.parent / "tessdata"
    if (tessdata_dir / "vie.traineddata").exists():
        return "vie"
    if (tessdata_dir / "eng.traineddata").exists():
        return "eng"
    available = set(pytesseract.get_languages(config=""))
    if "vie" in available:
        return "vie"
    if "eng" in available:
        return "eng"
    return "eng"


def _resize_for_ocr(image):
    width, height = image.size
    largest = max(width, height)
    if largest <= MAX_DIMENSION:
        return image
    scale = MAX_DIMENSION / float(largest)
    next_size = (max(1, int(width * scale)), max(1, int(height * scale)))
    return image.resize(next_size, Image.Resampling.LANCZOS)


def _prepare_for_ocr(image):
    image = ImageOps.exif_transpose(image).convert("RGB")
    image = _resize_for_ocr(image)
    width, height = image.size
    if width < TARGET_TEXT_WIDTH:
        scale = min(3.0, TARGET_TEXT_WIDTH / float(width))
        image = image.resize(
            (max(1, int(width * scale)), max(1, int(height * scale))),
            Image.Resampling.LANCZOS,
        )
    image = ImageOps.grayscale(image)
    return ImageOps.autocontrast(image)


def _clean(text):
    return " ".join((text or "").split())


def _is_noise_line(text):
    compact = _clean(text).lower().replace("`", "").replace("'", "").replace(" ", "")
    if not compact:
        return True
    return compact in {"x", "xx", "xxx", "xxxx"}


def _as_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return -1.0


def run_ocr(path):
    _configure_tesseract()
    image = Image.open(path)
    image = _prepare_for_ocr(image)
    width, height = image.size
    languages = _select_ocr_languages()
    tessdata_config = _local_tessdata_config()
    ocr_config = f'{tessdata_config} --oem 1 --psm 6 -c preserve_interword_spaces=1'.strip()
    text_config = f"{tessdata_config} --oem 1 --psm 6".strip()

    text_lines = [
        _clean(line)
        for line in pytesseract.image_to_string(image, lang=languages, config=text_config).splitlines()
    ]
    text_lines = [line for line in text_lines if line and not _is_noise_line(line)]

    data = pytesseract.image_to_data(
        image, lang=languages, output_type=Output.DICT, config=ocr_config,
    )

    line_map = {}
    count = len(data.get("text", []))
    for index in range(count):
        text = _clean(data["text"][index])
        conf = _as_float(data["conf"][index])
        if not text or conf < MIN_CONFIDENCE:
            continue

        key = (
            data.get("block_num", [0] * count)[index],
            data.get("par_num", [0] * count)[index],
            data.get("line_num", [0] * count)[index],
        )
        left = int(data["left"][index])
        top = int(data["top"][index])
        word_width = int(data["width"][index])
        word_height = int(data["height"][index])

        line = line_map.setdefault(
            key,
            {
                "words": [], "conf": [],
                "left": left, "top": top,
                "right": left + word_width, "bottom": top + word_height,
            },
        )
        line["words"].append(text)
        line["conf"].append(conf)
        line["left"] = min(line["left"], left)
        line["top"] = min(line["top"], top)
        line["right"] = max(line["right"], left + word_width)
        line["bottom"] = max(line["bottom"], top + word_height)

    blocks = []
    data_lines = []
    for key in sorted(line_map.keys()):
        line = line_map[key]
        text = _clean(" ".join(line["words"]))
        if not text or _is_noise_line(text):
            continue
        avg_conf = sum(line["conf"]) / max(1, len(line["conf"]))
        x = max(0.0, min(100.0, line["left"] / width * 100.0))
        y = max(0.0, min(100.0, line["top"] / height * 100.0))
        w = max(0.0, min(100.0, (line["right"] - line["left"]) / width * 100.0))
        h = max(0.0, min(100.0, (line["bottom"] - line["top"]) / height * 100.0))
        blocks.append({
            "text": text, "confidence": round(avg_conf, 2),
            "x": round(x, 2), "y": round(y, 2),
            "w": round(w, 2), "h": round(h, 2),
        })
        data_lines.append(text)

    if text_lines and len(text_lines) >= len(blocks):
        for index, block in enumerate(blocks):
            block["text"] = text_lines[index]

    full_text = "\n".join(text_lines or data_lines).strip()
    preview = full_text[:PREVIEW_CHARS]
    return {"text": full_text, "preview": preview, "blocks": blocks}


async def _download_image(url: str, dest: str) -> None:
    async with httpx.AsyncClient(timeout=90.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        with open(dest, "wb") as f:
            f.write(resp.content)


class OCRHandler(BaseHandler):
    async def process(self, payload: dict) -> dict:
        import asyncio

        image_url = payload.get("image_url")
        if not image_url:
            raise TaskError("Missing image_url in payload")
        parsed = urlparse(image_url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc:
            raise TaskError("image_url must be a valid http/https URL")

        tmp_dir = tempfile.mkdtemp(prefix="mindex_ocr_")
        tmp_path = os.path.join(tmp_dir, "image.png")

        try:
            await _download_image(image_url, tmp_path)

            img = Image.open(tmp_path)
            pixel_hash = hashlib.sha256(img.convert("RGB").tobytes()).hexdigest()
            img.close()

            cache_key = f"ocr:{pixel_hash}"
            cached = await cache.get(cache_key)
            if cached is not None:
                logger.info("Cache HIT for %s", cache_key[:20])
                return cached

            logger.info("Cache MISS for %s, running OCR", cache_key[:20])
            loop = asyncio.get_running_loop()
            result = await loop.run_in_executor(None, run_ocr, tmp_path)
            await cache.set(cache_key, result)
            return result
        finally:
            try:
                os.remove(tmp_path)
                os.rmdir(tmp_dir)
            except OSError:
                pass
