import os
from pathlib import Path

from dotenv import load_dotenv

_env_path = Path(__file__).resolve().parent.parent.parent / "backend" / ".env"
if _env_path.exists():
    load_dotenv(_env_path)
else:
    load_dotenv()


def _resolve_redis_url() -> str:
    explicit = os.environ.get("REDIS_URL")
    if explicit:
        return explicit
    if os.environ.get("USE_CLOUD_REDIS", "").lower() == "true":
        return os.environ.get("REDIS_URL_CLOUD", "redis://localhost:6379")
    return os.environ.get("REDIS_URL_LOCAL", "redis://localhost:6379")


REDIS_URL = _resolve_redis_url()
PROCESSING_PORT = int(os.environ.get("PROCESSING_PORT", "8000"))
TASK_REQUEST_QUEUE = os.environ.get("TASK_REQUEST_QUEUE", "mindex:processing:requests")
RESULT_KEY_PREFIX = os.environ.get("RESULT_KEY_PREFIX", "mindex:processing:result:")
RESULT_TTL_SECONDS = int(os.environ.get("RESULT_TTL_SECONDS", "3600"))

def _resolve_gemini_keys() -> list[str]:
    raw = os.environ.get("GEMINI_CHAT_KEYS", "")
    if raw:
        return [k.strip() for k in raw.split(",") if k.strip()]
    single = os.environ.get("GEMINI_API_KEY", "")
    if single:
        return [single.strip()]
    return []

GEMINI_API_KEYS = _resolve_gemini_keys()
GEMINI_VISION_MODEL = os.environ.get("GEMINI_VISION_MODEL", "gemini-2.5-flash-lite")
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL_SIZE", "base")
