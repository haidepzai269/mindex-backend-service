from __future__ import annotations

import json
import logging

import redis.asyncio as aioredis

logger = logging.getLogger("processing.cache")

_redis: aioredis.Redis | None = None
_PREFIX = "mindex:processing:cache:"
_TTL = 60 * 60 * 24


def init(redis_client: aioredis.Redis) -> None:
    global _redis
    _redis = redis_client


async def get(key: str) -> dict | None:
    if _redis is None:
        return None
    try:
        raw = await _redis.get(f"{_PREFIX}{key}")
        if raw is None:
            return None
        return json.loads(raw)
    except Exception:
        logger.debug("Cache miss (error) for %s", key)
        return None


async def set(key: str, value: dict) -> None:
    if _redis is None:
        return
    try:
        await _redis.set(f"{_PREFIX}{key}", json.dumps(value, ensure_ascii=False), ex=_TTL)
    except Exception:
        logger.debug("Cache set failed for %s", key)
