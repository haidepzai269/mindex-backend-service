from __future__ import annotations

import asyncio
import json
import logging
import time
import traceback

import redis.asyncio as aioredis

from app import config, registry

logger = logging.getLogger("processing.worker")


async def run_queue_consumer(redis_client: aioredis.Redis) -> None:
    logger.info("Queue consumer started, polling %s", config.TASK_REQUEST_QUEUE)
    while True:
        try:
            raw = await redis_client.lpop(config.TASK_REQUEST_QUEUE)
            if raw is None:
                await asyncio.sleep(1)
                continue
            await _handle_task(redis_client, raw)
        except asyncio.CancelledError:
            logger.info("Queue consumer shutting down")
            break
        except Exception:
            logger.exception("Queue consumer error, retrying in 2s")
            await asyncio.sleep(2)


async def _handle_task(redis_client: aioredis.Redis, raw: bytes) -> None:
    try:
        task = json.loads(raw)
    except json.JSONDecodeError:
        logger.error("Invalid JSON in queue: %s", raw[:200])
        return

    task_id = task.get("task_id", "unknown")
    task_type = task.get("task_type", "")
    payload = task.get("payload", {})

    result_key = f"{config.RESULT_KEY_PREFIX}{task_id}"
    start = time.monotonic()

    try:
        handler = registry.get_handler(task_type)
        result_data = await handler.process(payload)
        duration_ms = int((time.monotonic() - start) * 1000)

        result = {
            "task_id": task_id,
            "status": "completed",
            "result": result_data,
            "error": None,
            "duration_ms": duration_ms,
        }
        logger.info("Task %s (%s) completed in %dms", task_id, task_type, duration_ms)
    except Exception as exc:
        duration_ms = int((time.monotonic() - start) * 1000)
        result = {
            "task_id": task_id,
            "status": "failed",
            "result": None,
            "error": str(exc),
            "duration_ms": duration_ms,
        }
        logger.error("Task %s (%s) failed: %s\n%s", task_id, task_type, exc, traceback.format_exc())

    await redis_client.set(result_key, json.dumps(result, ensure_ascii=False), ex=config.RESULT_TTL_SECONDS)
