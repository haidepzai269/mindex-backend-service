from __future__ import annotations

import asyncio
import logging

import redis.asyncio as aioredis
from fastapi import FastAPI

from app import cache, config, registry
from app.router import router
from app.worker import run_queue_consumer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger("processing")

app = FastAPI(title="Mindex Processing Service")
app.include_router(router)

_redis_client: aioredis.Redis | None = None
_worker_task: asyncio.Task | None = None


@app.on_event("startup")
async def startup():
    global _redis_client, _worker_task

    from app.handlers import register_all
    register_all()
    logger.info("Registered task types: %s", registry.list_registered())

    _redis_client = aioredis.from_url(config.REDIS_URL, decode_responses=False)
    cache.init(_redis_client)
    _worker_task = asyncio.create_task(run_queue_consumer(_redis_client))
    logger.info("Processing service started on port %d", config.PROCESSING_PORT)


@app.on_event("shutdown")
async def shutdown():
    global _worker_task, _redis_client
    if _worker_task is not None:
        _worker_task.cancel()
        try:
            await _worker_task
        except asyncio.CancelledError:
            pass
    if _redis_client is not None:
        await _redis_client.aclose()
    logger.info("Processing service shut down")


@app.get("/health")
async def health():
    redis_status = "disconnected"
    if _redis_client is not None:
        try:
            await _redis_client.ping()
            redis_status = "connected"
        except Exception:
            redis_status = "disconnected"

    status = "healthy" if redis_status == "connected" else "unhealthy"
    return {
        "status": status,
        "redis": redis_status,
        "registered_tasks": registry.list_registered(),
    }
