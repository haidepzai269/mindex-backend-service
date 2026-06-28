from __future__ import annotations

import asyncio
import time

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app import registry
from app.handlers.base import TaskError

router = APIRouter()

SYNC_TIMEOUT_SECONDS = 120


class ProcessRequest(BaseModel):
    task_type: str
    payload: dict


@router.post("/api/v1/process")
async def process_sync(req: ProcessRequest):
    try:
        handler = registry.get_handler(req.task_type)
    except TaskError as exc:
        raise HTTPException(status_code=400, detail={
            "success": False,
            "error": "UNKNOWN_TASK_TYPE",
            "message": str(exc),
        })

    start = time.monotonic()
    try:
        result = await asyncio.wait_for(
            handler.process(req.payload),
            timeout=SYNC_TIMEOUT_SECONDS,
        )
        duration_ms = int((time.monotonic() - start) * 1000)
        return {
            "success": True,
            "task_type": req.task_type,
            "result": result,
            "duration_ms": duration_ms,
        }
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail={
            "success": False,
            "error": "PROCESSING_TIMEOUT",
            "message": f"Task exceeded {SYNC_TIMEOUT_SECONDS}s timeout",
        })
    except Exception as exc:
        raise HTTPException(status_code=422, detail={
            "success": False,
            "error": "PROCESSING_FAILED",
            "message": str(exc),
        })
