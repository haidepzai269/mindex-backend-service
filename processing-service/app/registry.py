from __future__ import annotations

from app.handlers.base import BaseHandler, TaskError

HANDLERS: dict[str, BaseHandler] = {}


def register(task_type: str, handler: BaseHandler) -> None:
    HANDLERS[task_type] = handler


def get_handler(task_type: str) -> BaseHandler:
    handler = HANDLERS.get(task_type)
    if handler is None:
        raise TaskError(f"Task type '{task_type}' is not registered")
    return handler


def list_registered() -> list[str]:
    return list(HANDLERS.keys())
