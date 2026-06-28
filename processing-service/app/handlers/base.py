from abc import ABC, abstractmethod


class TaskError(Exception):
    pass


class BaseHandler(ABC):
    @abstractmethod
    async def process(self, payload: dict) -> dict:
        ...
