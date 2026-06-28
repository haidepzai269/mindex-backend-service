from app import registry
from app.handlers.extract import ExtractHandler
from app.handlers.ocr import OCRHandler
from app.handlers.video import VideoHandler


def register_all() -> None:
    registry.register("extract", ExtractHandler())
    registry.register("ocr", OCRHandler())
    registry.register("video", VideoHandler())
