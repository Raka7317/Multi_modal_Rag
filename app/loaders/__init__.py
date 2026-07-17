import os
from .base import BaseLoader
from .text_loader import TextLoader
from .pdf_loader import PDFLoader
from .json_loader import JSONLoader
from .csv_loader import CSVLoader
from .image_loader import ImageLoader
from .audio_loader import AudioLoader
from .video_loader import VideoLoader

EXTENSION_MAP: dict[str, type[BaseLoader]] = {
    ".txt": TextLoader, ".md": TextLoader,
    ".pdf": PDFLoader,
    ".json": JSONLoader,
    ".csv": CSVLoader,
    ".png": ImageLoader, ".jpg": ImageLoader, ".jpeg": ImageLoader, ".webp": ImageLoader,
    ".mp3": AudioLoader, ".wav": AudioLoader, ".m4a": AudioLoader,
    ".mp4": VideoLoader, ".mov": VideoLoader, ".avi": VideoLoader,
}


def get_loader(path: str) -> BaseLoader:
    ext = os.path.splitext(path)[1].lower()
    loader_cls = EXTENSION_MAP.get(ext)
    if loader_cls is None:
        raise ValueError(f"No loader registered for extension '{ext}'")
    return loader_cls()


def load_any(path: str):
    return get_loader(path).load(path)
