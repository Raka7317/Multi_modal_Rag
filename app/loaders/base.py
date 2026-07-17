"""
Every loader converts a raw file into a list of `RawDocument` objects.
Keeping a single normalized output type is what makes the rest of the
pipeline (chunking -> embedding -> indexing) modality-agnostic.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Literal, Any
from abc import ABC, abstractmethod

Modality = Literal["text", "pdf", "json", "csv", "image", "audio", "video"]


@dataclass
class RawDocument:
    content: str                     # text representation (transcript, caption, extracted text, stringified row, etc.)
    modality: Modality
    source_path: str
    metadata: dict[str, Any] = field(default_factory=dict)
    # Optional: local path to a raw asset (frame/crop/audio-clip) that should
    # be embedded with CLIP's image tower instead of / in addition to text.
    asset_path: str | None = None


class BaseLoader(ABC):
    modality: Modality

    @abstractmethod
    def load(self, path: str) -> list[RawDocument]:
        ...
