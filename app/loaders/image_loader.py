import os
from .base import BaseLoader, RawDocument


class ImageLoader(BaseLoader):
    modality = "image"

    def __init__(self, captioner=None):
        # Optional callable(image_path) -> str, e.g. a BLIP/LLaVA captioner.
        # A caption gives BM25 / sparse retrieval something to match against,
        # since sparse search can't work on raw pixels.
        self.captioner = captioner

    def load(self, path: str) -> list[RawDocument]:
        caption = self.captioner(path) if self.captioner else f"[image: {os.path.basename(path)}]"
        return [RawDocument(
            content=caption,
            modality=self.modality,
            source_path=path,
            asset_path=path,
        )]
