from .base import BaseLoader, RawDocument


class TextLoader(BaseLoader):
    modality = "text"

    def load(self, path: str) -> list[RawDocument]:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            content = f.read()
        return [RawDocument(content=content, modality=self.modality, source_path=path)]
