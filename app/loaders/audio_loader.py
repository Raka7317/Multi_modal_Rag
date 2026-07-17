from .base import BaseLoader, RawDocument


class AudioLoader(BaseLoader):
    modality = "audio"

    def __init__(self, whisper_model_size: str = "base"):
        self.whisper_model_size = whisper_model_size
        self._model = None

    def _get_model(self):
        if self._model is None:
            import whisper
            self._model = whisper.load_model(self.whisper_model_size)
        return self._model

    def load(self, path: str) -> list[RawDocument]:
        model = self._get_model()
        result = model.transcribe(path)

        docs = []
        # Whole-file transcript
        docs.append(RawDocument(
            content=result["text"],
            modality=self.modality,
            source_path=path,
            asset_path=path,
        ))
        # Per-segment documents preserve timestamps for citation / seek-to-time UX
        for seg in result.get("segments", []):
            docs.append(RawDocument(
                content=seg["text"],
                modality=self.modality,
                source_path=path,
                metadata={"start": seg["start"], "end": seg["end"]},
            ))
        return docs
