import os
import cv2
from .base import BaseLoader, RawDocument
from .audio_loader import AudioLoader


class VideoLoader(BaseLoader):
    modality = "video"

    def __init__(self, frame_interval_sec: float = 5.0, frame_dir: str = "/tmp/rag_assets",
                 whisper_model_size: str = "base"):
        self.frame_interval_sec = frame_interval_sec
        self.frame_dir = frame_dir
        os.makedirs(self.frame_dir, exist_ok=True)
        self.audio_loader = AudioLoader(whisper_model_size=whisper_model_size)

    def load(self, path: str) -> list[RawDocument]:
        docs: list[RawDocument] = []

        # 1) Sample frames uniformly and index them as images (CLIP image tower)
        cap = cv2.VideoCapture(path)
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frame_step = max(int(fps * self.frame_interval_sec), 1)
        frame_idx = 0
        saved = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if frame_idx % frame_step == 0:
                asset_path = os.path.join(self.frame_dir, f"{os.path.basename(path)}_f{saved}.png")
                cv2.imwrite(asset_path, frame)
                timestamp_sec = frame_idx / fps
                docs.append(RawDocument(
                    content=f"[frame at {timestamp_sec:.1f}s of {os.path.basename(path)}]",
                    modality="image",
                    source_path=path,
                    asset_path=asset_path,
                    metadata={"timestamp_sec": timestamp_sec, "derived_from": "video_frame"},
                ))
                saved += 1
            frame_idx += 1
        cap.release()

        # 2) Extract + transcribe the audio track (via ffmpeg through moviepy)
        try:
            from moviepy.editor import VideoFileClip
            audio_path = path + ".extracted.wav"
            VideoFileClip(path).audio.write_audiofile(audio_path, logger=None)
            audio_docs = self.audio_loader.load(audio_path)
            for d in audio_docs:
                d.modality = "video"
                d.source_path = path
                d.metadata["derived_from"] = "video_audio_track"
            docs.extend(audio_docs)
        except Exception as e:
            docs.append(RawDocument(
                content=f"[audio extraction failed: {e}]",
                modality="video",
                source_path=path,
            ))

        return docs
