"""
CLIP puts text and images into the same vector space, which is what makes
hybrid multimodal retrieval possible with a single ANN index: a text query
embedding can be compared directly against embeddings of page-images, video
frames, and plain text chunks alike.

Non-visual modalities (json/csv/audio-transcript/text) are embedded with
CLIP's *text* tower. Image-bearing documents (image loader, pdf page
renders, video frames) are embedded with CLIP's *image* tower.
"""
import open_clip
import torch
from PIL import Image
from app.config import settings
from app.loaders.base import RawDocument


class ClipEmbedder:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model_name, pretrained=settings.clip_pretrained
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model_name)
        self.model.to(self.device).eval()

    @torch.no_grad()
    def embed_text(self, texts: list[str]) -> list[list[float]]:
        tokens = self.tokenizer(texts, context_length=77).to(self.device)
        feats = self.model.encode_text(tokens)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    @torch.no_grad()
    def embed_images(self, image_paths: list[str]) -> list[list[float]]:
        imgs = [self.preprocess(Image.open(p).convert("RGB")) for p in image_paths]
        batch = torch.stack(imgs).to(self.device)
        feats = self.model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy().tolist()

    def embed_documents(self, docs: list[RawDocument]) -> list[list[float]]:
        """Route each document to the correct CLIP tower and return
        embeddings in the same order as the input list."""
        embeddings: list[list[float] | None] = [None] * len(docs)

        image_idx = [i for i, d in enumerate(docs) if d.modality == "image" and d.asset_path]
        text_idx = [i for i in range(len(docs)) if i not in image_idx]

        if image_idx:
            paths = [docs[i].asset_path for i in image_idx]
            img_embeds = self.embed_images(paths)
            for i, emb in zip(image_idx, img_embeds):
                embeddings[i] = emb

        if text_idx:
            texts = [docs[i].content for i in text_idx]
            txt_embeds = self.embed_text(texts)
            for i, emb in zip(text_idx, txt_embeds):
                embeddings[i] = emb

        return embeddings  # type: ignore

    def embed_query(self, query: str) -> list[float]:
        return self.embed_text([query])[0]


_embedder: ClipEmbedder | None = None


def get_embedder() -> ClipEmbedder:
    global _embedder
    if _embedder is None:
        _embedder = ClipEmbedder()
    return _embedder
