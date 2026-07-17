"""
The hybrid search leg gives us a broad, fast shortlist. A cross-encoder
scores the (query, passage) pair jointly (rather than via separate
embeddings), which is far more accurate but too slow to run over the whole
corpus — so it's only applied to the top-N shortlist to pick the final
context passed to the LLM.
"""
from sentence_transformers import CrossEncoder
from app.config import settings

_cross_encoder: CrossEncoder | None = None


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(settings.cross_encoder_model)
    return _cross_encoder


def rerank(query: str, candidates: list[dict], top_n: int | None = None) -> list[dict]:
    top_n = top_n or settings.rerank_top_n
    if not candidates:
        return []

    model = get_cross_encoder()
    pairs = [(query, c["content"]) for c in candidates]
    scores = model.predict(pairs)

    for c, s in zip(candidates, scores):
        c["rerank_score"] = float(s)

    return sorted(candidates, key=lambda c: c["rerank_score"], reverse=True)[:top_n]
