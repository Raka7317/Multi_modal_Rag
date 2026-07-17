"""
Hybrid retrieval = run a dense kNN search and a sparse BM25 search against
the same OpenSearch index, then fuse the two ranked lists by a weighted,
min-max normalized similarity score (alpha controls the dense/sparse mix).
This catches both semantic matches (dense) and exact keyword/entity matches
(sparse) that CLIP embeddings alone can be weak on (e.g. IDs, numbers,
proper nouns).
"""
from app.config import settings
from app.embeddings.clip_embedder import get_embedder
from app.retrieval.vector_store import get_client


def _normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    lo, hi = min(scores.values()), max(scores.values())
    if hi == lo:
        return {k: 1.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def hybrid_search(query: str, k: int = 10) -> list[dict]:
    client = get_client()
    embedder = get_embedder()
    query_vec = embedder.embed_query(query)

    dense_resp = client.search(index=settings.opensearch_index, body={
        "size": settings.top_k_dense,
        "query": {"knn": {"embedding": {"vector": query_vec, "k": settings.top_k_dense}}},
    })
    sparse_resp = client.search(index=settings.opensearch_index, body={
        "size": settings.top_k_sparse,
        "query": {"match": {"content": query}},
    })

    dense_scores = {h["_id"]: h["_score"] for h in dense_resp["hits"]["hits"]}
    sparse_scores = {h["_id"]: h["_score"] for h in sparse_resp["hits"]["hits"]}
    sources = {h["_id"]: h["_source"] for h in dense_resp["hits"]["hits"]}
    sources.update({h["_id"]: h["_source"] for h in sparse_resp["hits"]["hits"]})

    dense_n = _normalize(dense_scores)
    sparse_n = _normalize(sparse_scores)

    alpha = settings.hybrid_alpha
    fused: dict[str, float] = {}
    for doc_id in set(dense_n) | set(sparse_n):
        fused[doc_id] = alpha * dense_n.get(doc_id, 0.0) + (1 - alpha) * sparse_n.get(doc_id, 0.0)

    ranked_ids = sorted(fused, key=fused.get, reverse=True)[:k]
    return [{"id": doc_id, "score": fused[doc_id], **sources[doc_id]} for doc_id in ranked_ids]
