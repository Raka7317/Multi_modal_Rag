from opensearchpy import OpenSearch, helpers
from app.config import settings
from app.loaders.base import RawDocument


def get_client() -> OpenSearch:
    auth = None
    if settings.opensearch_user:
        auth = (settings.opensearch_user, settings.opensearch_password)
    return OpenSearch(
        hosts=[{"host": settings.opensearch_host, "port": settings.opensearch_port}],
        http_auth=auth,
        use_ssl=settings.opensearch_use_ssl,
        verify_certs=settings.opensearch_use_ssl,
    )


def ensure_index(client: OpenSearch):
    if client.indices.exists(settings.opensearch_index):
        return
    body = {
        "settings": {
            "index": {"knn": True},
        },
        "mappings": {
            "properties": {
                "content": {"type": "text"},          # analyzed -> powers BM25 / sparse leg
                "modality": {"type": "keyword"},
                "source_path": {"type": "keyword"},
                "asset_path": {"type": "keyword"},
                "metadata": {"type": "object", "enabled": True},
                "embedding": {
                    "type": "knn_vector",
                    "dimension": settings.embedding_dim,
                    "method": {
                        "name": "hnsw",
                        "space_type": "cosinesimil",
                        "engine": "nmslib",
                        "parameters": {"ef_construction": 128, "m": 24},
                    },
                },
            }
        },
    }
    client.indices.create(settings.opensearch_index, body=body)


def index_documents(docs: list[RawDocument], embeddings: list[list[float]]):
    client = get_client()
    ensure_index(client)

    actions = []
    for doc, emb in zip(docs, embeddings):
        actions.append({
            "_index": settings.opensearch_index,
            "_source": {
                "content": doc.content,
                "modality": doc.modality,
                "source_path": doc.source_path,
                "asset_path": doc.asset_path,
                "metadata": doc.metadata,
                "embedding": emb,
            },
        })
    helpers.bulk(client, actions)
