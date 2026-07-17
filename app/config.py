"""
Central configuration. All values overridable via environment variables / .env
"""
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- CLIP embedding model (used for text, image, and video-frame embeddings) ---
    clip_model_name: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"
    embedding_dim: int = 512

    # --- Chunking ---
    chunk_size: int = 800
    chunk_overlap: int = 120

    # --- OpenSearch (vector + BM25 hybrid store) ---
    opensearch_host: str = "localhost"
    opensearch_port: int = 9200
    opensearch_index: str = "multimodal_rag"
    opensearch_use_ssl: bool = False
    opensearch_user: str | None = None
    opensearch_password: str | None = None

    # --- Retrieval ---
    top_k_dense: int = 20
    top_k_sparse: int = 20
    hybrid_alpha: float = 0.5   # weight between dense (1.0) and sparse (0.0) scores
    rerank_top_n: int = 5
    cross_encoder_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"

    # --- Memory ---
    mongodb_uri: str = "mongodb://localhost:27017"
    mongodb_db: str = "rag_memory"
    long_term_collection: str = "user_facts"

    # --- S3 (raw file + derived asset storage) ---
    s3_bucket: str = "my-multimodal-rag-bucket"
    aws_region: str = "us-east-1"

    # --- LLM (answer generation, used inside the LangGraph pipeline) ---
    llm_provider: str = "anthropic"
    anthropic_model: str = "claude-sonnet-5"

    class Config:
        env_file = ".env"


settings = Settings()
