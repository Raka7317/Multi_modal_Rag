import boto3
from app.config import settings
from app.loaders import load_any
from app.chunking.recursive_chunker import chunk_documents
from app.embeddings.clip_embedder import get_embedder
from app.retrieval.vector_store import index_documents


def upload_to_s3(local_path: str, key: str) -> str:
    s3 = boto3.client("s3", region_name=settings.aws_region)
    s3.upload_file(local_path, settings.s3_bucket, key)
    return f"s3://{settings.s3_bucket}/{key}"


def ingest_file(local_path: str, s3_key: str | None = None) -> int:
    """Full ingestion pipeline for one file. Returns number of chunks indexed."""
    if s3_key:
        upload_to_s3(local_path, s3_key)

    raw_docs = load_any(local_path)
    chunks = chunk_documents(raw_docs)

    embedder = get_embedder()
    embeddings = embedder.embed_documents(chunks)

    index_documents(chunks, embeddings)
    return len(chunks)
