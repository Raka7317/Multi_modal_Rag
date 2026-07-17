"""
Recursive chunking: try to split on the largest semantic boundary first
(paragraphs), fall back to smaller ones (sentences, words, characters) only
if a piece is still too big. Image/audio-frame RawDocuments with no
meaningful text body are passed through untouched (chunking pixels makes no
sense — they're embedded as single units by CLIP's image tower).
"""
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.loaders.base import RawDocument
from app.config import settings


def chunk_documents(docs: list[RawDocument]) -> list[RawDocument]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunked: list[RawDocument] = []
    for doc in docs:
        # Don't re-chunk assets that are fundamentally single visual/audio units
        if doc.modality == "image" or (doc.asset_path and not doc.content.strip()):
            chunked.append(doc)
            continue

        pieces = splitter.split_text(doc.content)
        for i, piece in enumerate(pieces):
            chunked.append(RawDocument(
                content=piece,
                modality=doc.modality,
                source_path=doc.source_path,
                asset_path=doc.asset_path,
                metadata={**doc.metadata, "chunk_index": i, "chunk_total": len(pieces)},
            ))
    return chunked
