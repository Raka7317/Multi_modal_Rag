import pandas as pd
from .base import BaseLoader, RawDocument


class CSVLoader(BaseLoader):
    modality = "csv"

    def __init__(self, rows_per_chunk: int = 20):
        # Group rows into small batches so each RawDocument stays semantically
        # coherent, rather than one document per single row (too fragmented)
        # or the whole file as one row (too coarse for retrieval).
        self.rows_per_chunk = rows_per_chunk

    def load(self, path: str) -> list[RawDocument]:
        df = pd.read_csv(path)
        docs = []
        for start in range(0, len(df), self.rows_per_chunk):
            batch = df.iloc[start:start + self.rows_per_chunk]
            content = batch.to_csv(index=False)
            docs.append(RawDocument(
                content=content,
                modality=self.modality,
                source_path=path,
                metadata={"row_start": start, "row_end": start + len(batch)},
            ))
        return docs
