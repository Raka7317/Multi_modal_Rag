import json
from .base import BaseLoader, RawDocument


class JSONLoader(BaseLoader):
    modality = "json"

    def __init__(self, jsonpath_records: str | None = None):
        """
        jsonpath_records: optional dotted path to a list within the JSON to
        iterate as separate records (e.g. "data.items"). If None, the whole
        file is flattened into one document, or each top-level list item
        becomes its own document if the root is a list.
        """
        self.jsonpath_records = jsonpath_records

    def load(self, path: str) -> list[RawDocument]:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if self.jsonpath_records:
            for key in self.jsonpath_records.split("."):
                data = data[key]

        records = data if isinstance(data, list) else [data]
        docs = []
        for i, record in enumerate(records):
            docs.append(RawDocument(
                content=json.dumps(record, ensure_ascii=False, indent=2),
                modality=self.modality,
                source_path=path,
                metadata={"record_index": i},
            ))
        return docs
