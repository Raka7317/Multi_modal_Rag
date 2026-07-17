import os
from pypdf import PdfReader
from pdf2image import convert_from_path
from .base import BaseLoader, RawDocument


class PDFLoader(BaseLoader):
    modality = "pdf"

    def __init__(self, extract_page_images: bool = True, image_dir: str = "/tmp/rag_assets"):
        self.extract_page_images = extract_page_images
        self.image_dir = image_dir
        os.makedirs(self.image_dir, exist_ok=True)

    def load(self, path: str) -> list[RawDocument]:
        docs: list[RawDocument] = []
        reader = PdfReader(path)

        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                docs.append(RawDocument(
                    content=text,
                    modality=self.modality,
                    source_path=path,
                    metadata={"page": i},
                ))

        # Also rasterize each page as an image so CLIP can index diagrams/
        # charts/scanned content that text extraction misses.
        if self.extract_page_images:
            images = convert_from_path(path, dpi=150)
            for i, img in enumerate(images):
                asset_path = os.path.join(self.image_dir, f"{os.path.basename(path)}_page{i}.png")
                img.save(asset_path)
                docs.append(RawDocument(
                    content=f"[page image {i} of {os.path.basename(path)}]",
                    modality="image",
                    source_path=path,
                    asset_path=asset_path,
                    metadata={"page": i, "derived_from": "pdf_page_render"},
                ))
        return docs
