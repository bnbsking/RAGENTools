from typing import List

from .base_parser import BaseParser, ChunkRecord
import pymupdf


class PDFParser(BaseParser):
    def pdf_page_text_generator(self, pdf_path: str):
        with pymupdf.open(pdf_path) as document:
            for page_num, page in enumerate(document, start=1):
                text = page.get_text()
                yield (page_num, text)

    def parse_one(self, file_path: str) -> List[ChunkRecord]:
        records = []
        for page_num, text in self.pdf_page_text_generator(file_path):
            if not text.strip():
                continue
            chunks = self.chunk_text(text)
            for chunk in chunks:
                record = ChunkRecord(
                    chunk=chunk,
                    source_path=file_path,
                    page=page_num
                )
                records.append(record)
        return records
