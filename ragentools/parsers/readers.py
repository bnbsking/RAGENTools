from abc import ABC, abstractmethod
from typing import List, Dict, Iterator

import pymupdf

from . import Document


class BaseReader(ABC):
    @abstractmethod
    def read(self) -> Iterator[Document]:
        pass


class ListReader(BaseReader):
    # 1 ele in list = 1 chunk
    def __init__(self, text_list: List[str], meta_list: List[Dict]):
        self.text_list = text_list
        self.meta_list = meta_list

    def read(self) -> Iterator[Document]:
        for text, meta in zip(self.text_list, self.meta_list):
            yield Document(page_content=text, metadata=meta)


class PDFReader(BaseReader):
    # 1 page = 1 chunk
    def __init__(self, pdf_paths: List[str]):
        self.pdf_paths = pdf_paths

    def read(self) -> Iterator[Document]:
        for pdf_path in self.pdf_paths:
            with pymupdf.open(pdf_path) as document:
                for page_num, page in enumerate(document, start=1):
                    text = page.get_text()
                    yield Document(
                        page_content=text,
                        metadata={"source_path": pdf_path, "page": page_num}
                    )
