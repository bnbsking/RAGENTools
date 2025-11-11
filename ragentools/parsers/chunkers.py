from typing import Iterator

from . import Document


class BaseChunker:
    def chunk(self, doc: Document) -> Iterator[Document]:
        raise NotImplementedError


class OverlapChunker(BaseChunker):
    """Split text into overlapping chunks."""
    def __init__(self, chunk_size: int = 800, overlap_size: int = 100):
        self.chunk_size = chunk_size
        self.overlap_size = overlap_size

    def chunk(self, doc: Document) -> Iterator[Document]:
        text = doc.page_content
        start = 0
        while start < len(text):
            end = start + self.chunk_size
            yield Document(page_content=text[start:end], metadata=doc.metadata)
            start += self.chunk_size - self.overlap_size
