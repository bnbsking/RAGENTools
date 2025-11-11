from typing import Iterator, List, Union

from . import Document
from .readers import BaseReader
from .chunkers import BaseChunker
from .savers import BaseSaver


class BaseParser:
    def __init__(self, reader: BaseReader, chunker: BaseChunker, saver: BaseSaver = None):
        self.reader = reader
        self.chunker = chunker
        self.saver = saver

    def _generate(self) -> Iterator[Document]:
        for doc in self.reader.read():
            for chunk in self.chunker.chunk(doc):
                yield chunk

    def run(self, lazy: bool = False) -> Union[List[Document], Iterator[Document]]:
        """Main entrypoint: supports both lazy and eager mode."""
        if self.saver:
            self.saver.save(self._generate())
        gen = self._generate()
        return gen if lazy else list(gen)
    