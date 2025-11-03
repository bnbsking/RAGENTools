from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, List
from langchain_core.documents import Document


@dataclass
class RetrievedChunk:
    scores: float
    content: str
    meta: Any


class BaseRetriever(ABC):
    @abstractmethod
    def query(self, query: str, **kwargs) -> List[RetrievedChunk]:
        pass

    def chunks_concat(self, chunks: List[RetrievedChunk]) -> str:
        chunks.sort(key=lambda x: x.scores)
        texts = []
        for i, chunk in enumerate(chunks):
            texts.append(f"Chunk {i+1} with score {chunk.scores}:\n{chunk.content}\n")
        return ("\n" + "="*10 + "\n").join(texts)
    