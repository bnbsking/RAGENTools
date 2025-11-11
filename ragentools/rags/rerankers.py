from abc import ABC
from dataclasses import dataclass
from typing import Any, List


@dataclass
class RetrievedChunk:
    scores: float
    content: str
    meta: Any


class BaseReranker(ABC):
    def rerank(self, chunks: List[RetrievedChunk], reverse=False) -> List[RetrievedChunk]:
        return sorted(chunks, key=lambda x: x.scores, reverse=reverse)

    def concat(self, chunks: List[RetrievedChunk]) -> str:
        texts = []
        for i, chunk in enumerate(chunks):
            texts.append(f"Chunk {i+1} with score {chunk.scores}:\n{chunk.content}\n")
        return ("\n" + "="*10 + "\n").join(texts)

    def __call__(self, chunks: List[RetrievedChunk], reverse=False) -> str:
        reranked_chunks = self.rerank(chunks, reverse=reverse)
        return self.concat(reranked_chunks)
    