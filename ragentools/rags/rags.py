from abc import ABC

from ragentools.rags.rag_engines import BaseRAGEngine
from ragentools.rags.rerankers import BaseReranker


class BaseRAG(ABC):
    def __init__(
            self,
            rag_engine: BaseRAGEngine,
            reranker: BaseReranker
        ):
        self.rag_engine = rag_engine
        self.reranker = reranker

    def index(self, **kwargs) -> None:
        self.rag_engine.index(**kwargs)
    
    def retrieve(self, query: str, **kwargs) -> str:
        retrieved_chunks = self.rag_engine.retrieve(query, **kwargs)
        text = self.reranker(retrieved_chunks, **kwargs)
        return text
    