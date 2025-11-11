from abc import ABC

from ragentools.rags.vectorstores_union import BaseVectorStoresUnion
from ragentools.rags.rerankers import BaseReranker


class BaseRAG(ABC):
    def __init__(
            self,
            vector_store_union: BaseVectorStoresUnion,
            reranker: BaseReranker
        ):
        self.vector_store_union = vector_store_union
        self.reranker = reranker

    def index(self, **kwargs) -> None:
        self.vector_store_union.index(**kwargs)
    
    def retrieve(self, query: str) -> str:
        retrieved_chunks = self.vector_store_union.retrieve(query)
        text = self.reranker(retrieved_chunks)
        return text
    