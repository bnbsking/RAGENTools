from typing import List, Tuple

from langchain_core.documents import Document
from ragentools.rags.embedding import CustomEmbedding
from ragentools.parsers import Document as Easy_Document


class BaseVectorStoreAdapter:
    def __init__(self, index):
        # index is cls as "langchain_community.vectorstores.FAISS"
        self.index = index

    @classmethod
    def from_documents(
            cls,
            documents: List[Easy_Document],
            embedding: CustomEmbedding
        ):
        # return is obj as from "langchain_community.vectorstores.FAISS"
        raise NotImplementedError

    @classmethod
    def load_local(cls, path: str) -> None:
        raise NotImplementedError

    def save_local(self, path: str) -> None:
        raise NotImplementedError

    def similarity_search_with_score(self, query: str, **kwargs) \
        -> List[Tuple[Document, float]]:
        raise NotImplementedError
