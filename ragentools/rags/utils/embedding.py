from typing import List

from langchain_core.embeddings import Embeddings


class LangChainEmbedding(Embeddings):
    """For embedding model of langchain_community.vectorstores"""
    def __init__(self, api, dim: int = 3072):
        self.api = api
        self.dim = dim

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.api.run_batches(texts, self.dim)

    def embed_query(self, text: str) -> List[float]:
        return self.api.run_batches([text], self.dim)[0]

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        return await self.api.arun_batches(texts, self.dim)

    async def aembed_query(self, text: str) -> List[float]:
        return await self.api.arun_batches([text], self.dim)[0]
    