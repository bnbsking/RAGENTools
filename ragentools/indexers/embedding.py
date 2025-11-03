from langchain_core.embeddings import Embeddings


class CustomEmbedding(Embeddings):
    def __init__(self, api, dim: int):
        self.api = api
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return self.api.run_batches(texts, self.dim)

    def embed_query(self, text: str) -> list[float]:
        return self.api.run_batches([text], self.dim)[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        return await self.api.arun_batches(texts, self.dim)

    async def aembed_query(self, text: str) -> list[float]:
        return await self.api.arun_batches([text], self.dim)[0]
