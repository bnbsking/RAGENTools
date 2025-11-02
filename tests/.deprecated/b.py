from llama_index.core import SimpleDirectoryReader, VectorStoreIndex
from llama_index.core import ComposableGraph
from llama_index.core.embeddings import BaseEmbedding
from ragentools.api_calls.google_gemini import GoogleGeminiEmbeddingAPI
import yaml
from typing import List
from pydantic import PrivateAttr


documents = SimpleDirectoryReader("/app/rags/medical/data").load_data()
print(type(documents))  # List[Document]  # LlamaIndex Document objects  # for pdf, len=pages
print(len(documents))
# .txt, .pdf, .docx, .md, .jpg, .mp4, ... files are supported


class CustomEmbedding(BaseEmbedding):
    _api: GoogleGeminiEmbeddingAPI = PrivateAttr()
    _dim: int = PrivateAttr(default=3072)

    def __init__(self, api, dim: int, **kwargs):
        super().__init__(**kwargs)
        self._api = api
        self._dim = dim

    def _get_text_embedding(self, text: str) -> List[float]:
        result = self._api.run(prompt=[text], dim=self._dim)
        return result[0]
    
    async def _aget_text_embedding(self, text: str) -> List[float]:
        result = await self._api.arun(prompt=[text], dim=self._dim)
        return result[0]

    def _get_query_embedding(self, query: str) -> List[float]:
        return self._get_text_embedding(query)

    async def _aget_query_embedding(self, query: str) -> List[float]:
        return await self._aget_text_embedding(query)


api_key = yaml.safe_load(open("/app/tests/api_keys.yaml"))["GOOGLE_API_KEY"]
api = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name="gemini-embedding-001")
embed_model = CustomEmbedding(api=api, dim=3072)


doc_indexes = []
for doc in documents:
    index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)
    doc_indexes.append(index)
    index.storage_context.persist(f"/app/rags/medical/index_storage/doc_{i}")


graph = ComposableGraph.from_indices(
    VectorStoreIndex,
    doc_indexes,
    index_summaries=[f"Summary of document {i}" for i in range(len(documents))],
)
graph.storage_context.persist("/app/rags/medical/index_storage/graph")
