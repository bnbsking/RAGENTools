from typing import Any, List

from llama_index.core import (
    ComposableGraph,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core.llms import LLM, CompletionResponse
from llama_index.core.settings import Settings
import os
from pydantic import PrivateAttr
import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI,
)


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


class GeminiChatLLM(LLM):
    _api: GoogleGeminiChatAPI = PrivateAttr()

    def __init__(self, api, **kwargs: Any):
        super().__init__(**kwargs)
        self._api = api

    def complete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Sync completion interface"""
        result = self._api.run(prompt=prompt, **kwargs)
        return CompletionResponse(text=result.text)

    async def acomplete(self, prompt: str, **kwargs: Any) -> CompletionResponse:
        """Async completion (can call same API for simplicity)"""
        return await self._api.arun(prompt=prompt, **kwargs)

    # The following methods are needed for abstract base compatibility
    def stream_complete(self, *args, **kwargs):
        raise NotImplementedError("Streaming not supported for GeminiChatLLM.")

    async def astream_complete(self, *args, **kwargs):
        raise NotImplementedError("Async streaming not supported.")

    def chat(self, *args, **kwargs):
        raise NotImplementedError("Chat interface not used in this example.")

    async def achat(self, *args, **kwargs):
        raise NotImplementedError("Async chat interface not used.")

    def stream_chat(self, *args, **kwargs):
        raise NotImplementedError("Streaming chat not supported.")

    async def astream_chat(self, *args, **kwargs):
        raise NotImplementedError("Async streaming chat not supported.")

    @property
    def metadata(self) -> dict:
        return {}


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/medical/v1/rag_medical_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indices"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["model_name"])
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["model_name"])
    embed_model = CustomEmbedding(api=api_emb, dim=3072)
    gemini_chat_llm = GeminiChatLLM(api=api_chat)

    documents = SimpleDirectoryReader(cfg_ind["data_folder"]).load_data()
    print(type(documents))  # List[Document]  # LlamaIndex Document objects  
    print(len(documents))  # for pdf, len=pages  # .txt, .pdf, .docx, .md, .jpg, .mp4, ... files are supported
    documents = documents[:3]  # limit to first 3 docs for testing

    # low-level indexing
    doc_indexes = []
    for i, doc in enumerate(documents):
        index = VectorStoreIndex.from_documents([doc], embed_model=embed_model)
        doc_indexes.append(index)  # LlamaIndex Index objects
        index.storage_context.persist(f"{cfg_ind['save_folder']}/doc_{i}")

    Settings.llm = gemini_chat_llm
    Settings.embed_model = embed_model
    
    graph_persist_dir = os.path.join(cfg_ind['save_folder'], "graph_root")
    os.makedirs(graph_persist_dir, exist_ok=True)

    # high-level indexing
    graph_storage_context = StorageContext.from_defaults()
    graph = ComposableGraph.from_indices(
        VectorStoreIndex,
        doc_indexes,
        index_summaries=[f"Summary of document {i}" for i in range(len(documents))],
        llm=gemini_chat_llm,
        storage_context=graph_storage_context
    )
    graph.root_index.storage_context.persist(persist_dir=graph_persist_dir)
    print(f"Graph Root Index ID: {graph.root_id}")  # 19d31d03-b58d-4ae6-a25a-f6edda274c5e
