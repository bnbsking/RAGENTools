import glob
import os
from typing import Any, List

from llama_index.core import (
    ComposableGraph,
    SimpleDirectoryReader,
    StorageContext,
    VectorStoreIndex
)
from llama_index.core.settings import Settings
from llama_index.core.embeddings import BaseEmbedding
from llama_index.core import load_indices_from_storage, load_graph_from_storage, load_index_from_storage
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.core.llms import LLM, CompletionResponse
from llama_index.core.llms import LLMMetadata
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
    def metadata(self) -> LLMMetadata:
        """Get LLM metadata."""
        # FIX: Return an instance of the LlamaIndex-specific LLMMetadata class
        return LLMMetadata(
            context_window=8192,
            # You can set other required metadata here
            num_output=2048, # A reasonable default for output tokens
            model_name=self._api.model_name, # Access the model name from your API wrapper
        )


cfg = yaml.safe_load(open("/app/rags/medical/v1/rag_medical_v1.yaml"))
cfg_api = cfg["api"]
cfg_ind = cfg["indices"]

api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["model_name"])
api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["model_name"])
embed_model = CustomEmbedding(api=api_emb, dim=3072)
gemini_chat_llm = GeminiChatLLM(api=api_chat)

root_id = "19d31d03-b58d-4ae6-a25a-f6edda274c5e"

Settings.llm = gemini_chat_llm
Settings.embed_model = embed_model

doc_indexes = []
for i in range(3):  # assuming 3 docs
    doc_dir = f"{cfg_ind['save_folder']}/doc_{i}"
    storage_context = StorageContext.from_defaults(persist_dir=doc_dir)
    index = load_index_from_storage(storage_context)
    doc_indexes.append(index)


# Load sub-index storage contexts
sub_storage_contexts = {}
for i in range(3):  # adjust number of docs
    sub_dir = f"{cfg_ind['save_folder']}/doc_{i}"
    sub_storage_context = StorageContext.from_defaults(persist_dir=sub_dir)
    sub_storage_contexts[f"doc_{i}"] = sub_storage_context



# 1. Define the directories where your indices are saved
#child_index_dir = cfg_ind['save_folder']
graph_persist_dir = os.path.join(cfg_ind['save_folder'], "graph_root")

# 2. Load the StorageContext for the Graph's Root Index
# Note: This is where we use from_defaults() to *load* the existing files.
index_store_dict = {}
for ctx in sub_storage_contexts.values():
    for index_id in ctx.index_store.to_dict()["index_store/data"]:
        index_store_dict[index_id] = ctx.index_store
graph_storage_context = StorageContext.from_defaults(
    persist_dir=graph_persist_dir,
    index_store=index_store_dict
)

# 3. FIX: Load the Child Index Objects and Map them by their Index ID
# We need to load the actual index objects, which contain the correct Index ID.
# loaded_child_indices = {}
# # Use glob to find all child index directories (doc_0, doc_1, etc.)
# for doc_storage_path in glob.glob(f"{child_index_dir}/doc_*"):
#     # Load the storage context for the individual child index
#     child_storage_context = StorageContext.from_defaults(
#         persist_dir=doc_storage_path
#     )
#     # Load the VectorStoreIndex object from its storage context
#     child_index = load_indices_from_storage(
#         child_storage_context, 
#         vector_store_index_cls=VectorStoreIndex # Specify the index type
#     )[0] # load_indices_from_storage returns a list, so we take the first item
    
#     # Map the index object using its unique ID
#     loaded_child_indices[child_index.index_id] = child_index
#     print(f"  -> Loaded Index ID: {child_index.index_id}")

# 4. Load the ComposableGraph
# We pass the root index storage context and the mapping of all child storage contexts
# loaded_graph = load_graph_from_storage(
#     graph_storage_context,
#     root_id=root_id,
#     llm=gemini_chat_llm,
#     embed_model=embed_model,
#     #all_indices=loaded_child_indices
# )
loaded_graph = load_graph_from_storage(graph_storage_context, root_id=root_id)

# 5. Create the Query Engine
# The query engine for a ComposableGraph is essentially a RouterQueryEngine, 
# which handles the two-level process automatically:
#   Level 1: Query the root index summaries (The router determines which child index to use).
#   Level 2: Query the selected child index (VectorStoreIndex retrieval).
query_engine = loaded_graph.as_query_engine(
    # Optional: Configure the child index query engine settings
    # For example, to tell the root index to select the top 2 child indexes:
    # router_kwargs={"select_n": 2}
    llm = gemini_chat_llm,
)

# 6. Run a Test Query
query = "What are the main advantages of the AutoDeco heads in transformer models?"
print(f"\n--- QUERY: {query} ---")
response = query_engine.query(query)
print(f"RESPONSE:\n{response}")

# # You can also inspect the source nodes to see which documents were retrieved
# print("\n--- Source Nodes Retrieved ---")
# for node in response.source_nodes:
#     print(f"Source: {node.metadata.get('filename', 'Unknown File')}, Score: {node.score:.2f}")
