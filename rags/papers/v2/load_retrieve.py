import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI,
)

from langchain_community.vectorstores import FAISS
from ragentools.rags.utils.embedding import LangChainEmbedding
from ragentools.rags.rags import BaseRAG
from ragentools.rags.rag_engines import TwoLevelRAGEngine
from ragentools.rags.rerankers import LLMReranker


# inputs
cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
cfg_api = cfg["api"]
cfg_par = cfg["parser"]
cfg_rag = cfg["rag"]


# init clients
api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"], retry_sec=65)
api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name="gemini-2.0-flash", retry_sec=65)
embed_model = LangChainEmbedding(api=api_emb, dim=3072)


# rag
rag_engine = TwoLevelRAGEngine(
        vector_store_cls=FAISS,
        embed_model = embed_model,
        api_chat = api_chat
)
rag_engine.load(load_folder = cfg_rag["save_folder"])
rag = BaseRAG(
    rag_engine=rag_engine,
    reranker=LLMReranker(
        api=api_chat,
        prompt_path="/app/ragentools/prompts/reranker.yaml")
)

print(rag.retrieve("What are the key areas that medicine focuses on to ensure well-being?"))
