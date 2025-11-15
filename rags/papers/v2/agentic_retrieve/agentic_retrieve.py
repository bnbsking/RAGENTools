import yaml

from langchain_community.vectorstores import FAISS

from ragentools.api_calls.google_gemini import GoogleGeminiChatAPI, GoogleGeminiEmbeddingAPI
from ragentools.rags.rag_engines import TwoLevelRAGEngine
from ragentools.rags.rerankers import LLMReranker
from ragentools.rags.agentic_rag import IterativeRAG
from ragentools.rags.utils.embedding import LangChainEmbedding


# inputs
cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
cfg_api = cfg["api"]
cfg_rag = cfg["rag"]

# init clients
api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name="gemini-2.0-flash", retry_sec=65)
api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"], retry_sec=65)
embed_model = LangChainEmbedding(api=api_emb, dim=3072)

rag_engine = TwoLevelRAGEngine(
    vector_store_cls = FAISS,
    embed_model = embed_model,
    api_chat = api_chat
)
rag_engine.load(load_folder=cfg_rag["save_folder"])
reranker = LLMReranker(api_chat, prompt_path="/app/ragentools/prompts/reranker.yaml")

rag = IterativeRAG(
    api=api_chat,
    rag_engine=rag_engine,
    reranker=reranker,
    need_retrieval_prompt_path="/app/ragentools/prompts/agentic_rag/need_retrieval.yaml",
    query_decomposer_prompt_path="/app/ragentools/prompts/agentic_rag/query_decomposer.yaml",
    is_sufficient_prompt_path="/app/ragentools/prompts/agentic_rag/is_sufficient.yaml",
    query_fixer_prompt_path="/app/ragentools/prompts/agentic_rag/query_fixer.yaml",
    #draw="/app/rags/papers/v2/agentic_retrieve/graph.png"
)

result = rag.retrieve("What is next day of Friday?")
print(result)

result = rag.retrieve("What are the key areas that medicine focuses on to ensure well-being? And What is the primary goal of healthcare practices?")
print(result)


