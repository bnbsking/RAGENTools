import json
import os
import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI
)
from ragentools.rags.utils.embedding import LangChainEmbedding
from ragentools.rags.rags import BaseRAG
from ragentools.rags.rag_engines import TwoLevelRAGEngine
from ragentools.rags.rerankers import BaseReranker
from langchain_community.vectorstores import FAISS


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
    cfg_api = cfg["api"]
    cfg_rag = cfg["rag"]
    cfg_gqa = cfg["gen_qa"]
    cfg_ans = cfg["ans"]

    # Init API
    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"])
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    embed_model = LangChainEmbedding(api=api_emb, dim=3072)

    # RAG
    rag_engine = TwoLevelRAGEngine(vector_store_cls=FAISS)
    rag_engine.load(cfg_rag["save_folder"], embed_model)
    rag = BaseRAG(
        rag_engine=rag_engine,
        reranker=BaseReranker()
    )
    
    # Answer
    data_list = json.load(open(cfg_gqa["save_path"], 'r', encoding='utf-8'))[:2]
    for i, data in enumerate(data_list):
        question = data["question"]
        retrieved_text = rag.retrieve(question)
        answer = api_chat.run(
            prompt=f"""Use the following RAG retrieved chunks to answer the question.
                Chunks: {retrieved_text}
                Question: {question}
            """,
            retry_sec=20,
        )
        data_list[i]["llm_response"] = answer
        data_list[i]["retrieved_text"] = retrieved_text
    os.makedirs(os.path.dirname(cfg_ans["save_path"]), exist_ok=True)
    json.dump(data_list, open(cfg_ans["save_path"], 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
