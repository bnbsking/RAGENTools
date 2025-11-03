import json
import os
import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI
)
from ragentools.indexers.embedding import CustomEmbedding
from ragentools.retrievers.retrievers import TwoLevelRetriever


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indexing"]
    cfg_qa = cfg["gen_qa"]
    cfg_ans = cfg["answering"]

    # Init API
    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"])
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    embed_model = CustomEmbedding(api=api_emb, dim=3072)

    # Load two-level retriever
    retriever = TwoLevelRetriever(
        embed_model=embed_model,
        fine_index_folder=cfg_ind["indices_save_folder"],
        coarse_index_path=os.path.join(cfg_ind["indices_save_folder"], "coarse_grained_index.faiss")
    )

    # Query
    data_list = json.load(open(cfg_qa["save_path"], 'r', encoding='utf-8'))[:2]
    for i, data in enumerate(data_list):
        question = data["question"]
        retrieved_chunks = retriever.query(question)
        retrieved_text = retriever.chunks_concat(retrieved_chunks)
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
