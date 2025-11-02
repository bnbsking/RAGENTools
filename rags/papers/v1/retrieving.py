import glob
import os
from typing import List
import yaml

import pandas as pd
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from ragentools.api_calls.google_gemini import GoogleGeminiEmbeddingAPI, GoogleGeminiChatAPI

# ---------------------------
# Custom Embedding Wrapper
# ---------------------------
class CustomEmbedding(Embeddings):
    def __init__(self, api: GoogleGeminiEmbeddingAPI, dim: int):
        self.api = api
        self.dim = dim

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            out.extend(self.api.run(batch, self.dim))
        return out

    def embed_query(self, text: str) -> list[float]:
        return self.api.run([text], self.dim)[0]

    async def aembed_documents(self, texts: list[str]) -> list[list[float]]:
        out = []
        for i in range(0, len(texts), 100):
            batch = texts[i:i+100]
            out.extend(await self.api.arun(batch, self.dim))
        return out

    async def aembed_query(self, text: str) -> list[float]:
        return await self.api.arun([text], self.dim)[0]

# ---------------------------
# Two-level Retriever
# ---------------------------
class TwoLevelRetriever:
    def __init__(self, embed_model: Embeddings, fine_index_folder: str, coarse_index_path: str):
        self.embed_model = embed_model

        # Load coarse-level index
        self.coarse_index = FAISS.load_local(
            coarse_index_path,
            embeddings=self.embed_model,
            allow_dangerous_deserialization=True
        )

        # Load fine-level indices (one per document)
        self.fine_indices = {}
        for faiss_file in glob.glob(os.path.join(fine_index_folder, "*.faiss")):
            doc_name = os.path.basename(faiss_file).replace(".faiss", "")
            self.fine_indices[doc_name] = FAISS.load_local(
                faiss_file,
                embeddings=self.embed_model,
                allow_dangerous_deserialization=True
            )

    def query(self, query_text: str, top_k_coarse: int = 3, top_k_fine: int = 5) -> List[Document]:
        """
        Query two-level FAISS:
        1. Retrieve top-k documents from coarse index
        2. Retrieve top-k chunks from fine indices of those documents
        """
        # 1. Coarse retrieval
        coarse_docs = self.coarse_index.similarity_search(query_text, k=top_k_coarse)
        retrieved_docs = []

        for doc in coarse_docs:
            source = doc.metadata["source_path"]
            if source not in self.fine_indices:
                continue
            fine_index = self.fine_indices[source]

            # 2. Fine retrieval
            fine_chunks = fine_index.similarity_search(query_text, k=top_k_fine)
            retrieved_docs.extend(fine_chunks)

        return retrieved_docs

    def query_with_llm(self, query_text: str, llm_api: GoogleGeminiChatAPI, top_k_coarse: int = 3, top_k_fine: int = 5) -> str:
        """
        Retrieve top chunks and answer using LLM
        """
        retrieved_docs = self.query(query_text, top_k_coarse, top_k_fine)
        context_text = "\n\n".join([d.page_content for d in retrieved_docs])

        prompt = f"Use the following context to answer the question:\n{context_text}\n\nQuestion: {query_text}\nAnswer concisely:"
        return llm_api.run(prompt)

# ---------------------------
# Example usage
# ---------------------------
if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indexing"]

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
    question = "Why did Kael and his companions enter Seraphel despite the warnings from the local tribes?"
    top_chunks = retriever.query(question)
    print("Retrieved Chunks:")
    for c in top_chunks:
        print("-", c.metadata, c.page_content[:100])

    # Query with LLM
    answer = retriever.query_with_llm(question, api_chat)
    print("\nLLM Answer:")
    print(answer)
