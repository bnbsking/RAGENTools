import glob
import os
from typing import List

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
import pandas as pd
import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI,
)
from ragentools.parsers.pdf_parser import PDFParser


def parse(cfg_ind: dict):
    parser = PDFParser(
        input_path_list=glob.glob(cfg_ind["data_folder"] + "*.pdf"),
        output_folder=os.path.join(cfg_ind["parsed_save_folder"])
    )
    parser.parse()


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


def recursive_summarization(api, texts: List[str], limit_len: int = 25000):
    # 8k * 4 = 32k limit
    if sum(len(t) for t in texts) <= limit_len:
        all_text = "\n".join(texts)
        return api.run(f"Summarize this text in 5 sentences:\n{all_text}")
    else:
        new_texts = []
        acc_chunks = []
        acc_len = 0
        for i, text in enumerate(texts):
            if i == len(texts) - 1 or acc_len + len(text) >= limit_len:
                acc_text = "\n".join(acc_chunks)
                acc_summary = api.run(f"Summarize this text in 3 sentences:\n{acc_text}")
                new_texts.append(acc_summary)
                acc_chunks = [text]
                acc_len = len(text)
            else:
                acc_chunks.append(text)
                acc_len += len(text)
        return recursive_summarization(api, new_texts, limit_len)


if __name__ == "__main__":
    cfg = yaml.safe_load(open("/app/rags/papers/v1/rags_papers_v1.yaml"))
    cfg_api = cfg["api"]
    cfg_ind = cfg["indexing"]

    api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
    api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"])
    api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"])
    embed_model = CustomEmbedding(api=api_emb, dim=3072)

    parse(cfg_ind)

    # Create fine-grained indices
    for csv_path in glob.glob(os.path.join(cfg_ind["parsed_save_folder"], "*.csv")):
        df = pd.read_csv(csv_path)
        docs = [
            Document(
                page_content=row['chunk'],
                metadata={"source_path": row['source_path'], "page": row['page']})
            for _, row in df.iterrows()
        ]
        faiss_index = FAISS.from_documents(docs, embedding=embed_model)
        faiss_index.save_local(os.path.join(cfg_ind["indices_save_folder"], os.path.basename(csv_path) + ".faiss"))
    
    # Create coarse-grained index
    file_summary_list = []
    for csv_path in glob.glob(os.path.join(cfg_ind["parsed_save_folder"], "*.csv")):
        df = pd.read_csv(csv_path)
        chunk_summaries = df["chunk"].tolist()
        file_summary = recursive_summarization(api_chat, chunk_summaries)
        file_summary_list.append(
            Document(
                page_content=file_summary,
                metadata={"source_path": os.path.basename(csv_path)}
            )
        )
    coarse_faiss_index = FAISS.from_documents(file_summary_list, embedding=embed_model)
    coarse_faiss_index.save_local(os.path.join(cfg_ind["indices_save_folder"], "coarse_grained_index.faiss"))
