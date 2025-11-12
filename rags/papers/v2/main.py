import glob
from typing import Iterator

import yaml

from ragentools.api_calls.google_gemini import (
    GoogleGeminiEmbeddingAPI,
    GoogleGeminiChatAPI,
)

from ragentools.parsers import Document
from ragentools.parsers.readers import PDFReader
from ragentools.parsers.chunkers import OverlapChunker
from ragentools.parsers.savers import PDFSaver
from ragentools.parsers.parsers import BaseParser

from langchain_community.vectorstores import FAISS
from ragentools.rags.utils.embedding import LangChainEmbedding
from ragentools.rags.rags import BaseRAG
from ragentools.rags.rag_engines import TwoLevelRAGEngine
from ragentools.rags.rerankers import BaseReranker


# inputs
cfg = yaml.safe_load(open("/app/rags/papers/v2/rags_papers_v2.yaml"))
cfg_api = cfg["api"]
cfg_par = cfg["parser"]
cfg_rag = cfg["rag"]


# init clients
api_key = yaml.safe_load(open(cfg_api["api_key_path"]))[cfg_api["api_key_env"]]
api_emb = GoogleGeminiEmbeddingAPI(api_key=api_key, model_name=cfg_api["emb_model_name"], retry_sec=60)
api_chat = GoogleGeminiChatAPI(api_key=api_key, model_name=cfg_api["chat_model_name"], retry_sec=60)
embed_model = LangChainEmbedding(api=api_emb, dim=3072)


# parser
parser = BaseParser(
    reader=PDFReader(pdf_paths=glob.glob(cfg_par["pdf_paths"])),
    chunker=OverlapChunker(
        chunk_size=cfg_par["chunk_size"],
        overlap_size=cfg_par["overlap_size"]
    ),
    saver=PDFSaver(save_folder=cfg_par["save_folder"])
)
parse_result: Iterator[Document] = parser.run(lazy=True)


# rag
rag = BaseRAG(
    rag_engine=TwoLevelRAGEngine(
        vector_store_cls=FAISS,
        embed_model = embed_model,
        api_chat = api_chat
    ),
    reranker=BaseReranker()
)
rag.index(
    docs = parse_result,
    coarse_key = "source_path",
    save_folder = cfg_rag["save_folder"],
)
print(rag.retrieve("What are the key areas that medicine focuses on to ensure well-being?"))
