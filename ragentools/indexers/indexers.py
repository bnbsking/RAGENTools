import glob
import os

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
import pandas as pd

from ragentools.indexers.embedding import CustomEmbedding
from ragentools.indexers.summarizer import recursive_summarization


def two_level_indexing(
        parsed_csv_folder: str,
        indices_save_folder: str,
        embed_model: CustomEmbedding,
        api_chat
    ):
    # Create fine-grained indices
    for csv_path in glob.glob(os.path.join(parsed_csv_folder, "*.csv")):
            df = pd.read_csv(csv_path)
            docs = [
                Document(
                    page_content=row['chunk'],
                    metadata={"source_path": row['source_path'], "page": row['page']})
                for _, row in df.iterrows()
            ]
            faiss_index = FAISS.from_documents(docs, embedding=embed_model)
            faiss_index.save_local(os.path.join(indices_save_folder, os.path.basename(csv_path) + ".faiss"))

    # Create coarse-grained index
    file_summary_list = []
    for csv_path in glob.glob(os.path.join(parsed_csv_folder, "*.csv")):
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
    coarse_faiss_index.save_local(os.path.join(indices_save_folder, "coarse_grained_index.faiss"))
