import glob
import os
from typing import List

from langchain_community.vectorstores import FAISS

from .base import BaseRetriever, RetrievedChunk
from ragentools.indexers.embedding import CustomEmbedding


class TwoLevelRetriever(BaseRetriever):
    def __init__(self, embed_model: CustomEmbedding, fine_index_folder: str, coarse_index_path: str):
        self.embed_model = embed_model

        # Load coarse-level index
        self.coarse_index = FAISS.load_local(
            coarse_index_path,
            embeddings=self.embed_model,
            allow_dangerous_deserialization=True
        )

        # Load fine-level indices (one per document)
        self.fine_indices = {}
        for fine_index_path in glob.glob(os.path.join(fine_index_folder, "*.faiss")):
            doc_name = os.path.basename(fine_index_path).replace(".faiss", "")
            self.fine_indices[doc_name] = FAISS.load_local(
                fine_index_path,
                embeddings=self.embed_model,
                allow_dangerous_deserialization=True
            )

    def query(self, query_text: str, top_k_coarse: int = 3, top_k_fine: int = 5) -> List[RetrievedChunk]:
        """
        Query two-level FAISS:
        1. Retrieve top-k documents from coarse index
        2. Retrieve top-k chunks from fine indices of those documents
        """
        retrieved_chunks = []
        
        # 1. Coarse retrieval
        coarse_retr = self.coarse_index.similarity_search_with_score(query_text, k=top_k_coarse)
        for coarse_doc, coarse_score in coarse_retr:
            source = coarse_doc.metadata["source_path"]
            if source not in self.fine_indices:
                continue
            fine_index = self.fine_indices[source]

            # 2. Fine retrieval
            fine_retr = fine_index.similarity_search_with_score(query_text, k=top_k_fine)
            for fine_doc, fine_score in fine_retr:
                retrieved_chunks.append(
                    RetrievedChunk(
                        scores=round(float(coarse_score * fine_score), 4),
                        content=fine_doc.page_content,
                        meta=fine_doc.metadata
                    )
            )
        return retrieved_chunks