from abc import ABC, abstractmethod
import glob
import os
from typing import Dict, Iterable, List, Type

from langchain_core.documents import Document
from ragentools.parsers import Document as Easy_Document
from ragentools.rags.utils.embedding import LangChainEmbedding
from ragentools.rags.utils.summarizer import recursive_summarization
from ragentools.rags.rerankers import RetrievedChunk
from ragentools.rags.utils.vectorstore_adapters import BaseVectorStoreAdapter


class BaseRAGEngine(ABC):
    @abstractmethod
    def index(self, docs: List[Easy_Document]) -> None:
        pass

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, query: str, **kwargs) -> List[RetrievedChunk]:
        raise NotImplementedError


class TwoLevelRAGEngine(BaseRAGEngine):
    def __init__(
            self,
            vector_store_cls: Type[BaseVectorStoreAdapter],
            embed_model: LangChainEmbedding,
            api_chat
        ):
        self.vector_store_cls = vector_store_cls
        self.embed_model = embed_model
        self.api_chat = api_chat
        self.coarse: BaseVectorStoreAdapter = None
        self.fine: Dict[str, BaseVectorStoreAdapter] = {}
    
    def _complete_a_fine(
            self,
            fine_chunks: List[Document],
            coarse_level: str,
            save_folder: str,
        ) -> None:
        # complete 1 fine-grained index
        save_path = os.path.join(save_folder, "fine", f"{coarse_level}")
        index = self.vector_store_cls.from_documents(fine_chunks, embedding=self.embed_model)
        self.fine[coarse_level] = index
        if save_folder:
            index.save_local(save_path)
            print(f"Save {len(fine_chunks)} fine chunks to {save_path}")
    
    def _get_a_coarse_doc(self, fine_chunks: List[Document], coarse_level: str) -> Document:
        summary = recursive_summarization(self.api_chat, [d.page_content for d in fine_chunks])
        return Document(page_content=summary, metadata={"coarse_level": coarse_level})

    def index(
            self,
            docs: Iterable[Easy_Document],
            coarse_key: str,
            save_folder: str = "",
        ):  # Make sure docs are sorted by coarse_level_func
        prev_coarse_level = ""
        acc_fine_chunks = []
        acc_coarse_summaries = []
        for i, doc in enumerate(docs):
            coarse_level = os.path.basename(str(doc.metadata.get(coarse_key, "dummy")))
            
            if i != 0 and coarse_level != prev_coarse_level:
                self._complete_a_fine(acc_fine_chunks, prev_coarse_level, save_folder)
                coarse_doc = self._get_a_coarse_doc(acc_fine_chunks, prev_coarse_level)
                acc_coarse_summaries.append(coarse_doc)
                acc_fine_chunks.clear()

            prev_coarse_level = coarse_level
            acc_fine_chunks.append(Document(
                page_content=doc.page_content,
                metadata=doc.metadata
            ))

        # save last fine-grained index
        if len(acc_fine_chunks) > 0:
            self._complete_a_fine(acc_fine_chunks, prev_coarse_level, save_folder)
            coarse_doc = self._get_a_coarse_doc(acc_fine_chunks, prev_coarse_level)
            acc_coarse_summaries.append(coarse_doc)
        
        # complete coarse-grained index
        index = self.vector_store_cls.from_documents(acc_coarse_summaries, embedding=self.embed_model)
        self.coarse = index
        if save_folder:
            index.save_local(os.path.join(save_folder, "coarse"))
            print(f"Save {len(acc_coarse_summaries)} coarse summaries to {save_folder}/coarse")

    def load(self, load_folder: str, embed_model: LangChainEmbedding,) -> None:
        # Load coarse-level index
        coarse_path = os.path.join(load_folder, "coarse")
        self.coarse = self.vector_store_cls.load_local(
            coarse_path,
            embeddings=embed_model,
            allow_dangerous_deserialization=True
        )

        # Load fine-level indices
        fine_folders = glob.glob(os.path.join(load_folder, "fine", "*"))
        for fine_folder in fine_folders:
            name = os.path.basename(fine_folder)
            self.fine[name] = self.vector_store_cls.load_local(
                fine_folder,
                embeddings=embed_model,
                allow_dangerous_deserialization=True
            )
        
    def retrieve(self, query_text: str, top_k_coarse: int = 3, top_k_fine: int = 5) -> List[RetrievedChunk]:
        """
        Query two-level FAISS:
        1. Retrieve top-k documents from coarse index
        2. Retrieve top-k chunks from fine indices of those documents
        """
        retrieved_chunks = []
        
        # 1. Coarse retrieval
        coarse_retr = self.coarse.similarity_search_with_score(query_text, k=top_k_coarse)
        for coarse_doc, coarse_score in coarse_retr:
            coarse_level = coarse_doc.metadata["coarse_level"]
            if coarse_level not in self.fine:
                continue
            fine_index = self.fine[coarse_level]

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
