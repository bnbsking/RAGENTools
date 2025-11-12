from ragentools.rags.rags import BaseRAG
from ragentools.rags.rag_engines import MSGraphRAGEngine
from ragentools.rags.rerankers import BaseReranker

# Make sure Indexing is done as in ./Notes.md

rag = BaseRAG(
    rag_engine=MSGraphRAGEngine(folder="/app/rags/papers/v3"),
    reranker=BaseReranker()
)
print(rag.retrieve("What are the key areas that medicine focuses on to ensure well-being?"))
