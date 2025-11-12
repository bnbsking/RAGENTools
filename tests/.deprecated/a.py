from neo4j import GraphDatabase
from neo4j_graphrag.indexes import create_vector_index, upsert_vectors
from neo4j_graphrag.types import EntityType
from neo4j_graphrag.retrievers import VectorRetriever

from ragentools.api_calls.google_gemini import GoogleGeminiEmbeddingAPI
from neo4j_graphrag.embeddings.base import Embedder


class MyAPI(Embedder):
    def __init__(self, api_key: str):
        super().__init__()
        self.api = GoogleGeminiEmbeddingAPI(
            api_key=api_key,
            model_name="gemini-embedding-001",
            retry_sec=60,
        )
    
    def embed_query(self, text: str) -> list[float]:
        response = self.api.run_batches([text])
        return response[0]
        

URI = "neo4j+s://f24e556a.databases.neo4j.io"
AUTH = ("neo4j", "12Rql35m6kFmkEgOsnhPRyjFNNZVabh344dZx62I5wI")

INDEX_NAME = "vector-index-name"


# Connect to Neo4j database
driver = GraphDatabase.driver(URI, auth=AUTH)
driver.verify_connectivity()


with driver.session() as session:
    # Drop old index if it exists
    session.run(f"DROP INDEX `{INDEX_NAME}` IF EXISTS")  # Cypher


# Creating the index
create_vector_index(
    driver,
    INDEX_NAME,
    label="Document",
    embedding_property="vectorProperty",
    dimensions=3072,
    similarity_fn="euclidean",
)

query_text = "How do I do similarity search in Neo4j?"
embedder = MyAPI(api_key="")
vector = embedder.embed_query(query_text)  # Example vector with 3072 dimensions

upsert_vectors(
    driver,
    ids=["1234"],
    embedding_property="vectorProperty",
    embeddings=[vector],
    entity_type=EntityType.NODE
)

# Create Embedder object
# Note: An OPENAI_API_KEY environment variable is required here


# Initialize the retriever
retriever = VectorRetriever(driver, INDEX_NAME, embedder)

# Run the similarity search
query_text = "How do I do similarity search in ?"
response = retriever.search(query_text=query_text, top_k=5)
print(response)


with driver.session() as session:
    result = session.run("""
        SHOW INDEXES
        YIELD name, type, entityType, labelsOrTypes, properties
        RETURN name, type, entityType, labelsOrTypes, properties
    """)
    print("=== Index List ===")
    for record in result:
        print(record.data())


with driver.session() as session:
    result = session.run("""
        MATCH (n:Document)
        WHERE n.vectorProperty IS NOT NULL
        RETURN count(n) AS vector_count
    """)
    print("Vectors stored as node properties:", result.single()["vector_count"])
