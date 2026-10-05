from langchain_core.tools import tool
from core import config

def search_document(query: str, vector_db):
    """
    Searches the vector database for relevant document chunks.
    """
    if vector_db is None:
        return "No vector database is currently loaded. Please upload a document first."
    
    try:
        retriever = vector_db.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})
        docs = retriever.invoke(query)
        results = [doc.page_content for doc in docs]
        if not results:
            return f"No relevant content found in the document for query: '{query}'."
        return "\n\n--- CHUNK ---\n\n".join(results)
    except Exception as e:
        return f"Error retrieving document content: {str(e)}"

def search_topic(topic_id: str, query: str, vector_db) -> str:
    """
    Like search_document, but only looks at chunks tagged with one topic
    (the full topic id, e.g. "D0001_T3", saved in ChromaDB metadata at
    ingestion). Returns "" if nothing is tagged with that topic.
    """
    if vector_db is None:
        return ""
    try:
        docs = vector_db.similarity_search(query, k=config.RETRIEVAL_K, filter={"topic_id": topic_id})
        return "\n\n--- CHUNK ---\n\n".join(doc.page_content for doc in docs)
    except Exception as e:
        print(f"Topic search failed: {e}")
        return ""


def create_retrieval_tool(vector_db):
    @tool
    def document_search(query: str) -> str:
        """Search the uploaded document for relevant information, facts, and conceptual context."""
        return search_document(query, vector_db)
    
    return document_search
