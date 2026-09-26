import os

_embedding_model = None

def get_embedding_model():
    """
    Lazy singleton loader for local dense HuggingFace embeddings.
    """
    global _embedding_model
    if _embedding_model is None:
        from langchain_huggingface import HuggingFaceEmbeddings
        _embedding_model = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    return _embedding_model


def store_chunks(chunks, document_name="default", metadatas=None):
    """
    Stores document chunks into persistent local ChromaDB.

    `metadatas`, when provided, must be a list parallel to `chunks` (e.g.
    [{"topic_id": "T1", "chunk_index": 0}, ...]) so that topic-tagged
    retrieval and analytics can filter/join on `topic_id` later without
    re-embedding anything.
    """
    if not chunks:
        print("Warning: No chunks found to store in the vector database.")
        return None

    from langchain_community.vectorstores import Chroma

    clean_name = "".join(c for c in document_name if c.isalnum() or c in (" ", "_")).rstrip()
    persist_dir = os.path.join("chroma_db", clean_name.replace(" ", "_"))

    embedding = get_embedding_model()

    if metadatas is not None and len(metadatas) != len(chunks):
        raise ValueError("metadatas must be the same length as chunks")

    db = Chroma.from_texts(
        texts=chunks,
        embedding=embedding,
        metadatas=metadatas,
        persist_directory=persist_dir
    )

    return db