"""
Topic taxonomy extraction (see docs/architectural_decisions.md, Decision 1: "Option 1C").

Cost model: exactly ONE LLM call per document, regardless of chunk count.
  1. Ask the LLM to read a summary/outline of the document and propose
     5-15 canonical topics with short descriptions.
  2. Embed those topic descriptions once, locally, with the same
     sentence-transformers model already used for chunk embeddings.
  3. Tag every chunk by local cosine similarity against the topic vectors
     (no further LLM calls).
"""

import json
from typing import List, Dict, Any, Optional

import numpy as np

from core.provider import get_llm
from core.vector_store import get_embedding_model
from tools.parser import safe_json_parse, extract_json_from_text

MIN_TOPICS = 5
MAX_TOPICS = 15

# Only feed the LLM a bounded slice of the document (outline-equivalent) to
# keep the single extraction call fast and cheap.
MAX_CHARS_FOR_EXTRACTION = 12000


def extract_document_topics(document_text: str, llm=None, max_topics: int = MAX_TOPICS) -> List[Dict[str, str]]:
    """
    Single LLM call: returns a canonical topic taxonomy for the document.
    Each item: {"topic_id": "T1", "name": ..., "description": ...}
    """
    if not document_text or not document_text.strip():
        return []

    llm = llm or get_llm()
    max_topics = max(MIN_TOPICS, min(max_topics, MAX_TOPICS))

    prompt = f"""
You are an expert curriculum designer extracting a canonical topic taxonomy from a study document.

Read the document content below and identify between {MIN_TOPICS} and {max_topics} canonical topics
that a student would need to master. Topics should be specific enough to be individually
quizzable (e.g. "Deadlock Handling" not just "Operating Systems"), but not so narrow that
they duplicate each other.

CRITICAL FORMAT RULES:
- Return ONLY a raw JSON array of objects. No markdown fences, no commentary.
- "topic_id" must be short and sequential: "T1", "T2", "T3", ...
- "description" should be 1-2 sentences summarizing what the topic covers.

JSON Schema:
[
  {{"topic_id": "T1", "name": "Deadlock Handling", "description": "Banker's algorithm, prevention, detection, and recovery strategies for process deadlocks."}}
]

DOCUMENT CONTENT:
{document_text[:MAX_CHARS_FOR_EXTRACTION]}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        topics = safe_json_parse(content)
        if not topics or not isinstance(topics, list):
            topics = extract_json_from_text(content)

        validated = []
        for i, t in enumerate(topics if isinstance(topics, list) else []):
            if isinstance(t, dict) and "name" in t:
                validated.append({
                    "topic_id": str(t.get("topic_id", f"T{i + 1}")).strip(),
                    "name": str(t["name"]).strip(),
                    "description": str(t.get("description", "")).strip(),
                })
        return validated[:max_topics]
    except Exception as e:
        print(f"Error extracting topics: {e}")
        return []


def _cosine_similarity_matrix(chunk_vectors: np.ndarray, topic_vectors: np.ndarray) -> np.ndarray:
    """Returns a (num_chunks, num_topics) matrix of cosine similarities."""
    chunk_norm = chunk_vectors / (np.linalg.norm(chunk_vectors, axis=1, keepdims=True) + 1e-10)
    topic_norm = topic_vectors / (np.linalg.norm(topic_vectors, axis=1, keepdims=True) + 1e-10)
    return chunk_norm @ topic_norm.T


def assign_topics_to_chunks(chunks: List[str], topics: List[Dict[str, str]], embedding_model=None) -> List[Dict[str, Any]]:
    """
    Tags each chunk with its closest topic via local cosine similarity.
    Zero additional LLM calls. Returns one dict per chunk:
      {"chunk": text, "topic_id": "T1", "similarity": 0.83}
    If `topics` is empty, every chunk is returned with topic_id=None.
    """
    if not chunks:
        return []

    if not topics:
        return [{"chunk": c, "topic_id": None, "similarity": None} for c in chunks]

    model = embedding_model or get_embedding_model()

    topic_texts = [f"{t['name']}: {t.get('description', '')}" for t in topics]
    topic_vectors = np.array(model.embed_documents(topic_texts))
    chunk_vectors = np.array(model.embed_documents(chunks))

    sims = _cosine_similarity_matrix(chunk_vectors, topic_vectors)
    best_idx = np.argmax(sims, axis=1)

    results = []
    for i, chunk in enumerate(chunks):
        j = best_idx[i]
        results.append({
            "chunk": chunk,
            "topic_id": topics[j]["topic_id"],
            "similarity": float(sims[i, j]),
        })
    return results


def extract_and_tag_document(document_text: str, chunks: List[str], llm=None, embedding_model=None, max_topics: int = MAX_TOPICS) -> Dict[str, Any]:
    """
    Convenience wrapper: runs the full 1-LLM-call pipeline and returns both
    the topic taxonomy and the per-chunk topic assignments.
    """
    topics = extract_document_topics(document_text, llm=llm, max_topics=max_topics)
    tagged_chunks = assign_topics_to_chunks(chunks, topics, embedding_model=embedding_model)

    topic_chunk_counts = {t["topic_id"]: 0 for t in topics}
    for tc in tagged_chunks:
        if tc["topic_id"] in topic_chunk_counts:
            topic_chunk_counts[tc["topic_id"]] += 1

    return {
        "topics": topics,
        "tagged_chunks": tagged_chunks,
        "topic_chunk_counts": topic_chunk_counts,
    }
