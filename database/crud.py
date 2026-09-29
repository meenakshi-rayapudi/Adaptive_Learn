"""
Small, framework-agnostic data-access helpers used by both the Streamlit UI
(core/engine.py, ui/components/sidebar.py) and offline scripts (scripts/seed_student_data.py).

Kept intentionally thin for Week 1: just enough CRUD to initialize students/
documents/topics. Aggregation into the ML feature vector (Q, A, T, R, F, E, H)
is Week 3 scope (engine/features.py).
"""

from typing import List, Dict, Optional

from .db import get_session
from .schema import Student, Document, Topic, Chunk


def get_or_create_student(student_id: str, name: Optional[str] = None, archetype: Optional[str] = None, is_synthetic: bool = False) -> Dict:
    session = get_session()
    try:
        student = session.get(Student, student_id)
        if student is None:
            student = Student(
                id=student_id,
                name=name or student_id,
                archetype=archetype,
                is_synthetic=is_synthetic,
            )
            session.add(student)
            session.commit()
        return {"id": student.id, "name": student.name, "archetype": student.archetype}
    finally:
        session.close()


def list_students(include_synthetic: bool = True) -> List[Dict]:
    session = get_session()
    try:
        query = session.query(Student)
        if not include_synthetic:
            query = query.filter(Student.is_synthetic.is_(False))
        students = query.order_by(Student.created_at.asc()).all()
        return [{"id": s.id, "name": s.name, "archetype": s.archetype} for s in students]
    finally:
        session.close()


def create_document(document_id: str, filename: str, source_type: str = "pdf", uploaded_by: Optional[str] = None) -> Dict:
    session = get_session()
    try:
        document = session.get(Document, document_id)
        if document is None:
            document = Document(
                id=document_id,
                filename=filename,
                source_type=source_type,
                uploaded_by=uploaded_by,
            )
            session.add(document)
            session.commit()
        return {"id": document.id, "filename": document.filename}
    finally:
        session.close()


def replace_topics(document_id: str, topics: List[Dict]) -> List[Dict]:
    """
    Replaces all topics for a document with the freshly extracted set.
    Accepts topic dicts with 'topic_id', 'topic_key', or 'id'.
    Returns list of dicts containing 'id' (canonical full ID, e.g. D0001_T1),
    'topic_id' (alias for full ID), 'topic_key' (short key e.g. T1),
    'name', and 'description'.
    """
    session = get_session()
    try:
        session.query(Topic).filter(Topic.document_id == document_id).delete()

        saved = []
        for t in topics:
            raw_id = str(t.get("topic_id") or t.get("topic_key") or t.get("id", ""))
            if raw_id.startswith(f"{document_id}_"):
                topic_key = raw_id[len(f"{document_id}_"):]
                full_id = raw_id
            else:
                topic_key = raw_id
                full_id = f"{document_id}_{topic_key}"

            topic = Topic(
                id=full_id,
                document_id=document_id,
                topic_key=topic_key,
                name=t.get("name", topic_key),
                description=t.get("description", ""),
            )
            session.add(topic)
            saved.append({
                "id": full_id,
                "topic_id": full_id,
                "topic_key": topic_key,
                "name": topic.name,
                "description": topic.description,
            })

        session.commit()
        return saved
    finally:
        session.close()


def get_topics_for_document(document_id: str) -> List[Dict]:
    session = get_session()
    try:
        topics = session.query(Topic).filter(Topic.document_id == document_id).all()
        return [
            {
                "id": t.id,
                "topic_id": t.id,
                "topic_key": t.topic_key,
                "name": t.name,
                "description": t.description,
            }
            for t in topics
        ]
    finally:
        session.close()


def replace_chunks(document_id: str, chunks_data: List[Dict]) -> List[Dict]:
    """
    Replaces all chunk records in SQLite for a document with the freshly tagged chunks.
    Each item in `chunks_data`:
      {
        "chunk_index": int,
        "topic_id": Optional[str],  # full canonical Topic.id e.g. "D0001_T1"
        "chroma_id": Optional[str], # e.g. "D0001_c0"
        "text_preview": str,
        "similarity": Optional[float]
      }
    """
    session = get_session()
    try:
        session.query(Chunk).filter(Chunk.document_id == document_id).delete()

        saved = []
        for c in chunks_data:
            chunk = Chunk(
                document_id=document_id,
                topic_id=c.get("topic_id"),
                chunk_index=c["chunk_index"],
                chroma_id=c.get("chroma_id"),
                text_preview=c.get("text_preview", "")[:300],
                similarity=c.get("similarity"),
            )
            session.add(chunk)
            saved.append({
                "document_id": document_id,
                "topic_id": chunk.topic_id,
                "chunk_index": chunk.chunk_index,
                "chroma_id": chunk.chroma_id,
                "similarity": chunk.similarity,
            })

        session.commit()
        return saved
    finally:
        session.close()


def get_chunks_for_document(document_id: str) -> List[Dict]:
    """Retrieves all chunk metadata rows stored in SQLite for a document."""
    session = get_session()
    try:
        chunks = session.query(Chunk).filter(Chunk.document_id == document_id).order_by(Chunk.chunk_index.asc()).all()
        return [
            {
                "id": c.id,
                "document_id": c.document_id,
                "topic_id": c.topic_id,
                "chunk_index": c.chunk_index,
                "chroma_id": c.chroma_id,
                "text_preview": c.text_preview,
                "similarity": c.similarity,
            }
            for c in chunks
        ]
    finally:
        session.close()


def get_chunk_counts_by_topic(document_id: str) -> Dict[str, int]:
    """
    Returns SQL aggregate count of chunks per topic for a document:
    { "D0001_T1": 5, "D0001_T2": 3, ... }
    """
    session = get_session()
    try:
        from sqlalchemy import func
        results = (
            session.query(Chunk.topic_id, func.count(Chunk.id))
            .filter(Chunk.document_id == document_id)
            .group_by(Chunk.topic_id)
            .all()
        )
        return {topic_id: count for topic_id, count in results if topic_id is not None}
    finally:
        session.close()

