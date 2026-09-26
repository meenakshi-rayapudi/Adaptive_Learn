"""
Small, framework-agnostic data-access helpers used by both the Streamlit UI
(core/engine.py, ui/components/sidebar.py) and offline scripts (scripts/seed_student_data.py).

Kept intentionally thin for Week 1: just enough CRUD to initialize students/
documents/topics. Aggregation into the ML feature vector (Q, A, T, R, F, E, H)
is Week 3 scope (engine/features.py).
"""

from typing import List, Dict, Optional

from .db import get_session
from .schema import Student, Document, Topic


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
    `topics` items look like {"topic_id": "T1", "name": ..., "description": ...}
    (see core/topic_extractor.py).
    """
    session = get_session()
    try:
        session.query(Topic).filter(Topic.document_id == document_id).delete()

        saved = []
        for t in topics:
            topic_key = t["topic_id"]
            full_id = f"{document_id}_{topic_key}"
            topic = Topic(
                id=full_id,
                document_id=document_id,
                topic_key=topic_key,
                name=t.get("name", topic_key),
                description=t.get("description", ""),
            )
            session.add(topic)
            saved.append({"id": full_id, "topic_key": topic_key, "name": topic.name, "description": topic.description})

        session.commit()
        return saved
    finally:
        session.close()


def get_topics_for_document(document_id: str) -> List[Dict]:
    session = get_session()
    try:
        topics = session.query(Topic).filter(Topic.document_id == document_id).all()
        return [{"id": t.id, "topic_key": t.topic_key, "name": t.name, "description": t.description} for t in topics]
    finally:
        session.close()
