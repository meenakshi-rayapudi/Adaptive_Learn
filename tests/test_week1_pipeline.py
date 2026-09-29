import unittest
import os
import sys
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from database.db import init_db, get_session
from database.schema import Student, Document, Topic, Chunk
from database.crud import (
    get_or_create_student,
    list_students,
    create_document,
    replace_topics,
    get_topics_for_document,
    replace_chunks,
    get_chunks_for_document,
    get_chunk_counts_by_topic,
)
from core.topic_extractor import _cosine_similarity_matrix
from core.engine import _make_document_id
from engine.feature_spec import (
    FeatureVector,
    log1p_transform,
    clip_scale,
    recency_penalty,
    forgetting_curve_retention,
    FEATURE_NAMES,
)


class TestWeek1DatabaseAndChunkPersistence(unittest.TestCase):
    def setUp(self):
        init_db()
        self.doc_id = "TEST_DOC_001"
        self.student_id = "TEST_STU_001"

    def test_document_and_topic_creation(self):
        # 1. Create document
        doc = create_document(self.doc_id, "test_file.pdf", source_type="pdf", uploaded_by=self.student_id)
        self.assertEqual(doc["id"], self.doc_id)

        # 2. Replace topics
        raw_topics = [
            {"topic_id": "T1", "name": "Deadlocks", "description": "Banker algorithm"},
            {"topic_id": "T2", "name": "Memory Management", "description": "Paging and virtual memory"},
        ]
        saved = replace_topics(self.doc_id, raw_topics)
        self.assertEqual(len(saved), 2)
        self.assertEqual(saved[0]["id"], f"{self.doc_id}_T1")
        self.assertEqual(saved[0]["topic_key"], "T1")
        self.assertEqual(saved[1]["id"], f"{self.doc_id}_T2")
        self.assertEqual(saved[1]["topic_key"], "T2")

        # 3. Query topics back
        retrieved = get_topics_for_document(self.doc_id)
        self.assertEqual(len(retrieved), 2)
        self.assertEqual(retrieved[0]["id"], f"{self.doc_id}_T1")

    def test_chunk_persistence_and_topic_counts(self):
        # Create doc and topics first
        create_document(self.doc_id, "test_file.pdf")
        replace_topics(self.doc_id, [
            {"topic_id": "T1", "name": "Deadlocks", "description": "Banker algorithm"},
            {"topic_id": "T2", "name": "Memory Management", "description": "Paging"},
        ])

        chunks_data = [
            {
                "chunk_index": 0,
                "topic_id": f"{self.doc_id}_T1",
                "chroma_id": f"{self.doc_id}_c0",
                "text_preview": "Processes waiting on locks can enter deadlocks.",
                "similarity": 0.88,
            },
            {
                "chunk_index": 1,
                "topic_id": f"{self.doc_id}_T1",
                "chroma_id": f"{self.doc_id}_c1",
                "text_preview": "Banker's algorithm checks safe states.",
                "similarity": 0.92,
            },
            {
                "chunk_index": 2,
                "topic_id": f"{self.doc_id}_T2",
                "chroma_id": f"{self.doc_id}_c2",
                "text_preview": "Virtual memory maps logical pages to physical frames.",
                "similarity": 0.85,
            },
        ]

        saved_chunks = replace_chunks(self.doc_id, chunks_data)
        self.assertEqual(len(saved_chunks), 3)

        # Verify retrieved chunks
        chunks_in_db = get_chunks_for_document(self.doc_id)
        self.assertEqual(len(chunks_in_db), 3)
        self.assertEqual(chunks_in_db[0]["chroma_id"], f"{self.doc_id}_c0")
        self.assertEqual(chunks_in_db[0]["topic_id"], f"{self.doc_id}_T1")
        self.assertAlmostEqual(chunks_in_db[0]["similarity"], 0.88, places=2)

        # Verify SQL group by counts
        counts = get_chunk_counts_by_topic(self.doc_id)
        self.assertEqual(counts[f"{self.doc_id}_T1"], 2)
        self.assertEqual(counts[f"{self.doc_id}_T2"], 1)

    def test_deterministic_document_id(self):
        id1 = _make_document_id("my_lecture.pdf")
        id2 = _make_document_id("my_lecture.pdf")
        id3 = _make_document_id("different_lecture.pdf")
        self.assertEqual(id1, id2)
        self.assertNotEqual(id1, id3)
        self.assertTrue(id1.startswith("D"))


class TestTopicCosineSimilarity(unittest.TestCase):
    def test_cosine_similarity_matrix(self):
        # Two orthogonal vectors
        chunk_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        topic_vecs = np.array([[1.0, 0.0], [0.0, 1.0]])
        sims = _cosine_similarity_matrix(chunk_vecs, topic_vecs)
        self.assertAlmostEqual(sims[0, 0], 1.0, places=4)
        self.assertAlmostEqual(sims[0, 1], 0.0, places=4)
        self.assertAlmostEqual(sims[1, 0], 0.0, places=4)
        self.assertAlmostEqual(sims[1, 1], 1.0, places=4)


class TestFeatureSpec(unittest.TestCase):
    def test_feature_transforms(self):
        fv = FeatureVector(
            student_id="S001",
            topic_id="T1",
            q=0.8,
            a=0.6,
            t=5000.0,
            r=15.0,
            f=0.7,
            e=10.0,
            h=5.0,
        )
        arr = fv.to_model_array()
        self.assertEqual(len(arr), 7)
        self.assertAlmostEqual(arr[0], 0.8) # Q
        self.assertAlmostEqual(arr[1], 0.6) # A
        self.assertAlmostEqual(arr[2], log1p_transform(5000.0)) # T
        self.assertAlmostEqual(arr[3], 15.0 / 30.0) # R
        self.assertAlmostEqual(arr[4], 0.7) # F
        self.assertAlmostEqual(arr[5], log1p_transform(10.0)) # E
        self.assertAlmostEqual(arr[6], 5.0 / 10.0) # H

    def test_forgetting_curve(self):
        retention_0 = forgetting_curve_retention(0, 10.0)
        self.assertAlmostEqual(retention_0, 1.0)
        retention_10 = forgetting_curve_retention(10, 10.0)
        self.assertAlmostEqual(retention_10, 0.367879, places=4)


if __name__ == "__main__":
    unittest.main()
