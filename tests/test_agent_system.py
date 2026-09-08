import unittest
import os
import sys

# Ensure project root in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from models.schemas import (
    FlashcardItem, FlashcardDeck, QuizQuestionItem, QuizDeck,
    StudyPlanSchedule, RemedialReport, AudioGenerationResult
)
from tools.parser import safe_json_parse, extract_json_from_text
from tools.chunker import chunk_text
from tools.youtube_tool import extract_video_id
from utils.audio_utils import clean_text
from core.engine import AgentArtifactStore, TutorAgent

class TestSchemasAndParsing(unittest.TestCase):
    def test_flashcard_schema(self):
        item = FlashcardItem(front="What is ATP?", back="Adenosine Triphosphate, cellular energy currency.")
        self.assertEqual(item.front, "What is ATP?")
        self.assertEqual(item.difficulty, "medium")

    def test_quiz_schema(self):
        item = QuizQuestionItem(
            question="Is mitochondria the powerhouse of the cell?",
            type="true_false",
            options=["True", "False"],
            answer="True",
            explanation="Mitochondria produce cellular ATP."
        )
        self.assertEqual(item.type, "true_false")
        self.assertEqual(item.answer, "True")

    def test_safe_json_parse(self):
        raw = '`json\n[{"front": "Q", "back": "A"}]\n`'
        parsed = safe_json_parse(raw)
        self.assertEqual(len(parsed), 1)
        self.assertEqual(parsed[0]["front"], "Q")

    def test_extract_json_from_text(self):
        messy = 'Sure, here are your cards:\n[{"front": "Concept 1", "back": "Def 1"}]\nHope this helps!'
        extracted = extract_json_from_text(messy)
        self.assertEqual(len(extracted), 1)
        self.assertEqual(extracted[0]["back"], "Def 1")

    def test_chunker(self):
        text = "word " * 300
        chunks = chunk_text(text)
        self.assertTrue(len(chunks) >= 1)

    def test_youtube_url_extraction(self):
        url1 = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
        url2 = "https://youtu.be/dQw4w9WgXcQ?si=123"
        self.assertEqual(extract_video_id(url1), "dQw4w9WgXcQ")
        self.assertEqual(extract_video_id(url2), "dQw4w9WgXcQ")

    def test_audio_clean_text(self):
        dirty = "# Header\n| col1 | col2 |\n|---|---|\ncode and **bold** text."
        cleaned = clean_text(dirty)
        self.assertNotIn("#", cleaned)
        self.assertNotIn("|", cleaned)
        self.assertIn("code and bold text.", cleaned)

    def test_agent_artifact_store(self):
        store = AgentArtifactStore()
        store.flashcards = [{"front": "A", "back": "B"}]
        store.study_plan = "Plan"
        artifacts = store.get_artifacts()
        self.assertEqual(len(artifacts["flashcards"]), 1)
        self.assertEqual(artifacts["study_plan"], "Plan")
        store.reset_recent()
        self.assertIsNone(store.flashcards)
        self.assertIsNone(store.study_plan)

if __name__ == "__main__":
    unittest.main()
