from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field


class FlashcardItem(BaseModel):
    front: str = Field(description="The question, key term, or concept on the front of the flashcard")
    back: str = Field(description="The explanation, definition, or answer on the back of the flashcard")
    topic: Optional[str] = Field(default="General", description="Topic or sub-domain of the flashcard")
    difficulty: Optional[Literal["easy", "medium", "hard"]] = Field(default="medium", description="Estimated difficulty level")


class FlashcardDeck(BaseModel):
    deck_title: str = Field(description="Title or topic of the flashcard deck")
    cards: List[FlashcardItem] = Field(description="List of flashcards")


class QuizQuestionItem(BaseModel):
    question: str = Field(description="The quiz question text")
    type: Literal["mcq", "true_false"] = Field(description="Question type: 'mcq' or 'true_false'")
    options: List[str] = Field(default_factory=list, description="4 distinct options for MCQ; empty or ['True', 'False'] for true/false")
    answer: str = Field(description="The exact correct option string")
    explanation: Optional[str] = Field(default="", description="Educational explanation for why the answer is correct")


class QuizDeck(BaseModel):
    topic: str = Field(description="Topic of the quiz")
    questions: List[QuizQuestionItem] = Field(description="List of quiz questions")


class StudyPlanSchedule(BaseModel):
    topic: str = Field(description="Topic or title of the study material")
    complexity_level: str = Field(description="Brief complexity analysis of the material")
    day1_foundations: str = Field(description="Foundational concepts and terminology to master on Day 1")
    day2_deep_dive: str = Field(description="Challenging mechanisms, calculations, or deep dive on Day 2")
    day3_assessment: str = Field(description="Practical synthesis, self-testing, and mastery check on Day 3")
    markdown_summary: str = Field(description="Complete formatted markdown presentation of the 3-day study plan")


class MissedQuestionRecord(BaseModel):
    question: str
    correct_answer: str
    user_answer: str


class RemedialReport(BaseModel):
    concept_breakdown: str = Field(description="Identification and explanation of the core concepts the student missed")
    mnemonics: List[str] = Field(default_factory=list, description="Memory tips or mnemonics for rapid recall")
    deep_dive_questions: List[str] = Field(default_factory=list, description="Conceptual reflection questions to test comprehension")
    markdown_guide: str = Field(description="Complete structured markdown remedial study guide")


class AudioGenerationResult(BaseModel):
    file_path: str = Field(description="Path to the generated MP3 file")
    text_content: str = Field(description="The text content that was vocalized")
    duration_estimate_seconds: Optional[int] = Field(default=None, description="Estimated duration in seconds")
