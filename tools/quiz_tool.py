import json
from typing import List, Dict, Any
from core.provider import get_llm
from tools.parser import safe_json_parse, extract_json_from_text

DIFFICULTY_GUIDE = {
    "easy": "straightforward recall of key terms and basic definitions",
    "medium": "understanding and applying the concepts",
    "hard": "multi-step reasoning, edge cases, and telling similar concepts apart",
}


def generate_quiz(text: str, topic: str = "General", num_questions: int = 5, difficulty: str = "medium") -> List[Dict[str, Any]]:
    """
    Generates structured quiz questions (MCQ and True/False) with explanations from educational text.
    """
    if not text or not text.strip():
        return []

    difficulty = difficulty.lower().strip()
    if difficulty not in DIFFICULTY_GUIDE:
        difficulty = "medium"

    llm = get_llm()

    prompt = f"""
You are an expert AI Assessment Designer and Academic Tutor.

Topic: {topic}
Difficulty: {difficulty} - questions should test {DIFFICULTY_GUIDE[difficulty]}.

Create {num_questions} rigorous practice questions based strictly on the text provided below.
Include a mix of Multiple Choice (MCQ) and True/False questions.

CRITICAL FORMAT RULES:
- Return ONLY a raw JSON array of question objects.
- Do NOT include markdown code blocks, backticks, or introductory remarks.
- For 'mcq', 'options' must be a list of 4 distinct answer choices, and 'answer' must EXACTLY match one of the choices.
- For 'true_false', 'options' must be ['True', 'False'] and 'answer' must be either 'True' or 'False'.
- Include a brief 'explanation' for each question explaining why the answer is correct.

JSON Schema:
[
  {{
    "question": "What is ...?",
    "type": "mcq",
    "options": ["Option A", "Option B", "Option C", "Option D"],
    "answer": "Option B",
    "explanation": "Because ..."
  }},
  {{
    "question": "Statement ...?",
    "type": "true_false",
    "options": ["True", "False"],
    "answer": "True",
    "explanation": "Because ..."
  }}
]

TEXT CONTENT:
{text[:8000]}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)

        questions = safe_json_parse(content)
        if not questions or not isinstance(questions, list):
            questions = extract_json_from_text(content)

        validated_quiz = []
        for q in (questions if isinstance(questions, list) else []):
            if isinstance(q, dict) and "question" in q and "answer" in q:
                q_type = q.get("type", "mcq")
                options = q.get("options", [])
                if q_type == "true_false" and not options:
                    options = ["True", "False"]
                
                validated_quiz.append({
                    "question": str(q["question"]).strip(),
                    "type": q_type,
                    "options": [str(opt).strip() for opt in options],
                    "answer": str(q["answer"]).strip(),
                    "explanation": str(q.get("explanation", "Correct answer based on the study text.")).strip()
                })
        return validated_quiz
    except Exception as e:
        print(f"Error generating quiz: {e}")
        return []
