import json
from typing import List, Dict, Any, Optional
from core.provider import get_llm
from tools.parser import safe_json_parse, extract_json_from_text
from langchain_core.tools import tool

def generate_flashcards(text: str, topic: str = "General", num_cards: int = 10, topic_id: str = None) -> List[Dict[str, Any]]:
    """
    Generates structured flashcards from educational text for a given topic.
    """
    if not text or not text.strip():
        return []

    llm = get_llm()
    topic_id = topic_id or topic

    prompt = f"""
You are an expert AI Academic Tutor that creates active-recall study flashcards.

Topic: {topic}

IMPORTANT INSTRUCTIONS:
- Generate up to {num_cards} high-yield, conceptual flashcards based ONLY on the provided text.
- Each flashcard must have a clear "front" (question/concept/term) and a comprehensive "back" (explanation/answer/definition).
- Return ONLY a raw JSON array of objects. No intro text, no markdown wrappers, no backticks.

JSON Schema:
[
  {{
    "front": "What is ...?",
    "back": "Detailed definition or mechanism ...",
    "topic": "{topic}",
    "difficulty": "medium"
  }}
]

TEXT CONTENT:
{text[:8000]}
"""

    try:
        response = llm.invoke(prompt)
        content = response.content if hasattr(response, "content") else str(response)
        
        cards = safe_json_parse(content)
        if not cards or not isinstance(cards, list):
            cards = extract_json_from_text(content)
            
        validated_cards = []
        for c in (cards if isinstance(cards, list) else []):
            if isinstance(c, dict) and "front" in c and "back" in c:
                validated_cards.append({
                    "front": str(c["front"]).strip(),
                    "back": str(c["back"]).strip(),
                    "topic": c.get("topic", topic),
                    "topic_id": c.get("topic_id") or topic_id,
                    "difficulty": c.get("difficulty", "medium"),
                    "status": "new",
                    "review_count": 0,
                    "last_seen": None
                })
        return validated_cards
    except Exception as e:
        print(f"Error generating flashcards: {e}")
        return []
