from typing import List, Dict, Any, Union
from core.provider import get_llm

def generate_remedial_guide(missed_questions: Union[List[Dict[str, Any]], str], context: str = "") -> str:
    """
    Analyzes student errors on quiz questions and generates a personalized diagnostic remedial guide.
    """
    if not missed_questions:
        return "No errors identified. The student scored 100%! Ready for advanced topics."

    llm = get_llm()

    if isinstance(missed_questions, list):
        questions_str = ""
        for i, q in enumerate(missed_questions):
            q_text = q.get("question", "Unknown question")
            correct = q.get("answer") or q.get("correct_answer", "N/A")
            user_ans = q.get("user_answer", "N/A")
            explanation = q.get("explanation", "")
            questions_str += f"{i+1}. Question: {q_text}\n   - Correct Answer: {correct}\n   - Student Selected: {user_ans}\n"
            if explanation:
                questions_str += f"   - Concept Note: {explanation}\n"
            questions_str += "\n"
    else:
        questions_str = str(missed_questions)

    prompt = f"""
You are an expert Adaptive AI Tutor specialized in cognitive diagnostic assessment and remediation.

A student just completed a practice assessment and missed the following questions:

{questions_str}

RELEVANT STUDY CONTEXT:
{context[:6000] if context else "Use general domain knowledge based on the questions."}

TASK:
Generate a comprehensive, supportive, and actionable "Personalized Remedial Mastery Guide".

Structure your response with the following sections:
# 📘 Personalized Remedial Mastery Guide

### 🔍 1. Root-Cause Concept Diagnosis
- Pinpoint the exact fundamental principles or misconceptions that led to the errors.
- Explain clearly *why* the student's chosen answer was incorrect versus the correct answer.

### 💡 2. Simplified Conceptual Refresher
- Provide clear, plain-language explanations of the missed concepts with analogies or real-world examples.

### 🧠 3. High-Yield Mnemonics & Memory Anchors
- Provide practical memory devices, acronyms, or visual cues to lock in these facts permanently.

### 🎯 4. Targeted Checkpoint Questions
- Provide 2-3 conceptual practice questions with hints to test if the student has bridged the learning gap.

Format in clean, professional markdown with emojis and clear headers.
"""

    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Could not generate remedial guide: {str(e)}"
