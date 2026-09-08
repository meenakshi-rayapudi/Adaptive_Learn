from core.provider import get_llm

def generate_study_plan(content: str, doc_name: str = "Uploaded Document") -> str:
    """
    Analyzes educational content and generates an actionable 3-day study mastery plan.
    """
    if not content or not content.strip():
        return "No study content available to generate a plan. Please upload a document first."

    llm = get_llm()

    prompt = f"""
You are an expert Educational Learning Architect and Academic Coach.
Document/Topic Title: '{doc_name}'

Based on the study material provided below, generate a comprehensive, highly motivating "3-Day Mastery Plan" for a student.

Structure your response clearly using the following markdown format:

# 📅 3-Day Mastery Plan: {doc_name}

### 📊 Complexity & Difficulty Assessment
- **Estimated Study Time:** [e.g. 2 hours/day]
- **Difficulty Level:** [Beginner / Intermediate / Advanced]
- **Prerequisites & Key Themes:** [Summary]

---

### 🟢 Day 1: Foundations & Core Concepts
- **Focus Areas:** [List 3-4 foundational definitions & principles]
- **Key Terms to Master:** [Bullet points]
- **Target Milestone:** [What should be clear by end of Day 1]

---

### 🟡 Day 2: Deep Dive & Conceptual Connections
- **Focus Areas:** [Challenging mechanisms, relationships, problem solving]
- **Active Practice Strategy:** [How to apply the knowledge]
- **Target Milestone:** [What complex questions they should be able to answer]

---

### 🔴 Day 3: Synthesis, Self-Testing & Exam Readiness
- **Focus Areas:** [Comprehensive review and active recall]
- **Self-Assessment Check:** [Key questions to answer without notes]
- **Final Mastery Goal:** [Definition of complete readiness]

---

STUDY MATERIAL:
{content[:7000]}
"""

    try:
        response = llm.invoke(prompt)
        return response.content if hasattr(response, "content") else str(response)
    except Exception as e:
        return f"Could not generate study plan: {str(e)}"
