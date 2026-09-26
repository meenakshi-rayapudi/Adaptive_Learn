import os
import sys
from core.engine import process_document, create_tutor_agent, run_agent_query
from tools.parser import text_to_md
from core.provider import get_llm

if __name__ == "__main__":
    print("=" * 60)
    print("🎓 Veras Autonomous AI Academic Tutor (CLI Mode)")
    print("=" * 60)

    doc_path = input("Enter path to study document (PDF/TXT/MD) [or press Enter for sample]: ").strip()
    
    if not doc_path:
        sample_file = "sample_study_material.txt"
        if not os.path.exists(sample_file):
            with open(sample_file, "w", encoding="utf-8") as f:
                f.write("""# Photosynthesis and Cellular Respiration
Photosynthesis is the biological process used by plants, algae, and certain bacteria to convert light energy into chemical energy stored in glucose.
The general chemical equation for photosynthesis is: 6CO2 + 6H2O + light energy -> C6H12O6 + 6O2.
It occurs in two main stages: the Light-Dependent Reactions (in thylakoid membranes) and the Calvin Cycle (in the stroma).
Chlorophyll a and b are the primary pigments absorbing blue and red light while reflecting green light.
ATP and NADPH produced in the light reactions power the fixation of carbon dioxide into carbohydrates.
Cellular respiration is the reciprocal metabolic pathway where organisms break down glucose with oxygen to produce ATP, releasing CO2 and H2O.""")
        doc_path = sample_file

    if not os.path.exists(doc_path):
        print(f"Error: File not found at '{doc_path}'")
        sys.exit(1)

    print(f"\n📂 Ingesting and indexing document: '{doc_path}'...")
    llm = get_llm()
    full_text = text_to_md(llm, doc_path)
    ingestion = process_document(doc_path)
    vector_db = ingestion["vector_db"]

    if ingestion["topics"]:
        print(f"\n🗂️ Extracted {len(ingestion['topics'])} topics:")
        for t in ingestion["topics"]:
            count = ingestion["topic_chunk_counts"].get(t["topic_id"], 0)
            print(f"  [{t['topic_id']}] {t['name']} ({count} chunks)")

    print("🧠 Deploying Autonomous AI Tutor Agent...")
    agent = create_tutor_agent(vector_db=vector_db, full_text=full_text, doc_name=os.path.basename(doc_path))

    chat_history = []
    print("\n✅ AI Tutor ready! Try asking:")
    print(" - 'Summarize the core concepts'")
    print(" - 'Create 5 flashcards for this material'")
    print(" - 'Give me a 3-question practice quiz'")
    print(" - 'Generate a 3-day study mastery plan'")
    print(" - 'Create an audio summary narration'")
    print(" (Type 'exit' or 'quit' to end session)\n")

    while True:
        try:
            query = input("\n👤 Student: ").strip()
            if not query:
                continue
            if query.lower() in ("exit", "quit", "q"):
                print("\n👋 Happy studying! Goodbye.")
                break

            print("\n🤖 AI Tutor (Reasoning & Calling Tools)...")
            result = run_agent_query(agent, query, chat_history)

            if isinstance(result, dict):
                response_text = result.get("response", "")
                artifacts = result.get("artifacts", {})
                
                print("\n" + "=" * 25 + " TUTOR RESPONSE " + "=" * 25)
                print(response_text)

                if artifacts.get("flashcards"):
                    print(f"\n🗂️ [Agent Generated {len(artifacts['flashcards'])} Flashcards in Memory]")
                if artifacts.get("quiz"):
                    print(f"\n📝 [Agent Generated {len(artifacts['quiz'])}-Question Quiz in Memory]")
                if artifacts.get("study_plan"):
                    print("\n🗓️ [Agent Generated 3-Day Study Mastery Plan]")
                if artifacts.get("audio_file"):
                    print(f"\n🎧 [Agent Synthesized Audio at: {artifacts['audio_file']}]")
            else:
                response_text = str(result)
                print("\n" + "=" * 25 + " TUTOR RESPONSE " + "=" * 25)
                print(response_text)

            chat_history.append(("human", query))
            chat_history.append(("ai", response_text))

        except KeyboardInterrupt:
            print("\nSession interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\nError: {e}")
