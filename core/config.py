from dotenv import load_dotenv
import os

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
MODEL_NAME = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Token budget controls. Groq's free tier caps requests at ~8,000 tokens/minute,
# and the agent resends history + tool results on every step, so these keep a
# single request well under that. Raise them in .env on a paid tier.
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "4"))
MAX_HISTORY_MESSAGES = int(os.getenv("MAX_HISTORY_MESSAGES", "6"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1200"))
MAX_TOOL_OUTPUT_CHARS = int(os.getenv("MAX_TOOL_OUTPUT_CHARS", "2500"))


