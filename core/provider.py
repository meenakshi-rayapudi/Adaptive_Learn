from . import config
from langchain_groq import ChatGroq
import os

def get_llm(model_name: str = None, temperature: float = None, api_key: str = None):
    """
    Returns an initialized ChatGroq LLM instance.
    """
    key = api_key or config.GROQ_API_KEY or os.getenv("GROQ_API_KEY", "")
    model = model_name or config.MODEL_NAME
    temp = temperature if temperature is not None else config.TEMPERATURE
    
    return ChatGroq(
        model=model,
        temperature=temp,
        max_retries=config.MAX_RETRIES,
        api_key=key if key else None
    )
