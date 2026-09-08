'''import os
import sys

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from core.provider import get_llm
from markitdown import MarkItDown

def text_to_md(client, source: str) -> str:
    """
    Converts a document to Markdown. 
    If an image is found, it uses the groq client to describe it.
    """

    if not source.startswith("http") and not os.path.exists(source):
        raise FileNotFoundError(f"The file/video at {source} was not found.")

    try:
        md = MarkItDown(llm_client=client, llm_model="llama-3.2-11b-vision-preview")
        result = md.convert(source)
        return result.text_content

    except Exception as e:
        return str(e)'''
import os
import sys
import json
import re

script_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(script_dir))

from core.provider import get_llm
from markitdown import MarkItDown


def text_to_md(client, source: str) -> str:
    """
    Converts a document to Markdown.
    If it's already a text or markdown file, it reads it directly.
    Otherwise, uses MarkItDown for conversion.
    """

    if not source.startswith("http") and not os.path.exists(source):
        raise FileNotFoundError(f"The file/video at {source} was not found.")

    # Bypass MarkItDown for plain text/markdown files
    if source.lower().endswith(('.txt', '.md')):
        try:
            with open(source, 'r', encoding='utf-8') as f:
                return f.read()
        except Exception as e:
            return f"Error reading file: {str(e)}"

    try:
        md = MarkItDown(llm_client=client, llm_model="llama-3.2-11b-vision-preview")
        result = md.convert(source)
        return result.text_content

    except Exception as e:
        return str(e)

def safe_json_parse(text: str):
    """
    Safely parses JSON returned from the LLM.
    Handles cases where the model adds extra text
    or markdown formatting.
    """

    try:
        return json.loads(text)

    except json.JSONDecodeError:
        cleaned = re.sub(r"```json|```", "", text).strip()

        try:
            return json.loads(cleaned)
        except:
            return []


def extract_json_from_text(text: str):
    """
    Extract JSON list from LLM response if extra text exists.
    """

    match = re.search(r"\[.*\]", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group())
        except:
            return []

    return []