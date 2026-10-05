import edge_tts
import asyncio
import re

# ✅ CLEAN TEXT FUNCTION
def clean_text(text):
    # remove markdown symbols, borders, extra chars
    text = text or ""
    text = text.replace("|", " ")
    text = re.sub(r'\[.*?\]\(.*?\)', ' ', text)
    text = re.sub(r'#{1,6}\s*', '', text)
    text = re.sub(r'[-_=]{2,}', ' ', text)
    text = re.sub(r'[*`_~]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


async def text_to_speech(text, file_path="output.mp3"):
    cleaned_text = clean_text(text)

    communicate = edge_tts.Communicate(
        cleaned_text,
        voice="en-US-AriaNeural"
    )
    await communicate.save(file_path)


def generate_audio(text, output_file="summary.mp3"):
    asyncio.run(text_to_speech(text, output_file))
    return output_file