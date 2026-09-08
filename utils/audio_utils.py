import edge_tts
import asyncio
import re

# ✅ CLEAN TEXT FUNCTION
def clean_text(text):
    # remove markdown symbols, borders, extra chars
    text = re.sub(r'\|.*?\|', '', text)        # remove table rows
    text = re.sub(r'[-_=]{2,}', '', text)      # remove lines like ----
    text = re.sub(r'[#$*`]', '', text)         # remove symbols
    text = re.sub(r'\s+', ' ', text)           # remove extra spaces
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