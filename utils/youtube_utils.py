from tools.youtube_tool import get_youtube_transcript

def extract_video_id(url):
    from tools.youtube_tool import extract_video_id as ev
    return ev(url)

def get_transcript(url):
    return get_youtube_transcript(url)
