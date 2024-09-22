from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "results/detail.mp3"
response = client.audio.speech.create(
  model="tts-1-hd",
  voice="nova",
  input=
    """
    For further details, please check our website and papers. Thanks!
    """
)

response.stream_to_file(speech_file_path)