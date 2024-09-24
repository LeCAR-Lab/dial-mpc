from pathlib import Path
from openai import OpenAI
client = OpenAI()

speech_file_path = Path(__file__).parent / "voice/idea.mp3"
response = client.audio.speech.create(
  model="tts-1-hd",
  voice="nova",
  input=
    """
    Our work built on top of the theoretical landscape analysis of MPPI and the connection between MPPI and single-step diffusion.
    """
)

response.stream_to_file(speech_file_path)