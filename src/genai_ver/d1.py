from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)

with open("data/sample_image.png", "rb") as f:
    image_bytes = f.read()

response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=[
        types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
        "画像の内容を説明してください",
    ],
)

print(response.text)
