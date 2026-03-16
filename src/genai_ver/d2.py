from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents="東京の今日の天気を調べてください",
    config=types.GenerateContentConfig(
        tools=[types.Tool(google_search=types.GoogleSearch())]
    ),
)

print(response.text)
