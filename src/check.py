# 環境変数の確認
import os
from dotenv import load_dotenv

load_dotenv()
# GOOGLE_CLOUD_PROJECT = os.getenv("GOOGLE_CLOUD_PROJECT")
# GOOGLE_CLOUD_LOCATION = os.getenv("GOOGLE_CLOUD_LOCATION")
GOOGLE_APPLICATION_CREDENTIALS = os.getenv("GOOGLE_APPLICATION_CREDENTIALS")

# print(f"{GOOGLE_CLOUD_PROJECT=}")
# print(f"{GOOGLE_CLOUD_LOCATION=}")
print(f"{GOOGLE_APPLICATION_CREDENTIALS=}")

# GenAI SDKでAPIを叩く

from google import genai

client = genai.Client(vertexai=False)
# client = genai.Client(vertexai=False, project=GOOGLE_CLOUD_PROJECT, location=GOOGLE_CLOUD_LOCATION) # 直接指定も可能

response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview", contents="LLMについて1行で教えてください"
)
print(response.text)
