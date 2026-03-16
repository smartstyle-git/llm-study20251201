from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)

# 小説を書くような創造性が必要な場合は温度を高くする
keywords = "冒険、魔法、勇者、魔王"
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=f"""次のキーワードを使って短い小説を書いてください。
キーワード: {keywords}
""",
    config=GenerateContentConfig(temperature=1),
    # config=GenerateContentConfig(temperature=0),
)
print(response.text)


# 翻訳など正確性が重要な場合は温度を低くする
english = "Hello, how are you?"
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=f"""次の英語を日本語に翻訳してください。
{english}
""",
    # config=GenerateContentConfig(temperature=1),
    config=GenerateContentConfig(temperature=0.1),
)
print(response.text)
