from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


# 演習: ここでGoogle検索を使えるようにツールを設定しよう
# ヒント: config の tools に types.Tool(google_search=types.GoogleSearch()) を追加
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents="東京の今日の天気を調べてください",
    # 演習: ここでGoogle検索ツールを設定しよう
    config=types.GenerateContentConfig(
        tools=[]  # 演習: ここにツールを追加しよう
    ),
)

print(response.text)

