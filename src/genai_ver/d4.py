from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


# 関数を定義
def get_current_temperature(location: str) -> dict[str, str]:
    """今日の気温を調べる関数"""
    # MEMO: 実装はここに書く
    return {"気温": "25℃"}


def get_current_humidity(location: str) -> dict[str, str]:
    """今日の湿度を調べる関数"""
    # MEMO: 実装はここに書く
    return {"湿度": "50%"}


response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents="今日の東京の気温と湿度を調べてください",
    config=types.GenerateContentConfig(
        tools=[get_current_temperature, get_current_humidity]
    ),
)

print(response.text)
