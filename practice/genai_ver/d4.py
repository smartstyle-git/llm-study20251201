from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


# 演習: ここで自作関数をいくつか定義しよう
# ヒント: docstringを書くとLLMがその関数の使い方を理解してくれます
def get_current_temperature(location: str) -> dict[str, str]:
    # 演習: ここにdocstringを書いて関数の説明をしよう
    """演習: ここに関数の説明を書こう（例: 今日の気温を調べる関数）"""
    # MEMO: 実際の実装はここに書く（演習では不要）
    return {"気温": "25℃"}


# 演習: ここで自作関数をツールとして設定しよう
# ヒント: tools のリストに直接関数を渡すことができます
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents="今日の東京の気温と湿度を調べてください",
    config=types.GenerateContentConfig(
        # 演習: ここに自作関数をツールとして追加しよう
        tools=[]  # 演習: ここに関数を追加しよう
    ),
)

print(response.text)

