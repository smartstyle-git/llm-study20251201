from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

client = genai.Client(vertexai=False)

input_text = "私はサッカーを趣味にしています。"
response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=f"""入力文から趣味を単語で抽出してください。
入力文: {input_text}
""",
    config=GenerateContentConfig(
        thinking_config=ThinkingConfig(
            thinking_budget=0,  # 思考の上限を0にする
            # include_thoughts=True, # レスポンスに思考過程を含める（thinking_budgetが0の場合は使えない）
        ),
    ),
)

# 思考結果を表示
for part in response.candidates[0].content.parts:
    if part and part.thought:
        print(part.text)
print("-" * 100)
print(response.text)
