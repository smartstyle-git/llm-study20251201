from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig

client = genai.Client(vertexai=False)

# 演習: ここにinput_text変数に代入する処理を書こう
input_text = ""
response = client.models.generate_content(
    # 演習: ここで思考が使えるモデルを指定してください
    model="",
    # 演習: ここにinput_text変数を埋め込んでプロンプトを完成させよう
    contents=f"""ここにプロンプトを書いてね""",
    config=GenerateContentConfig(
        thinking_config=ThinkingConfig(
            # 演習: ここで思考の上限を設定してください
            thinking_budget=None,
            # 演習: ここで思考過程を含めるかどうかを指定してください
            include_thoughts=None,
        ),
    ),
)

# 思考結果を表示
for part in response.candidates[0].content.parts:
    if part and part.thought:
        print(part.text)
print("-" * 100)
print(response.text)
