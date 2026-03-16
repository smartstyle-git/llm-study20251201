from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)

# 演習: ここにinput_text変数に代入する処理を書こう
input_text = ""
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    # 演習: ここにinput_text変数を埋め込んでプロンプトを完成させよう
    contents=f"""ここにプロンプトを書いてね""",
    # 演習: ここで温度を設定して違いを観察しよう
    # タスクによってどんな温度が適切か、応答速度、創造性、正確性のトレードオフを体験しよう
    config=GenerateContentConfig(temperature=None),
)
print(response.text)
