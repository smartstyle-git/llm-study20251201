from dotenv import load_dotenv

load_dotenv()

from google import genai

client = genai.Client(vertexai=False)

input_text = "私はサッカーを趣味にしています。"
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    # 演習: ここにinput_text変数を埋め込んでプロンプトを完成させよう
    # python文字列""はfを付けた上で、変数を{}で囲むと埋め込めます
    contents=f"""ここにプロンプトを書いてね""",
)
print(response.text)
