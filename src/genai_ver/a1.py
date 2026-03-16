from dotenv import load_dotenv

load_dotenv()

from google import genai

client = genai.Client(vertexai=False)

input_text = "私はサッカーを趣味にしています。"
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=f"""入力文から趣味を単語で抽出してください。
入力文: {input_text}
""",
)
print(response.text)
