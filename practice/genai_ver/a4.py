from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


# 演習: historyリストに会話履歴を蓄積して連続的な対話を実現しよう
# ヒント: types.Content(role="user", parts=[...]) と types.Content(role="model", parts=[...]) を使う
print("Ctrl+Cで終了")
history = []
while True:
    user_input = input("入力してください: ") or "exit"
    if user_input == "exit":
        break
    # 演習: ここでユーザーの入力をhistoryに追加しよう
    # history.append(...)

    response = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        # 演習: ここでhistoryを渡して、これまでの会話履歴を全てLLMに送ろう
        contents=None,
        config=types.GenerateContentConfig(
            # 適当な指示に変えてOK
            system_instruction="必ず英語で応答してください"
        ),
    )
    
    # 演習: ここでLLMの応答をhistoryに追加しよう
    # history.append(...)
    print(response.text)
