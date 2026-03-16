from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types

client = genai.Client(vertexai=False)


# セルフ実装
print("Ctrl+Cで終了")
history = []
while True:
    user_input = input("入力してください: ") or "exit"
    if user_input == "exit":
        break
    history.append(types.Content(role="user", parts=[types.Part(text=user_input)]))
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=history,
        config=types.GenerateContentConfig(
            system_instruction="必ず英語で応答してください"
        ),
    )
    history.append(types.Content(role="model", parts=[types.Part(text=response.text)]))
    print(response.text)


# # 簡単な書き方（内部ではセルフ実装と同じように実装している）
# chat = client.chats.create(
#     model="gemini-3.1-flash-lite-preview",
#     config=types.GenerateContentConfig(system_instruction="必ず英語で応答してください"),
# )
# print("Ctrl+Cで終了")
# while True:
#     response = chat.send_message(input("入力してください: ") or "exit")
#     if response.text == "exit":
#         break
#     print(response.text)
