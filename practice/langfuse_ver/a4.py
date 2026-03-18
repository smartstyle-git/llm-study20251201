from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
# 演習: ここでChatPromptTemplateを定義しよう
# ヒント: system メッセージと MessagesPlaceholder を使って履歴を管理
prompt = ChatPromptTemplate.from_messages(
    [
        # 演習: ここにsystemメッセージを追加しよう
        # ("system", "..."),
        # 演習: ここにMessagesPlaceholderを追加しよう（variable_name="history"）
        # MessagesPlaceholder(variable_name="history"),
    ]
)
chain = prompt | llm | StrOutputParser()


print("Ctrl+Cで終了")
history = []
while True:
    user_input = input("入力してください: ") or "exit"
    if user_input == "exit":
        break
    # 演習: ここでユーザーの入力をhistoryに追加しよう
    # ヒント: HumanMessage(content=user_input) を使う
    # history.append(...)
    
    response = chain.invoke(
        {
            "history": history
        },
        config={"callbacks": [langfuse_handler]}
    )
    
    # 演習: ここでLLMの応答をhistoryに追加しよう
    # ヒント: AIMessage(content=response) を使う
    # history.append(...)
    
    print(response)

