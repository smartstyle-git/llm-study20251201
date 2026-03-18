from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "必ず英語で応答してください"),
        MessagesPlaceholder(variable_name="history"),
    ]
)
chain = prompt | llm | StrOutputParser()


print("Ctrl+Cで終了")
history = []
while True:
    user_input = input("入力してください: ") or "exit"
    if user_input == "exit":
        break
    history.append(HumanMessage(content=user_input))
    response = chain.invoke(
        {
            "history": history
        },
        config={"callbacks": [langfuse_handler]}
    )
    history.append(AIMessage(content=response))
    print(response)
