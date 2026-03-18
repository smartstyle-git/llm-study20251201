from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

@tool
def func_bird(input_str: str) -> str:
    """鳥に関する質問に答える"""
    print("called func_bird")
    return "それは鳥です。"


@tool
def func_add(a: int, b: int) -> int:
    """足し算をする"""
    print("called func_add")
    return a + b


@tool
def func_mul(a: int, b: int) -> int:
    """掛け算をする"""
    print("called func_mul")
    return a * b


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
tools = [func_bird, func_add, func_mul]

agent = create_agent(llm, tools)
result = agent.invoke({"messages": [("human", "ハトについて教えて")]})
print(result["messages"][-1].content)

result = agent.invoke(
    {
        "messages": [
            ("human", "3と4を足した値に1+3を足した値同士を掛け算するとどうなる？")
        ]
    },
    config={"callbacks": [langfuse_handler]}
)
print(result["messages"][-1].content)
