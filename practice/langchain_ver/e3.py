from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_agent
from langchain_core.tools import tool


# 演習: ここで自作関数をtoolとして定義しよう
# ヒント: @tool デコレーターを使って、docstringで関数の説明を書く
@tool
def func_bird(input_str: str) -> str:
    # 演習: ここに関数の説明をdocstringで書こう
    """演習: ここに関数の説明を書こう"""
    return ""


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
# 演習: ここでツールのリストを作成しよう
tools = []  # 演習: ここにツールを追加しよう

# 演習: ここでReAct Agentを作成しよう
# ヒント: create_agent(llm, tools) を使う
agent = None

result = agent.invoke(
    # 演習: ここに質問を追加しよう
    {"messages": [("human", "")]}
)
print(result["messages"][-1].content)
