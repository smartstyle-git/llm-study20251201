from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(
    model="gemini-3.1-flash-lite-preview",
    thinking_budget=0,  # 思考の上限を0にする
    # include_thoughts=True, # レスポンスに思考過程を含める（thinking_budgetが0の場合は使えない）
)
prompt = PromptTemplate.from_template(
    """入力文から趣味を単語で抽出してください。
入力文: {input_text}"""
)
chain = prompt | llm

# 実行時に変数を渡す
result = chain.invoke({"input_text": "私はサッカーを趣味にしています。"})
print(result.content)
