from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite")
prompt = PromptTemplate.from_template(
    # 演習: ここにinput_text変数を定義して、プロンプトを完成させよう
    # 変数は{}で囲むと定義できます。LangChainではfを付けません
    """""",
)
# 演習: ここでchainを作成しよう
# ヒント: prompt | llm | StrOutputParser() のようにパイプで繋ぐ
chain = None

# 実行時に変数を渡す
result = chain.invoke({
    # 演習: ここに変数input_textに代入する処理を書こう
    "": "私はサッカーを趣味にしています。"
})
print(result)
