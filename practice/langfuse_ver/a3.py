from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

llm = ChatGoogleGenerativeAI(
    # 演習: ここで思考が使えるモデルを指定してください
    model="",
    # 演習: ここで思考の上限を設定してください
    thinking_budget=None,
    # 演習: ここで思考過程を含めるかどうかを指定してください
    include_thoughts=None,
)
prompt = PromptTemplate.from_template(
    # 演習: ここにinput_text変数を埋め込んでプロンプトを完成させよう
    """""",
)
chain = prompt | llm

# 実行時に変数を渡す
result = chain.invoke(
    {
        # 演習: ここに変数input_textに代入する処理を書こう
        "": ""
    },
    config={"callbacks": [langfuse_handler]}
)
print(result.content)
