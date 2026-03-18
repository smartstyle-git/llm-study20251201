from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# 演習: ここで温度を設定して違いを観察しよう
# タスクによってどんな温度が適切か、応答速度、創造性、正確性のトレードオフを体験しよう
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=None)
prompt = PromptTemplate.from_template(
    # 演習: ここにinput_text変数を定義して、プロンプトを完成させよう
    """""",
)
chain = prompt | llm | StrOutputParser()

# 実行時に変数を渡す
result = chain.invoke(
    {
        # 演習: ここで変数input_textに代入する値を渡そう
        "": ""
    },
    config={"callbacks": [langfuse_handler]}
)
print(result)
