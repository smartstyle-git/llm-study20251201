from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


# 演習: ここでコメント分析結果のクラスを定義しよう
class CommentAnalysis(BaseModel):
    # 演習: ここに4つのフィールドを定義しよう(str, intなど適切な型を使う)
    pass


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)
# 演習: ここにコメントを分析するプロンプトを書こう
prompt = PromptTemplate.from_template(
    """"""
)
# 演習: ここで構造化出力を使うchainを作成しよう
chain = None

result = chain.invoke(
    {
        "input_text": "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
    }
)
print(result)

