from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field
from typing import Literal

langfuse_handler = CallbackHandler()


# 演習: ここでコメントの感情を分類するクラスを定義しよう
# ヒント: Literal["aaa", "bbb", "ccc"] のようにLiteral[]を使うと選択肢を制限できます
class CommentAnalysis(BaseModel):
    # 演習: 「ここに...書いてね」の部分を埋めて完成させよう
    sentiment: Literal["ここに選択肢1を書いてね", "ここに選択肢2を書いてね"] = Field(
        description="ここにこのフィールドの説明を書いてね（この説明はLLMに渡されます）"
    )

    # int でも可能
    # sentiment: int = Field(description="-1: ネガティブ, 0: ニュートラル, 1: ポジティブ", ge=-1, le=1)


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)
# 演習: ここにコメントの感情を判定するプロンプトを書こう
prompt = PromptTemplate.from_template(
    """"""
)
# 演習: ここで構造化出力を使うchainを作成しよう
# ヒント: llm.with_structured_output(クラス名) を使う
chain = None

result = chain.invoke(
    {
        "input_text": "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
    },
    config={"callbacks": [langfuse_handler]}
)
print(result.sentiment)

