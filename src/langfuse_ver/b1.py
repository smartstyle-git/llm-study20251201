from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler
from pydantic import BaseModel, Field
from typing import Literal

langfuse_handler = CallbackHandler()


class CommentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="判定結果。positive: ポジティブ、negative: ネガティブ、neutral: ニュートラル"
    )

    # int でも可能
    # sentiment: int = Field(description="-1: ネガティブ, 0: ニュートラル, 1: ポジティブ", ge=-1, le=1)


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)
prompt = PromptTemplate.from_template(
    """次のコメントがポジティブかネガティブかニュートラルか判定してください。
# コメント
```
{input_text}
```
"""
)
chain = prompt | llm.with_structured_output(CommentAnalysis)
result = chain.invoke(
    {
        "input_text": "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
    },
    config={"callbacks": [langfuse_handler]}
)
print(result.sentiment)
