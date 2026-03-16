from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from pydantic import BaseModel, Field


class CommentAnalysis(BaseModel):
    product_name: str = Field(description="商品名")
    positive_points: str = Field(description="ポジティブな点")
    negative_points: str = Field(description="ネガティブな点")
    score: int = Field(description="5段階のスコア", ge=1, le=5)


llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.1)
prompt = PromptTemplate.from_template(
    """次のコメントを分析して、商品名、ポジティブな点、ネガティブな点、5段階のスコアを返してください。
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
    }
)
print(result)
