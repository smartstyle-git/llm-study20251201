from dotenv import load_dotenv

load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)


class CommentAnalysis(BaseModel):
    product_name: str = Field(description="商品名")
    positive_points: str = Field(description="ポジティブな点")
    negative_points: str = Field(description="ネガティブな点")
    score: int = Field(description="5段階のスコア", ge=1, le=5)


input_text = "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=f"""次のコメントを分析して、商品名、ポジティブな点、ネガティブな点、5段階のスコアを返してください。
# コメント
```
{input_text}
```
""",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        response_schema=CommentAnalysis,
        temperature=0.1,
    ),
)
comment_result = CommentAnalysis.model_validate_json(response.text)
print(comment_result)
