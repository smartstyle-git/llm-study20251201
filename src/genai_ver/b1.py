from dotenv import load_dotenv

load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)


class CommentAnalysis(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="判定結果。positive: ポジティブ、negative: ネガティブ、neutral: ニュートラル"
    )

    # int でも可能
    # sentiment: int = Field(description="-1: ネガティブ, 0: ニュートラル, 1: ポジティブ", ge=-1, le=1)


input_text = "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
response = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=f"""次のコメントがポジティブかネガティブかニュートラルか判定してください。
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
