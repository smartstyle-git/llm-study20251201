from dotenv import load_dotenv

load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)


# 演習: コメントの感情（ポジティブ/ネガティブ/ニュートラル）を分類するためのクラスを定義しよう
# ヒント: Literal["aaa", "bbb", "ccc"] のようにLiteral[]を使うと選択肢を制限できます
class CommentAnalysis(BaseModel):
    # 演習: 「ここに...書いてね」の部分を埋めて完成させよう
    sentiment: Literal["ここに選択肢1を書いてね", "ここに選択肢2を書いてね"] = Field(
        description="ここにこのフィールドの説明を書いてね（この説明はLLMに渡されます）"
    )

    # int でも可能
    # sentiment: int = Field(description="-1: ネガティブ, 0: ニュートラル, 1: ポジティブ", ge=-1, le=1)


input_text = "スマート加湿器を購入。静音性は期待通り。給水が面倒なのがマイナス。5点満点中3点といったところ。"
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    # 演習: ここにコメントの感情を判定するプロンプトを書こう
    contents=f"""ここにプロンプトを書いてね""",
    config=GenerateContentConfig(
        # 演習: response_schema に上で定義した CommentAnalysis クラスを指定して構造化出力を完成させよう
        response_mime_type="application/json",
        response_schema=None,
        temperature=0.1,
    ),
)
comment_result = CommentAnalysis.model_validate_json(response.text)
print(comment_result)
