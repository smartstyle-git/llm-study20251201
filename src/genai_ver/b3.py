from dotenv import load_dotenv

load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)


class CategoryFeedback(BaseModel):
    category: Literal["機能", "品質", "価格", "デザイン", "使い勝手"] = Field(
        description="カテゴリ"
    )
    positive_points: str = Field(description="そのカテゴリに関するポジティブな点")
    negative_points: str = Field(description="そのカテゴリに関するネガティブな点")


class CommentAnalysis(BaseModel):
    product_name: str = Field(description="商品名")
    categories: list[CategoryFeedback] = Field(description="カテゴリ別のフィードバック")
    overall_score: int = Field(description="5段階の総合スコア", ge=1, le=5)


input_text = """
エアポッズプロ第2世代を先週購入しました！まず驚いたのがノイズキャンセリング機能で、
通勤の地下鉄で使ってみたら騒音がほぼ完全に消えてびっくり。映画見る時の空間オーディオも
めちゃくちゃ臨場感あって最高です。音質もクリアで低音がしっかり出てる感じがいいですね。
デザインもシンプルで気に入ってて、ケースが小さいからポケットに入れやすいのも便利。
装着感も快適で、一日中つけてても耳が痛くならないし、タッチ操作も直感的で使いやすいです。

ただ正直39,800円は高いなぁって思います。もうちょっと安かったら文句なしなんですが。
あとバッテリー持ちが前のモデルから大して良くなってないみたいで、長時間使う時は
こまめに充電しないといけないのが面倒。雨の日に使ったら音飛びもちょっとあったし、
白だから指紋とか汚れが目立つのも気になります。あとiPhoneとMacで使い分けてるんですが、
ペアリングの切り替えが時々不安定でイライラすることも。

まぁトータルで見れば満足してるんですけど、値段とバッテリーのこと考えると
5点満点で4点かなって感じです。
"""
response = client.models.generate_content(
    model="gemini-3.1-flash-lite-preview",
    contents=f"""**コメント**を分析して、以下の内容を抽出してください。
# 抽出内容
- 商品名
- カテゴリ別のフィードバック（カテゴリ、ポジティブな点、ネガティブな点）
- 総合スコア（5段階）

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
print(comment_result.model_dump_json(indent=2))
