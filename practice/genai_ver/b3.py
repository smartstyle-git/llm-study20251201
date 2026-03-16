from dotenv import load_dotenv

load_dotenv()

from google import genai
from pydantic import BaseModel, Field
from typing import Literal
from google.genai.types import GenerateContentConfig

client = genai.Client(vertexai=False)


# 演習: ここでカテゴリ別のフィードバックを表すクラスを定義しよう
class CategoryFeedback(BaseModel):
    # 演習: ここにカテゴリとポジティブ/ネガティブな点のフィールドを定義しよう（Literal[], strなど適切な型を使う）
    pass


class CommentAnalysis(BaseModel):
    # 演習: ここに商品名、カテゴリ別フィードバックのリスト、総合スコアを定義しよう（list[], str, intなど適切な型を使う）
    pass


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
    model="gemini-2.5-flash-lite",
    # 演習: ここにコメントを分析して各カテゴリごとにフィードバックを抽出するプロンプトを書こう
    contents=f"""ここにプロンプトを書いてね""",
    config=GenerateContentConfig(
        response_mime_type="application/json",
        # 演習: ここで構造化出力のスキーマを指定しよう
        response_schema=None,
        temperature=0.1,
    ),
)
comment_result = CommentAnalysis.model_validate_json(response.text)
print(comment_result.model_dump_json(indent=2))
