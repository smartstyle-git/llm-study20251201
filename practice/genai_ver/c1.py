from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai.types import GenerateContentConfig
from pydantic import BaseModel, Field
import asyncio

client = genai.Client(vertexai=False)


class Evaluation(BaseModel):
    """評価結果"""

    # 演習: ここに評価のためのフィールドを定義しよう（bool, list[str]など適切な型を使う）
    pass


class ArticleEvaluationResult(BaseModel):
    """記事の総合評価結果"""

    technical_accuracy: Evaluation = Field(description="技術的正確性の評価")
    clarity: Evaluation = Field(description="わかりやすさの評価")
    structure: Evaluation = Field(description="構成・論理展開の評価")
    seo: Evaluation = Field(description="SEO最適化の評価")


async def evaluate_aspect(system_prompt: str, article: str) -> Evaluation:
    """特定の観点で記事を評価"""
    # 演習: ここで非同期でLLMを呼び出して評価を取得しよう
    # 構造化出力でEvaluationクラスのJSON形式で返すように設定しよう
    response = await client.aio.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=article,
        config=GenerateContentConfig(
            system_instruction=system_prompt,
            # 演習: ここで構造化出力の設定を追加しよう
            response_mime_type=None,
            response_schema=None,
            temperature=0.1,
        ),
    )
    return Evaluation.model_validate_json(response.text)


# 各評価軸のプロンプト
# 演習: ここに4つの評価軸のプロンプトを定義しよう
TECHNICAL_ACCURACY_PROMPT = """
ここにプロンプトを書いてね
"""
CLARITY_PROMPT = """
ここにプロンプトを書いてね
"""
STRUCTURE_PROMPT = """
ここにプロンプトを書いてね
"""
SEO_PROMPT = """
ここにプロンプトを書いてね
"""


async def evaluate_article(article: str) -> ArticleEvaluationResult:
    """記事を4つの評価軸で並列評価"""
    # 演習: ここで4つの評価を並列で実行しよう
    # asyncio.gather を使うと複数の非同期処理を並列実行できます
    technical, clarity, structure, seo = await asyncio.gather(
        # 演習: ここに4つの評価関数呼び出しを記述しよう
        # evaluate_aspect(...),
    )

    return ArticleEvaluationResult(
        technical_accuracy=technical,
        clarity=clarity,
        structure=structure,
        seo=seo,
    )


if __name__ == "__main__":
    # 評価対象の記事を読み込み
    with open("data/bad_quality_article.md") as f:
        article = f.read()

    # 並列評価を実行
    results = asyncio.run(evaluate_article(article))

    # 評価結果を表示
    print(results.model_dump_json(indent=2))
