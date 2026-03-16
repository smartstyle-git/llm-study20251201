from dotenv import load_dotenv

load_dotenv()

from google import genai
from google.genai import types
from pydantic import BaseModel, Field
import asyncio
from pathlib import Path

# C1の評価クラスをインポート
from c1 import evaluate_article, ArticleEvaluationResult

client = genai.Client(vertexai=False)


class ArticleRevision(BaseModel):
    """記事の修正版"""

    revised_article: str = Field(description="修正後の記事（マークダウン形式）")
    changes_summary: list[str] = Field(description="主な変更点のサマリー")


def revise_article(
    article: str, evaluation: ArticleEvaluationResult
) -> ArticleRevision:
    """4つの評価結果を元に記事を修正"""

    # 評価結果をテキスト形式で整形
    technical = evaluation.technical_accuracy
    clarity = evaluation.clarity
    structure = evaluation.structure
    seo = evaluation.seo

    # 演習: ここで評価フィードバックを使って記事を修正するLLM呼び出しを実装しよう
    # system_instruction にフィードバックと修正方針を含める
    # 構造化出力でArticleRevisionクラスを指定する
    response = client.models.generate_content(
        model="gemini-3.1-flash-lite-preview",
        contents=article,
        config=types.GenerateContentConfig(
            # 演習: ここにシステム指示を追加しよう（修正方針とフィードバックを含める）
            system_instruction=f"""ここにプロンプトを書いてね""",
            # 演習: ここで構造化出力の設定を追加しよう
            response_mime_type=None,
            response_schema=None,
            temperature=0.3,
        ),
    )
    return ArticleRevision.model_validate_json(response.text)


async def main():
    # 評価対象の記事を読み込み
    with open("data/bad_quality_article.md") as f:
        article = f.read()

    print("=== 記事の評価を実行中 ===")
    # C1の並列評価を実行
    evaluation_results = await evaluate_article(article)
    print(evaluation_results.model_dump_json(indent=2))

    print("=== 評価結果に基づいて記事を修正中 ===")
    # 評価結果を元に記事を修正
    revision = revise_article(article, evaluation_results)

    print("=== 修正完了 ===")
    print("【主な変更点】")
    for i, change in enumerate(revision.changes_summary, 1):
        print(f"{i}. {change}")

    # 修正後の記事を保存
    result_dir = Path("result")
    result_dir.mkdir(parents=True, exist_ok=True)

    output_path = "result/revised_article.md"
    with open(output_path, "w") as f:
        f.write(revision.revised_article)

    print(f"修正後の記事を {output_path} に保存しました。")


if __name__ == "__main__":
    asyncio.run(main())

