from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from pathlib import Path

# C1の評価クラスとevaluate_articleをインポート
from c1 import evaluate_article, ArticleEvaluationResult

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.3)


class ArticleRevision(BaseModel):
    """記事の修正版"""

    revised_article: str = Field(description="修正後の記事（マークダウン形式）")
    changes_summary: list[str] = Field(description="主な変更点のサマリー")


# 演習: ここで修正用のプロンプトを作成しよう
# ヒント: system メッセージに修正方針とフィードバックを含める
revision_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            # 演習: ここにfeedbackを変数に含めたシステム指示を追加しよう
            """""",
        ),
        # 演習: ここにarticleを受け取るhumanメッセージを追加しよう
        # ("human", "{article}"),
    ]
)

# 演習: ここで構造化出力を使うchainを作成しよう
revision_chain = None


def revise_article(
    article: str, evaluation: ArticleEvaluationResult
) -> ArticleRevision:
    """4つの評価結果を元に記事を修正"""

    # 評価結果をテキスト形式で整形
    technical = evaluation.technical_accuracy
    clarity = evaluation.clarity
    structure = evaluation.structure
    seo = evaluation.seo

    feedback = f""""""

    return revision_chain.invoke({"article": article, "feedback": feedback})


def main():
    # 評価対象の記事を読み込み
    with open("data/bad_quality_article.md") as f:
        article = f.read()

    print("=== 記事の評価を実行中 ===")
    # C1の並列評価を実行
    evaluation_results = evaluate_article(article)
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
    main()
