from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel
from pydantic import BaseModel, Field

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0.1)


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


def create_evaluation_chain(system_prompt: str):
    """プロンプトから評価チェーンを作成"""
    # 演習: ここでChatPromptTemplateを作成しよう
    # ヒント: system メッセージと human メッセージを含めて、構造化出力を使う
    prompt = ChatPromptTemplate.from_messages(
        [
            # 演習: ここにメッセージを追加しよう
            # ("system", "システムプロンプト"),
            # ("human", "{article}"),
        ]
    )
    # 演習: ここで構造化出力を使うchainを作成しよう
    return None


# 各評価軸のプロンプト
# 演習: ここに4つの評価軸のプロンプトを定義しよう
TECHNICAL_ACCURACY_PROMPT = """
"""
CLARITY_PROMPT = """
"""
STRUCTURE_PROMPT = """
"""
SEO_PROMPT = """
"""


def evaluate_article(article: str) -> ArticleEvaluationResult:
    """記事を4つの評価軸で並列評価"""
    # 演習: ここでRunnableParallelを使って並列実行しよう
    # ヒント: 辞書形式で各評価チェーンを定義
    parallel_chain = RunnableParallel(
        {
            # 演習: ここに4つの評価チェーンを追加しよう
            # "...": create_evaluation_chain(...),
        }
    )

    results = parallel_chain.invoke({"article": article})

    return ArticleEvaluationResult.model_validate(results)


if __name__ == "__main__":
    # 評価対象の記事を読み込み
    with open("data/bad_quality_article.md") as f:
        article = f.read()

    # 並列評価を実行
    results = evaluate_article(article)

    # 評価結果を表示
    print(results.model_dump_json(indent=2))
