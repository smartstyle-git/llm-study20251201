from dotenv import load_dotenv

load_dotenv()

from google import genai
import numpy as np
from google.genai import types

client = genai.Client(vertexai=False)


target_texts = ["漫画", "アニメ"]
# 演習: ここでEmbedding APIを使って2つのテキストをベクトル化しよう
result = client.models.embed_content(
    # 演習: ここでモデル名を指定しよう（text-multilingual-embedding-002 など）
    model="",
    # 演習: ここでベクトル化したいテキストのリストを渡そう
    contents=None,
    config=types.EmbedContentConfig(
        # 演習: ここでタスクタイプを指定しよう
        # task_type は "SEMANTIC_SIMILARITY" を指定
        task_type="",
    ),
)

embedding1 = np.array(result.embeddings[0].values)
embedding2 = np.array(result.embeddings[1].values)

print(f"{target_texts[0]}: {embedding1[:5]}...({len(embedding1)}dimensions)")
print(f"{target_texts[1]}: {embedding2[:5]}...({len(embedding2)}dimensions)")

# 演習: ここでコサイン類似度を計算しよう
similarity = None
print("cosine similarity: ", similarity)
