from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np


target_texts = ["漫画", "アニメ"]
# 演習: ここでEmbeddingモデルを作成しよう
model = GoogleGenerativeAIEmbeddings(
    # 演習: ここでモデル名を指定しよう（gemini-embedding-001 など）
    model="",
    # 演習: ここでタスクタイプを指定しよう
    # task_type は "semantic_similarity" を指定
    task_type="",
)
# 演習: ここでテキストをベクトル化しよう
results = model.embed_documents(
    target_texts,
)


embedding1 = np.array(results[0])
embedding2 = np.array(results[1])

print(f"{target_texts[0]}: {embedding1[:5]}...({len(embedding1)}dimensions)")
print(f"{target_texts[1]}: {embedding2[:5]}...({len(embedding2)}dimensions)")

# 演習: ここでコサイン類似度を計算しよう
similarity = None
print("cosine similarity: ", similarity)
