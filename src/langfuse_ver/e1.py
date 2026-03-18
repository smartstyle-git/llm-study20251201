from dotenv import load_dotenv

load_dotenv()


from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_core.runnables import RunnableLambda
from langfuse.langchain import CallbackHandler
import numpy as np

langfuse_handler = CallbackHandler()

target_texts = ["漫画", "アニメ"]
model = GoogleGenerativeAIEmbeddings(
    # model="text-multilingual-embedding-002",
    model="gemini-embedding-001",
    # dimensions=768,
    task_type="semantic_similarity",
)

embedding_runnable = RunnableLambda(model.embed_documents)

results = model.embedding_runnable(
    target_texts,
    config={"callbacks": [langfuse_handler]}
)

embedding1 = np.array(results[0])
embedding2 = np.array(results[1])

print(f"{target_texts[0]}: {embedding1[:5]}...({len(embedding1)}dimensions)")
print(f"{target_texts[1]}: {embedding2[:5]}...({len(embedding2)}dimensions)")

normed_embedding1 = embedding1 / np.linalg.norm(embedding1)
normed_embedding2 = embedding2 / np.linalg.norm(embedding2)
print("cosine similarity: ", np.dot(normed_embedding1, normed_embedding2))
