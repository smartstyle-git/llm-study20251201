from dotenv import load_dotenv

load_dotenv()

from google import genai
import numpy as np
from google.genai import types

client = genai.Client(vertexai=False)


target_texts = ["漫画", "アニメ"]
result = client.models.embed_content(
    # model="gemini-embedding-001",
    model="text-multilingual-embedding-002",
    contents=target_texts,
    config=types.EmbedContentConfig(
        task_type="SEMANTIC_SIMILARITY",
        # output_dimensionality=768
    ),
)

embedding1 = np.array(result.embeddings[0].values)
embedding2 = np.array(result.embeddings[1].values)

print(f"{target_texts[0]}: {embedding1[:5]}...({len(embedding1)}dimensions)")
print(f"{target_texts[1]}: {embedding2[:5]}...({len(embedding2)}dimensions)")

normed_embedding1 = embedding1 / np.linalg.norm(embedding1)
normed_embedding2 = embedding2 / np.linalg.norm(embedding2)
print("cosine similarity: ", np.dot(normed_embedding1, normed_embedding2))
