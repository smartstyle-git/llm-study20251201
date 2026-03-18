from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
import base64

with open("data/sample_image.png", "rb") as f:
    image_bytes = f.read()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
# 演習: ここで画像を含むプロンプトを作成しよう
# ヒント: humanメッセージに type: "image_url" を使って画像を含める
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "human",
            [
                # 演習: ここで指示内容と画像をLLMに渡そう
                # {"type": "text", "text": "指示内容"},
                # {"type": "image_url", "image_url": {"url": "data:{file_type};base64,{base64_data}"}},
            ],
        ),
    ]
)
chain = prompt | llm | StrOutputParser()

# 実行時に変数を渡す
result = chain.invoke(
    {
        "file_type": "image/png",
        "base64_data": base64.b64encode(image_bytes).decode("utf-8"),
    }
)
print(result)

