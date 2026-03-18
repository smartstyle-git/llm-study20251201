from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=1)
prompt = PromptTemplate.from_template(
    """次のキーワードを使って短い小説を書いてください。
キーワード: {keywords}"""
)
chain = prompt | llm | StrOutputParser()

# 実行時に変数を渡す
result = chain.invoke({"keywords": "冒険、魔法、勇者、魔王"})
print(result)


llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", temperature=0)
prompt = PromptTemplate.from_template(
    """次の英語を日本語に翻訳してください。
{english}"""
)
chain = prompt | llm | StrOutputParser()

# 実行時に変数を渡す
result = chain.invoke(
    {
        "english": "Hello, how are you?"
    },
    config={"callbacks": [langfuse_handler]}
)

print(result)
