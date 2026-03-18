from dotenv import load_dotenv

load_dotenv()

# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langfuse.langchain import CallbackHandler

langfuse_handler = CallbackHandler()

# llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
llm = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview")
prompt = PromptTemplate.from_template(
    """入力文から趣味を単語で抽出してください。
入力文: {input_text}"""
)
chain = prompt | llm | StrOutputParser()

# 実行時に変数を渡す
result = chain.invoke(
    {
        "input_text": "私はサッカーを趣味にしています。"
    },
    config={"callbacks": [langfuse_handler]}
)
print(result)
