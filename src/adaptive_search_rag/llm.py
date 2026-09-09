import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI


def get_llm() -> ChatOpenAI:
    load_dotenv()
    return ChatOpenAI(
        model_name=os.getenv("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        openai_api_key=os.getenv("DEEPSEEK_API_KEY"),
        openai_api_base=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        temperature=0,
        extra_body={"thinking": {"type": "disabled"}},
    )
