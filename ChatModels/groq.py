from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


def get_llm_chat_result(query, embeddings, model="qwen/qwen3.8-27b"):
    model = ChatGroq(model="qwen/qwen3.8-27b")
    prompt = {"Company Context": f"{embeddings}", "user": f"{query}"}
    res = model.invoke(str(prompt))
    return res
