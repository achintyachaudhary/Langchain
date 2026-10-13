from dotenv import load_dotenv

from langchain_groq import ChatGroq

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b", temperature=0, max_tokens=8)
result = model.invoke("What is the date today?")
print(result)