from langchain_core.prompts import load_prompt
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
# template = PromptTemplate(
#     name="chat bot template",
#     input_variables=["query"],
#     template="This is chatbot, user asked query: {query}",
# )
# template.save("bot_query.json")
if __name__ == "__main__":
    prompt = load_prompt(
        "/Users/archaudhary/projects/campusx_langchain/chatbot/bot_query.json"
    )
    history = []
    model = ChatGroq(model="qwen/qwen3.8-27b")
    print("Hi, how's your day?. Ask your questions here")
    while True:
        query = str(input("You: "))
        chain = prompt | model
        result = chain.invoke({"query": query, "history": history})
        history.append(query)
        print(f"AI: {result.content}")
