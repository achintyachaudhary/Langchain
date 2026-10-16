from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()
if __name__ == "__main__":
    history = [
        SystemMessage(
            content="You are Senior python Enginner, Punnet Superstart. who only answer queries related to python and nothing else!"
        )
    ]

    model = ChatGroq(model="qwen/qwen3.8-27b")
    print("Hi, how's your day?. Ask your questions here")
    ctr = 0
    while True:
        query = str(input("You: "))
        hm = HumanMessage(content=query)
        history.append(hm)
        result = model.invoke(history)
        history.append(AIMessage(content=result.content))
        print(f"AI: {result.content}")
        ctr += 1
        if ctr == 5:
            break
    print(history)
