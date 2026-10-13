from dotenv import load_dotenv

load_dotenv()

from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(
    model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    task="text-generation",
    pipeline_kwargs = dict(
        max_new_tokens=512,
        temperature=0.1
    )
    )

chat_model = ChatHuggingFace(llm=llm)

response = chat_model.invoke("Where is Bengaluru")
print(response)