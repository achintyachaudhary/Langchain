# from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
# from langchain_core.prompts import PromptTemplate
# from langchain_groq import ChatGroq
# from dotenv import load_dotenv

# load_dotenv()

# model = ChatGroq(model="qwen/qwen3.8-27b")


# template1 = PromptTemplate(
#     input_variables=["topic"],
#     template="Create a 5 line summary for topic - {topic}",
#     validate_template=True,
# )
# temp1 = template1.invoke({"topic": "Guns in real life"})
# res1 = model.invoke(temp1)

# template2 = PromptTemplate(
#     input_variables=["content"],
#     template="Create a 5 word summary for content - {content}",
#     validate_template=True,
# )

# temp2 = template2.invoke({"content": res1.content})
# res2 = model.invoke(temp2)
# print(res2.content)

from langchain_core.output_parsers import StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")

parser = StrOutputParser()

template1 = PromptTemplate(
    input_variables=["topic"],
    template="Create a 5 line summary for topic - {topic}",
    validate_template=True,
)

template2 = PromptTemplate(
    input_variables=["content"],
    template="Create a 5 word summary for content - {content}",
    validate_template=True,
)

chain = template1 | model | parser | template2 | model | parser
res = chain.invoke({"topic": "guns and safety"})
print(res)
