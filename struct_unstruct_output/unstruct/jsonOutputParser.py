from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")

json_op_format = JsonOutputParser()

template1 = PromptTemplate(
    input_variables=["topic"],
    template="Create a 1 line summary for topic - {topic} \n{op_format}",
    validate_template=True,
    partial_variables={"op_format": json_op_format.get_format_instructions()},
)

chain = template1 | model | json_op_format
res = chain.invoke({"topic": "Machine Guns"})
print(res)
