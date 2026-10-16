from langchain_classic.output_parsers import (
    PydanticOutputParser,
    ResponseSchema,
    StructuredOutputParser,
)
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")


class Sch(BaseModel):
    age: int
    desc: str
    location: str
    money: str


sop = PydanticOutputParser(pydantic_object=Sch)
template1 = PromptTemplate(
    input_variables=["topic"],
    template="Create a story on - {topic} \n{op_format}",
    validate_template=True,
    partial_variables={"op_format": sop.get_format_instructions()},
)

chain = template1 | model | sop
res = chain.invoke({"topic": "machine guns"})
print(res)
