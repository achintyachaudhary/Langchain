from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")


class DetailedReport(BaseModel):
    detailed_report: str


class SummaryResponse(BaseModel):
    summary: str


detailed_op_parser = PydanticOutputParser(pydantic_object=DetailedReport)
summary_op_parser = PydanticOutputParser(pydantic_object=SummaryResponse)

template_detailed_report = PromptTemplate(
    input_variables=["topic"],
    template="Create Detailed report of 10 lines for topic - {topic} \n{op_format}",
    validate_template=True,
    partial_variables={"op_format": detailed_op_parser.get_format_instructions()},
)

template_summary = PromptTemplate(
    input_variables=["detailed_report"],
    template="Create 5 liner summary from {detailed_report}\n{op_format}",
    validate_template=True,
    partial_variables={"op_format": summary_op_parser.get_format_instructions()},
)
chain = (
    template_detailed_report
    | model
    | detailed_op_parser
    | template_summary
    | model
    | summary_op_parser
)
res = chain.invoke({"topic": "Machine Guns"})
print(chain.get_graph().print_ascii())
print(res)
