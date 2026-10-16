from langchain_classic.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

model = ChatGroq(model="qwen/qwen3.8-27b")


schema = [
    ResponseSchema(name="age", description="age of the fictional character"),
    ResponseSchema(name="location", description="location of the fictional character"),
    ResponseSchema(
        name="net wealth", description="net wealth of the fictional character"
    ),
]
sop = StructuredOutputParser.from_response_schemas(schema)
template1 = PromptTemplate(
    input_variables=["topic"],
    template="Create a story on - {topic} \n{op_format}",
    validate_template=True,
    partial_variables={"op_format": sop.get_format_instructions()},
)

chain = template1 | model | sop
res = chain.invoke({"topic": "machine guns"})
print(res)
