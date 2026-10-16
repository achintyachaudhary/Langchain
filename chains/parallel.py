from langchain_classic.output_parsers import PydanticOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableParallel
from langchain_groq import ChatGroq
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()


class TopicOutputParser(BaseModel):
    summary: str


topic_parser = PydanticOutputParser(pydantic_object=TopicOutputParser)
topic_template = PromptTemplate(
    name="user_topic_template",
    input_variables=["topic"],
    validate_template=True,
    template="I am a teacher who will be giving notes on topic - {topic}, create a 20 lines about this topic\n {output_description}",
    partial_variables={"output_description": topic_parser.get_format_instructions()},
)


class NoteOutputParser(BaseModel):
    notes: str


note_parser = PydanticOutputParser(pydantic_object=NoteOutputParser)
notes_template = PromptTemplate(
    name="notes_template",
    input_variables=["summary"],
    validate_template=True,
    template="Below is the summary of topic i want to teach, can you create 5 liner notes that are important - {summary}\n\n {output_description}",
    partial_variables={"output_description": note_parser.get_format_instructions()},
)


class QuizOutputParser(BaseModel):
    quiz_questions: list[str]


quiz_parser = PydanticOutputParser(pydantic_object=QuizOutputParser)
quiz_template = PromptTemplate(
    name="quiz_template",
    input_variables=["summary"],
    validate_template=True,
    template="Can you create 5 quiz questions from below topic - {summary}\n\n{output_description}",
    partial_variables={"output_description": quiz_parser.get_format_instructions()},
)


class ResultTemplate(BaseModel):
    notes: list[str]
    quizeeees: list[str]


result_parser = PydanticOutputParser(pydantic_object=ResultTemplate)


result_template = PromptTemplate(
    name="compile_template",
    input_variables=["notes", "quiz_questions"],
    validate_template=True,
    template="Below is provided summary and quiz can you merge\n Notes - {notes}\n Quiz - {quiz_questions}\n\n {output_description}",
    partial_variables={"output_description": result_parser.get_format_instructions()},
)
model = ChatGroq(model="qwen/qwen3.8-27b")

parallel_chain = RunnableParallel(
    {
        "notes": notes_template | model | note_parser,
        "quiz": quiz_template | model | quiz_parser,
    }
)

chain = (
    topic_template
    | model
    | topic_parser
    | RunnableLambda(lambda result: {"summary": result.summary})
    | parallel_chain
    | RunnableLambda(
        lambda result: {
            "notes": result["notes"].notes,
            "quiz_questions": result["quiz"].quiz_questions,
        }
    )
    | result_template
    | model
    | result_parser
)

res = chain.invoke({"topic": "Environment Pollution"})
print(res)
