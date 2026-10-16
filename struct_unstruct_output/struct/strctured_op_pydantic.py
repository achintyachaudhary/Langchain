from typing import Annotated, Literal, TypedDict

from langchain_core.prompts import FewShotPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field
from dotenv import load_dotenv

load_dotenv()


class ChatModelResponse(BaseModel):
    category: str = Field(
        description="this is category the issue belongs to, make sure just have domain name for it in all Caps"
    )
    satisfied: Annotated[Literal["Yessss", "Noooooo"], "Is user statified?"]


model = ChatGroq(model="qwen/qwen3.8-27b")
model = model.with_structured_output(ChatModelResponse)
res = model.invoke(
    "Facing issue with running my iphone, it causes heating issue and doesn't charge properly, not happy with the product"
)

print(res)
