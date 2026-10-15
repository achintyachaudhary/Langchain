from langchain_core.prompts import PromptTemplate


def get_researcher_prompt(prompt, context):
    template = PromptTemplate(
        input_variables=["prompt", "context"],
        template="User Have uploaded some docs, they could be asking question, mentioned below"
        ". Make sure you answer it, if it is some general question answer that too, but let them know this is Answer not related to docs you uploaded"
        ". User Prompt - {prompt}"
        " Knowledge base related to prompt - {context}",
        validate_template=True,
    )
    return template.format(prompt=prompt, context=context)
