from typing import List, TypedDict, Annotated
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.language_models.base import BaseLanguageModel
from langgraph.graph.message import add_messages

class State(TypedDict):
    question: str
    context: List[Document] 
    answer: str
    prompt: PromptTemplate
    llm: BaseLanguageModel
             


def generate(state:State) -> dict:
    if not state["context"]:
        docs_content = "No relevant context found."
    else:
        docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    formatted_prompt = state["prompt"].format(
    question=state["question"],
    context=docs_content
)
    response = state["llm"].invoke(formatted_prompt)
    return {"answer": response.content}