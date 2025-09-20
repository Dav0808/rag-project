from fastapi import FastAPI
from generators.generate import State
from helpers.helper_functions import build_state
from rag_pipeline import invoke_graph
from pydantic import BaseModel
from stores.vector_store import load_documents, vector_store
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model

app = FastAPI(
    title="Game Theory RAG API",
    description="A RAG (Retrieval Augmented Generation) API for Game Theory questions",
    version="1.0.0"
)

class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {"Hello": "Game Theory enthusiast!"}

@app.post("/question")
async def ask_question(request: QuestionRequest):
    state = build_state(request.question)
    context, answer = invoke_graph(state)
    return {"context": context, "answer": answer}