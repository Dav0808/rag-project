from fastapi import FastAPI
from rag_pipeline import invoke_graph
from pydantic import BaseModel

app = FastAPI()

class Question(BaseModel):
    question: str

@app.get("/")
def root():
    return {"Hello": "Game Theory enthusiast!"}

@app.post("/question")
async def invokegraph(q: Question):
    context, answer = invoke_graph(q.question)
    return {"context": context, "answer": answer}