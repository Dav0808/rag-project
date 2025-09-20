from generators.generate import State
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
prompt_template = """Use the following pieces of context to answer the question at the end.
{context}
Question: {question}
Helpful Answer:"""
prompt = PromptTemplate.from_template(prompt_template)

def build_state(question: str) -> State:
    return {
        "question": question,
        "context": [],
        "answer": "",
        "prompt": prompt,
        "llm": llm
    }
