from app.generators.generate import State
from langchain_core.prompts import PromptTemplate
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
# prompt_template = """Use the following pieces of context to answer the question at the end.
# {context}
# Question: {question}
# Helpful Answer:
# - Provide a clear and structured explanation.
# - If feasible, add a graph or diagram (e.g., ASCII sketch, mermaid diagram, or textual description of a chart) to illustrate the explanation.
# """

prompt_template = """You are a helpful Game Theory tutor. Use the following pieces of context to answer the question at the end.

{context}

Question: {question}

Helpful Answer:
- Explain the concept clearly and step by step.
- If feasible, add a graph, diagram, or illustrative example.
- Only use information provided in the context below.
- Cite sources for each factual statement, including PDF filename and page number.
- Use the format [source_filename, page X].
- Make the explanation student-friendly, as if teaching someone learning the topic for the first time.
"""
prompt = PromptTemplate.from_template(prompt_template)

def build_state(question: str) -> State:
    return {
        "question": question,
        "context": [],
        "answer": "",
        "prompt": prompt,
        "llm": llm
    }
