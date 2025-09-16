from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from dotenv import load_dotenv
import os
from langchain_core.prompts import PromptTemplate
from langchain_core.documents import Document
from typing_extensions import List, TypedDict
from langgraph.graph import START, StateGraph
from IPython.display import Image, display


load_dotenv()

llm = init_chat_model("gpt-4o-mini", model_provider="openai")
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

file_path = "./documents/OsborneRubinsteinMasterpiece.pdf"
loader = PyPDFLoader(file_path)

# file_path1 = "./documents/osborne.pdf"
# loader1 = PyPDFLoader(file_path1)
loader_all = MergedDataLoader(loaders=[loader])
docs = loader_all.load()
vector_store.add_documents(docs)

# print(len(docs))
# characters = 0
# max_char = 0
# for i in range(len(docs)):
#     char_length = len(docs[i].page_content)
#     characters += char_length
#     if char_length > max_char:
#         max_char = char_length
    
# print(f"Total characters: {characters}")
# print(f"Max characters per document: {max_char}")

template = """Use the following pieces of context to answer the question at the end.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
Be as formal as possible and also give an intuition and an example to illustrate the answer. Also draw a graph when describing an example.
Always say "thanks for asking!" at the end of the answer.

{context}

Question: {question}

Helpful Answer:"""
prompt = PromptTemplate.from_template(template)

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}


def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()
# display(Image(graph.get_graph().draw_mermaid_png()))

# result = graph.invoke({"question": "What is a strategy in an extensive form game?"})

def invoke_graph(question:str):
    result = graph.invoke({"question": question})
    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
    return result['context'], result['answer']
    

# print(f"Context: {result['context']}\n\n")
# print(f"Answer: {result['answer']}")
