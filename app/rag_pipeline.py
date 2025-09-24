
from langgraph.graph import START, StateGraph
from app.generators.generate import generate, State
from app.retrievers.retriever import retrieve
from stores.vector_store import load_documents

load_documents()
graph_builder = StateGraph(State).add_sequence([retrieve, generate])
graph_builder.add_edge(START, "retrieve")
graph = graph_builder.compile()


def invoke_graph(initial_state: State):
    result = graph.invoke(initial_state)
    print(f"Context: {result['context']}\n\n")
    print(f"Answer: {result['answer']}")
    return result['context'], result['answer']
    


