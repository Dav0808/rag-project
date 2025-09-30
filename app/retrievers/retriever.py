import shutil
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from stores.vector_store import vector_store, inspect_vector_store 

def retrieve(state: dict):
    """
    Retrieve relevant documents for a query.
    state: {"question": "your query"}
    """
    retrieved_docs = vector_store.similarity_search(state["question"])
    # inspect_vector_store()
    return {"context": retrieved_docs}
