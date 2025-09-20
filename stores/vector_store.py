import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.document_loaders.merge import MergedDataLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()


embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="game_theory",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

DOCUMENTS_DIR = "./documents"

def load_documents():
    """Load new documents into Chroma, skipping already loaded ones."""
    
    # Get list of PDF files
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

    # Get list of already loaded document names from metadata (if stored)
    existing_docs = {doc.metadata.get("source") for doc in vector_store.similarity_search("")}

    # Filter out already loaded files
    new_files = [f for f in pdf_files if f not in existing_docs]

    if not new_files:
        print("All documents already loaded. Skipping embedding.")
        return

    # Create loaders for new files
    loaders = [PyPDFLoader(os.path.join(DOCUMENTS_DIR, f)) for f in new_files]
    merged_loader = MergedDataLoader(loaders=loaders)
    docs = merged_loader.load()
    
    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
    all_chunks = []

    for doc, filename in zip(docs, new_files):
        doc.metadata["source"] = filename

    # Split the document into manageable chunks
    chunks = text_splitter.split_documents([doc])
    all_chunks.extend(chunks)

    vector_store.add_documents(all_chunks)
    print(f"Loaded and persisted {len(all_chunks)} chunks.")
