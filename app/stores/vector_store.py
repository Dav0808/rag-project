import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = Chroma(
    collection_name="game_theory",
    embedding_function=embeddings,
    persist_directory="./chroma_langchain_db",
)

DOCUMENTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "documents")


def load_documents():
    """Load new documents into Chroma, skipping already loaded ones."""
    
    # List all PDFs
    pdf_files = {f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")}
    print('pdf files:', pdf_files)
    
    data = vector_store._collection.get(include=["metadatas"])
    all_sources = {m.get("source") for m in data["metadatas"]}
    print("All sources in DB:", all_sources)
    if pdf_files == all_sources:
        return
    
    new_files = list(pdf_files - all_sources)

    all_chunks = []

    for filename in new_files:
        pdf_path = os.path.join(DOCUMENTS_DIR, filename)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Assign source metadata
        for i, doc in enumerate(docs, start=1):
            doc.metadata["source"] = filename
            doc.metadata["page"] = i

        # Skip pages with very little text
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 800]

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=4000, chunk_overlap=200)
        chunks = text_splitter.split_documents(docs)
        all_chunks.extend(chunks)

    if all_chunks:
         batch_size = 100
         for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            print(f"[DEBUG] Embedding batch {i//batch_size + 1}, size {len(batch)}")
            vector_store.add_documents(batch)
        # vector_store.add_documents(all_chunks)
    # print(f"Loaded and persisted {len(all_chunks)} chunks.")
    else:
        print("No meaningful text found in new PDFs.")

# Check sample documents in your vector store
def inspect_vector_store():
    """See what's actually in your vector store"""
    # Get a larger sample
    sample_docs = vector_store.similarity_search("", k=10)
    
    for i, doc in enumerate(sample_docs):
        print(f"\n--- Document {i} ---")
        print(f"Source: {doc.metadata.get('source')}")
        print(f"Page: {doc.metadata.get('page', 'N/A')}")
        print(f"Content length: {len(doc.page_content)}")
        print(f"Content preview: {doc.page_content}...")
        # print(f"Content end: ...{doc.page_content[-200:]}")
