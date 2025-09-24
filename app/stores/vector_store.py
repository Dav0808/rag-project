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
    pdf_files = [f for f in os.listdir(DOCUMENTS_DIR) if f.endswith(".pdf")]

    # Get names of already loaded documents
    existing_docs = {doc.metadata.get("source") for doc in vector_store.similarity_search("")}

    # Filter out already loaded files
    new_files = [f for f in pdf_files if f not in existing_docs]

    if not new_files:
        print("All documents already loaded. Skipping embedding.")
        return

    all_chunks = []

    for filename in new_files:
        pdf_path = os.path.join(DOCUMENTS_DIR, filename)
        loader = PyPDFLoader(pdf_path)
        docs = loader.load()

        # Assign source metadata
        for doc in docs:
            doc.metadata["source"] = filename

        # Skip pages with very little text
        docs = [doc for doc in docs if len(doc.page_content.strip()) > 50]

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=100)
        chunks = text_splitter.split_documents(docs)
        print(f"[DEBUG] Created {len(chunks)} chunks from {filename}")
        for i, chunk in enumerate(chunks[:3]):  # print first 3 for inspection
            print(f"[DEBUG] Chunk {i} length: {len(chunk.page_content)}")
            print(f"[DEBUG] First 200 chars:\n{chunk.page_content[:200]}")
        all_chunks.extend(chunks)

    if all_chunks:
         batch_size = 100  # tune this if needed
         for i in range(0, len(all_chunks), batch_size):
            batch = all_chunks[i : i + batch_size]
            print(f"[DEBUG] Embedding batch {i//batch_size + 1}, size {len(batch)}")
            vector_store.add_documents(batch)
        # vector_store.add_documents(all_chunks)
    # print(f"Loaded and persisted {len(all_chunks)} chunks.")
    else:
        print("No meaningful text found in new PDFs.")
