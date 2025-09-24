RAG Project: Game Theory Tutor


A Retrieval-Augmented Generation (RAG) pipeline for Game Theory tutoring. The system leverages LangChain, Chroma, and OpenAI embeddings to provide precise, context-aware answers from Game Theory books.

🚀 Features

Index PDFs and retrieve relevant context dynamically

Generate embeddings with OpenAI text-embedding-3-large

RAG pipeline delivers accurate, context-aware answers

Load new documents without reindexing entire dataset

Extensible for multi-agent interactions and adaptive tutoring

🛠 Tech Stack

Python 3.13

LangChain, Chroma, OpenAI Embeddings

FastAPI + Uvicorn for API

Docker for containerization

Azure App Service for deployment

⚡ Quick Start
Clone & Install
git clone https://github.com/Dav0808/rag-project.git
cd rag-project
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
pip install -r requirements.txt


Set environment variables in .env:
LANGSMITH_TRACING=true
OPENAI_API_KEY=<your_openai_api_key>
LANGSMITH_API_KEY=<your_openai_api_key>

Load Documents

Place PDFs in ./documents and run:

python load_documents.py

Run API Locally
uvicorn main:app --host 0.0.0.0 --port 3100 --reload


Access at: http://localhost:3100

🐳 Docker Deployment

Build and run locally:

docker build -t rag-project:latest .
docker run -p 3100:3100 rag-project:latest


For Azure App Service:

# Build and tag
docker build -t <ACR_LOGIN>/rag-project:latest .
docker push <ACR_LOGIN>/rag-project:latest

# Configure App Service
az webapp config container set \
  --name <APP_SERVICE_NAME> \
  --resource-group <RESOURCE_GROUP> \
  --docker-custom-image-name <ACR_LOGIN>/rag-project:latest

# Restart to pull new image
az webapp restart --name <APP_SERVICE_NAME> --resource-group <RESOURCE_GROUP>


Replace <ACR_LOGIN>, <APP_SERVICE_NAME>, <RESOURCE_GROUP> with your Azure details.

🔮 Future Improvements

Persistent vector store on cloud (Azure File Share, Chroma Cloud, or Pinecone)

Adaptive difficulty based on user performance

Multi-agent tutoring simulations

Integration with additional LLMs

🤝 Contributing

Contributions welcome! Follow PEP 8 style and document new features. Open issues or PRs freely.

📄 License

MIT License – see LICENSE
