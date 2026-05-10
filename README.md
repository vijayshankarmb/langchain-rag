# LangChain PDF RAG

A simple Retrieval-Augmented Generation (RAG) system built with LangChain, Ollama, and ChromaDB. This project demonstrates how to ingest a PDF document, store its embeddings in a local vector database, and query it using a local LLM.

## Architecture

The project consists of two main scripts:

1. **`ingestion.py`**:
   - Loads a PDF file (`data/sample.pdf`) using `PyPDFLoader`.
   - Splits the document into smaller chunks (500 characters with 50 character overlap) using `RecursiveCharacterTextSplitter`.
   - Generates embeddings for these chunks using the local Ollama model `nomic-embed-text`.
   - Stores the embeddings in a local Chroma vector database (`db/chroma_db`).

2. **`query.py`**:
   - Connects to the local Chroma vector database.
   - Takes a user query (e.g., "What is python?") and retrieves the top 3 most relevant document chunks using similarity search.
   - Constructs a prompt containing the retrieved context and the question.
   - Passes the prompt to a local LLM via Ollama (`qwen2.5:3b`) to generate an answer based *only* on the provided context.

## Prerequisites

- **Python**: `>=3.14` (as specified in `pyproject.toml`).
- **Ollama**: You need to have Ollama installed and running locally.
- **Ollama Models**: Pull the required models before running the scripts:
  ```bash
  ollama pull nomic-embed-text
  ollama pull qwen2.5:3b
  ```
- **uv** (Optional but recommended): The project uses `uv` for dependency management (indicated by `uv.lock`).

## Installation

You can install the dependencies using `uv` or `pip`:

Using `uv`:
```bash
uv sync
```

Using `pip`:
```bash
pip install -r requirements.txt
```

## Usage

1. **Prepare the Data**:
   - Create a `data/` directory in the project root if it doesn't exist.
   - Place a PDF file named `sample.pdf` inside the `data/` directory.

2. **Ingest the PDF**:
   Run the ingestion script to process the PDF and populate the vector database:
   ```bash
   python ingestion.py
   ```
   You should see the output: `vector store created`.

3. **Query the Document**:
   Run the query script to ask a question based on the ingested document. You can modify the `query` variable in `query.py` to ask different questions.
   ```bash
   python query.py
   ```

## Dependencies

The core libraries used in this project are:
- `langchain`
- `langchain-community`
- `langchain-ollama`
- `langchain-chroma`
- `chromadb`
- `pypdf`
