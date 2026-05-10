from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

loader = PyPDFLoader(file_path="data/sample.pdf")
docs = loader.load()

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="db/chroma_db",
    collection_name="lc_pdf_rag",
)

print("vector store created")



