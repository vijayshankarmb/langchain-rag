
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vector_store = Chroma(
    persist_directory="db/chroma_db",
    collection_name="lc_pdf_rag",
    embedding_function=embeddings
)

retriever = vector_store.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 3}
)

query = "What is python?"

results = retriever.invoke(query)

context = "\n\n".join([doc.page_content for doc in results])

prompt = f"""
Answer the question only from the provided context.

Context:
{context}

Question:
{query}

If the answer is not in the context, say "I don't know".
"""

llm = ChatOllama(model="qwen2.5:3b")

response = llm.invoke(prompt)

print(response.content)

