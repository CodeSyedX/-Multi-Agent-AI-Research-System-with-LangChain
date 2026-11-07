import os
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import FastEmbedEmbeddings

def retrieve_top_chunks(raw_text: str, query: str, top_k: int = 4) -> str:
    """Splits raw text into chunks, embeds them locally via FastEmbed (0 API calls), 
    and retrieves top_k relevant snippets using ChromaDB."""
    if not raw_text.strip():
        return raw_text

    # 1. Chunk document into ~500 character segments with 50 character overlap
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    docs = text_splitter.create_documents([raw_text])

    # 2. Local ONNX embeddings via FastEmbed (0 Gemini API calls)
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")

    # 3. Embed & store chunks in an in-memory Chroma vector database
    vector_db = Chroma.from_documents(documents=docs, embedding=embeddings)

    # 4. Perform similarity search for top-k matching chunks
    matching_docs = vector_db.similarity_search(query, k=top_k)

    # 5. Concatenate matching chunks into clean context
    retrieved_context = "\n\n---\n\n".join([doc.page_content for doc in matching_docs])

    return retrieved_context