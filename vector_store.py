from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
# ⚡ NEW LOCAL IMPORT: Runs models locally on your system hardware
from langchain_huggingface import HuggingFaceEmbeddings

def create_local_vector_db(raw_text: str):
    """
    Takes raw scraped text, splits it into dense chunks, 
    embeds them LOCALLY using Hugging Face, and returns an in-memory Chroma retriever.
    """
    # 1. Initialize the Local Embedding Model (Runs entirely on CPU/GPU, zero API quota used!)
    print("🧠 [RAG Engine] Loading local Hugging Face embedding model...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}  # Keeps execution lightweight and Windows-friendly
    )
    
    # 2. Break down the text into small 1,000-character chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    docs = text_splitter.create_documents([raw_text])
    
    # 3. Build the local Chroma vector database inside RAM
    print(f"📦 [RAG Engine] Embedding {len(docs)} text chunks locally into Chroma vector store...")
    
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings,
        collection_name="research_index"
    )
    
    # 4. Return as a retriever that fetches the top 4 most relevant chunks
    return vector_store.as_retriever(search_kwargs={"k": 4})