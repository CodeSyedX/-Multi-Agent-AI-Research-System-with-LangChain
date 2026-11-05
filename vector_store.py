import re
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def create_local_vector_db(search_results_str: str):
    """
    Parses structural search results, extracts text + source metadata,
    embeds them locally, and returns a metadata-aware Chroma retriever.
    """
    # 1. Initialize local embedding model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Parse individual search result blocks from tools.py output
    blocks = search_results_str.split("\n----------------\n")
    raw_documents = []
    
    for idx, block in enumerate(blocks, start=1):
        if not block.strip():
            continue
        
        # Regex extraction for Title, URL, and Content
        title_match = re.search(r"Title:\s*(.*)", block)
        url_match = re.search(r"URL:\s*(.*)", block)
        snippet_match = re.search(r"Snippet:\s*(.*)", block, re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else f"Source [{idx}]"
        url = url_match.group(1).strip() if url_match else "No Link Available"
        
        if snippet_match:
            content = snippet_match.group(1).strip()
        else:
            # Clean fallback if Snippet tag is missing
            content = block.replace(title, "").replace(url, "").strip()
        
        doc = Document(
            page_content=f"Source Title: {title}\nFact Content: {content}",
            metadata={"source_url": url, "source_title": title, "citation_id": f"[{idx}]"}
        )
        raw_documents.append(doc)

    if not raw_documents:
        # Fallback empty document if input text parsing fails completely
        raw_documents.append(Document(page_content=search_results_str, metadata={"source_url": "", "source_title": "Raw Data"}))

    # 3. Split documents into chunks while retaining metadata
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = text_splitter.split_documents(raw_documents)
    
    print(f"📦 [RAG Engine] Indexing {len(split_docs)} metadata-tracked chunks into Chroma...")
    
    # Instantiate in-memory Chroma instance
    vector_store = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 4})