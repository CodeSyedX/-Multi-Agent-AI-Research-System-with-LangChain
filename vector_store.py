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
    # 1. Initialize local embedding weights
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )
    
    # 2. Parse out individual website blocks from your tools.py format
    # Blocks are separated by "\n----------------\n"
    blocks = search_results_str.split("\n----------------\n")
    raw_documents = []
    
    for block in blocks:
        if not block.strip():
            continue
        
        # Regex extraction for Title, URL, and Content
        title_match = re.search(r"Title:\s*(.*)", block)
        url_match = re.search(r"URL:\s*(.*)", block)
        snippet_match = re.search(r"Snippet:\s*(.*)", block, re.DOTALL)
        
        title = title_match.group(1).strip() if title_match else "Unknown Source"
        url = url_match.group(1).strip() if url_match else "No Link Available"
        content = snippet_match.group(1).strip() if snippet_match else block
        
        # Construct a LangChain Document with explicit metadata mapping
        doc = Document(
            page_content=f"Source Title: {title}\nFact Content: {content}",
            metadata={"source_url": url, "source_title": title}
        )
        raw_documents.append(doc)

    # 3. Split the documents while explicitly preserving their metadata attachments
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    split_docs = text_splitter.split_documents(raw_documents)
    
    print(f"📦 [RAG Engine] Indexing {len(split_docs)} metadata-tracked chunks into Chroma...")
    
    vector_store = Chroma.from_documents(
        documents=split_docs, 
        embedding=embeddings,
        collection_name="cited_research_index"
    )
    
    return vector_store.as_retriever(search_kwargs={"k": 4})