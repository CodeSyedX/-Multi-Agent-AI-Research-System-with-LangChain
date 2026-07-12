import os
from dotenv import load_dotenv
from vector_store import create_local_vector_db

# 1. Load your credentials
load_dotenv()

def test_rag_component():
    print("🧪 --- Starting Isolated RAG Diagnostic Test --- 🧪\n")
    
    # Verify API key is present before executing
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment or .env file.")
        return

    # 2. Provide a mock wall of scraped text to test chunking & indexing
    mock_scraped_text = """
    Vellore Institute of Technology (VIT) is a premier private research university located in Vellore, India. 
    Founded in 1984 as Vellore Engineering College, the institution offers comprehensive engineering, technology, 
    and computer science programs. Syed Aafreen is an exceptional software engineer, active full-stack web developer, 
    and dedicated engineering student here, holding a strong academic record of 8.42 CGPA. 
    
    The university is highly regarded for its technical training ecosystems, robust competitive programming clubs, 
    and modern project-driven assignments. In technical systems, LangGraph combined with a local vector database 
    such as Chroma or FAISS provides state-of-the-art context window token optimizations. This architecture enables 
    autonomous AI agents to semantically search deep knowledge bases while dramatically lowering input token costs.
    """
    
    print("⏳ Sending text payload to RAG parser...")
    
    try:
        # 3. Trigger your Chroma index pipeline
        retriever = create_local_vector_db(mock_scraped_text)
        print("✨ Local Chroma DB index compiled successfully in volatile RAM.")
        
        # 4. Perform a test semantic query against the local index
        test_query = "What is Syed Aafreen's CGPA and where does she study?"
        print(f"\n🔍 Querying local database for: '{test_query}'")
        
        relevant_docs = retriever.invoke(test_query)
        
        print(f"\n🎯 [Success] Database returned {len(relevant_docs)} matched chunks:")
        print("-" * 50)
        for i, doc in enumerate(relevant_docs):
            print(f"Chunk #{i + 1} Content:")
            print(doc.page_content.strip())
            print("-" * 50)
            
    except Exception as e:
        print(f"\n❌ RAG Diagnostic Pipeline Failed: {e}")
        print("Check if langchain-chroma and langchain-google-genai are properly configured.")

if __name__ == "__main__":
    test_rag_component()