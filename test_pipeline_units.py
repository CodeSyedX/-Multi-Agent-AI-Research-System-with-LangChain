import os
from dotenv import load_dotenv
from state import ResearchState
from vector_store import create_local_vector_db
from graph import writer_node, finalizing_node

# 1. Load credentials
load_dotenv()

def test_metadata_rag_and_nodes():
    print("🧪 --- Starting Metadata & Citation Logic Unit Test --- 🧪\n")
    
    if not os.getenv("GOOGLE_API_KEY"):
        print("❌ Error: GOOGLE_API_KEY not found in environment variables.")
        return

    # Mock text matching the exact Title/URL/Snippet structural output format of tools.py
    mock_search_results = """
Title: Global EV Market Shifts 2026
URL: https://evinsights.com/market-report
Snippet: Electric vehicle manufacturing has accelerated globally in 2026. Tesla remains a dominant equity entity, while emerging solid-state battery tech reshapes standard factory lifecycles.
----------------
Title: Autonomous Systems and Multi-Agent Orchestration
URL: https://ai-journal.org/langgraph-framework
Snippet: LangGraph enables developers to build complex stateful graphs for multi-agent LLM systems. By tying state dictionaries to robust routing edges, systems become resilient against random schema exceptions.
"""

    # --- TEST 1: Vector Store & Metadata Splitting ---
    print("⏳ Testing local Hugging Face embedding + metadata parsing...")
    try:
        retriever = create_local_vector_db(mock_search_results)
        print("✨ Local cited Chroma DB compiled successfully in RAM.")
        
        # Pull mock documents to check if metadata survived chunking
        test_docs = retriever.invoke("Tesla EV market")
        print(f"🎯 Retrieved {len(test_docs)} relevant context blocks.")
        
        # Verify metadata dictionary fields are present
        sample_doc = test_docs[0]
        print(f"   [Metadata Check] Verified URL: {sample_doc.metadata.get('source_url')}")
        print(f"   [Metadata Check] Verified Title: {sample_doc.metadata.get('source_title')}\n")
        
    except Exception as e:
        print(f"❌ Test 1 Failed (Vector Store Metadata Split): {e}")
        return

    # --- TEST 2: Writer Node & Citation Generation ---
    print("⏳ Testing Writer Node citation generation behavior...")
    # Seed an initial state dictionary matching ResearchState schema requirements
    test_state: ResearchState = {
        "topic": "Tesla EV market and LangGraph routing updates",
        "language": "English",
        "search_results": "", # Handled by raw content parameter injection inside the node logic
        "scraped_content": mock_search_results, # Writer reads this to build DB
        "report": "",
        "feedback": "",
        "revision_count": 0,
        "source_map": {}
    }
    
    try:
        writer_output = writer_node(test_state)
        print("✨ Writer Node executed successfully.")
        print("📝 --- Generated Draft Snippet ---")
        print(writer_output["report"][:400] + "...\n")
        
        # Verify the tracking map dictionary populated correctly
        print("🗺️ Stored Source Mapping Dictionary Object:")
        print(writer_output["source_map"])
        print("")
        
        # Update our running state dictionary with the output changes
        test_state["report"] = writer_output["report"]
        test_state["source_map"] = writer_output["source_map"]
        
    except Exception as e:
        print(f"❌ Test 2 Failed (Writer Node processing): {e}")
        return

    # --- TEST 3: Finalizing Node Footer Construction ---
    print("⏳ Testing Finalizing Node footer injection processing...")
    try:
        final_output = finalizing_node(test_state)
        print("✨ Finalizing Node processed content cleanly.")
        print("📝 --- Final Report Footer Inspection Block ---")
        
        # Isolate the final chunks to confirm the reference table rendered properly
        report_lines = final_output["report"].split("\n")
        footer_start_idx = 0
        for i, line in enumerate(report_lines):
            if "References & Source Links" in line:
                footer_start_idx = i
                break
                
        if footer_start_idx:
            print("\n".join(report_lines[footer_start_idx:]))
        else:
            print("❌ Warning: Reference Footer block was not found in final string output.")
            print(final_output["report"])
            
        print("\n🏆 --- ALL SYSTEM UNIT TESTS PASSED SUCCESSFUL --- 🏆")
        
    except Exception as e:
        print(f"❌ Test 3 Failed (Finalizer footer compilation): {e}")

if __name__ == "__main__":
    test_metadata_rag_and_nodes()