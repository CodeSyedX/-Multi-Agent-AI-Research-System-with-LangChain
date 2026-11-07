import os
from tavily import TavilyClient
from state import DualReportState

# Initialize Tavily client using environment key
tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

def extractor_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 1 — Extractor Agent (Tavily AI Search & Extraction)\n" + "=" * 60)
    topic = state["topic"]
    
    # Search and extract cleaned text in one call (0 LLM tokens spent)
    search_response = tavily.search(
        query=topic,
        search_depth="advanced",
        max_results=3,
        include_answer=False
    )
    
    # Combine cleaned site content into structured text
    extracted_blocks = []
    for result in search_response.get("results", []):
        title = result.get("title", "Source")
        url = result.get("url", "")
        content = result.get("content", "")
        extracted_blocks.append(f"### {title}\nURL: {url}\n\n{content}")
        
    cleaned_context = "\n\n---\n\n".join(extracted_blocks)
    
    return {
        "raw_source_text": cleaned_context,
        "extracted_data": cleaned_context
    }