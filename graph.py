import os
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

from langgraph.graph import StateGraph, START, END
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate

# Custom tools from tools.py
from tools import web_search, scrape_url  
from vector_store import create_local_vector_db  

# Initialize Gemini with valid model identifier
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

# --- Helper Functions ---
def _extract_text_content(content) -> str:
    """Helper to safely extract string text whether LLM returns a str or list of content blocks."""
    if isinstance(content, list):
        extracted = []
        for block in content:
            if isinstance(block, dict):
                extracted.append(block.get("text", ""))
            else:
                extracted.append(str(block))
        return "".join(extracted)
    return str(content)

# --- Define Nodes ---

def search_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent executing query...")
    print("=" * 50)
    
    topic = state['topic']
    try:
        results = web_search.invoke(topic)
    except Exception as e:
        print(f"Custom web search tool call failed: {e}")
        results = f"Search fallback for topic: {topic}"

    return {"search_results": str(results)}

def reader_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent extracting content & cleaning information...")
    print("=" * 50)
    
    search_results = state["search_results"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Reader Agent. Analyze these web search results and extract the single most relevant "
        "and reliable URL link to scrape. Respond with ONLY the raw URL string. Do not include markdown, spaces, or extra text.\n\n"
        "Search Results:\n{search_results}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"search_results": search_results[:2000]})
    target_url = _extract_text_content(response.content).strip()
    print(f"Target URL selected for deep scraping: {target_url}")
    
    try:
        scraped_text = scrape_url.invoke(target_url)
    except Exception as e:
        print(f"Scrape tool call failed: {e}")
        scraped_text = f"Fallback content based on snippets: {search_results}"
    
    return {"scraped_content": scraped_text}

def writer_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 3 - Writer drafting report using Citations & RAG Retrieval...")
    print("=" * 50)
    
    topic = state["topic"]
    
    retriever = create_local_vector_db(state["scraped_content"])
    print(f"Fetching top context documents matching topic: '{topic}'")
    relevant_docs = retriever.invoke(topic)
    
    rag_context_blocks = []
    source_reference_map = {}
    
    for idx, doc in enumerate(relevant_docs, start=1):
        url = doc.metadata.get("source_url", "No Link Available")
        title = doc.metadata.get("source_title", f"Source [{idx}]")
        citation_key = f"[{idx}]"
        
        # Key source_map by citation key for the finalizer node
        source_reference_map[citation_key] = {
            "index": idx,
            "title": title,
            "url": url
        }
        
        rag_context_blocks.append(f"--- [Source Reference ID: {idx}] ---\n{doc.page_content}")
        
    rag_context = "\n\n".join(rag_context_blocks)
    reference_guide = "\n".join([f"ID {item['index']}: {item['title']} ({item['url']})" for item in source_reference_map.values()])
    
    critic_notes = state.get("feedback", "")
    if critic_notes:
        print("Integrating Critic adjustments into RAG context...")
        rag_context += f"\n\nCRITIC REVISION REQS TO FIX:\n{critic_notes}"
        
    prompt = ChatPromptTemplate.from_template(
        "You are a professional research scientist. Draft a comprehensive report on '{topic}' in the language: {language}.\n\n"
        "Guidelines for Citations:\n"
        "1. You MUST use inline bracket citations like [1], [2] to reference facts from the context below.\n"
        "2. Match the bracket ID exactly to the [Source Reference ID: X] provided in the text context block.\n"
        "3. Do not create a bibliography footer or list links at the bottom yourself; your output must only use inline tags.\n\n"
        "Factual Context:\n{research}\n\n"
        "Source Reference Map Context:\n{reference_guide}"
    )
    
    response = (prompt | llm).invoke({
        "topic": topic,
        "research": rag_context,
        "reference_guide": reference_guide,
        "language": state["language"]
    })
    
    report_text = _extract_text_content(response.content)
    
    print("\nDraft Generated successfully.")
    return {"report": report_text, "source_map": source_reference_map}

def critic_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 4 - Critic reviewing draft...")
    print("=" * 50)
    
    prompt = ChatPromptTemplate.from_template(
        "Critically evaluate this research report written in {language}:\n\n{report}\n\n"
        "Check for accuracy, comprehensive depth, and structure. If it is high quality, "
        "reply with exactly the keyword 'APPROVED'. Otherwise, detail what specific issues need to be fixed."
    )
    
    chain = prompt | llm
    response = chain.invoke({
        "report": state["report"],
        "language": state["language"]
    })
    
    feedback_text = _extract_text_content(response.content)
    print("\nCritic Feedback:\n", feedback_text)
    
    current_revisions = state.get("revision_count", 0)
    return {"feedback": feedback_text, "revision_count": current_revisions + 1}

def finalizing_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 5 - Finalizing Agent compiling Citation Reference Table...")
    print("=" * 50)
    
    report_text = state.get("report", "")
    source_map = state.get("source_map", {})
    
    # Extract unique numerical citation tags like [1], [2] from report
    used_markers = sorted(
        list(set(re.findall(r'\[\d+\]', report_text))),
        key=lambda x: int(x.strip("[]"))
    )
    
    reference_footer = "\n\n### References & Source Links\n"
    if used_markers:
        for marker in used_markers:
            if marker in source_map:
                source_info = source_map[marker]
                reference_footer += f"- **{marker}** [{source_info['title']}]({source_info['url']})\n"
    elif source_map:
        for key, info in source_map.items():
            reference_footer += f"- **{key}** [{info['title']}]({info['url']})\n"
    else:
        reference_footer += "_No external references verified._\n"
        
    final_output = report_text + reference_footer
    return {"report": final_output}

# --- Routing Logic ---
def route_after_critic(state: ResearchState) -> str:
    feedback = state["feedback"].strip().upper()
    revisions = state.get("revision_count", 0)
    
    if "APPROVED" in feedback or revisions >= 3:
        return "accept"
    else:
        print(f"\n[System] Critic rejected draft. Routing back to Writer. (Attempt {revisions}/3)")
        return "reject"

# --- Build Workflow ---
workflow = StateGraph(ResearchState)
workflow.add_node("search_agent", search_node)
workflow.add_node("reader_agent", reader_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("critic_agent", critic_node)
workflow.add_node("finalizing_agent", finalizing_node)

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", "reader_agent")
workflow.add_edge("reader_agent", "writer_agent")
workflow.add_edge("writer_agent", "critic_agent")

workflow.add_conditional_edges(
    "critic_agent",
    route_after_critic,
    {
        "accept": "finalizing_agent", 
        "reject": "writer_agent"
    }
)
workflow.add_edge("finalizing_agent", END)

app = workflow.compile()