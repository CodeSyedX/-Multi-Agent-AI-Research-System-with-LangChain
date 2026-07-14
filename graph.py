from langgraph.graph import StateGraph, START, END
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
# Correctly import your actual custom tools from tools.py
from tools import web_search, scrape_url  
from vector_store import create_local_vector_db  

# Initialize Gemini 3.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.2)

# --- Define Nodes ---

def search_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent executing query (Powered by Gemini 3.5 Flash)...")
    print("=" * 50)
    
    topic = state['topic']
    try:
        # Call your custom web search tool from tools.py using .invoke()
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
    
    # Let Gemini 3.5 Flash parse the web search text and pick the absolute best URL to target
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Reader Agent. Analyze these web search results and extract the single most relevant "
        "and reliable URL link to scrape. Respond with ONLY the raw URL string. Do not include markdown, spaces, or extra text.\n\n"
        "Search Results:\n{search_results}"
    )
    
    chain = prompt | llm
    target_url = chain.invoke({"search_results": search_results[:2000]}).content.strip()
    print(f"🎯 Target URL selected for deep scraping: {target_url}")
    
    try:
        # Pass the extracted URL straight to your custom BeautifulSoup tool from tools.py
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
    
    # Initialize the updated metadata-aware vector database using your custom scraped text
    retriever = create_local_vector_db(state["scraped_content"])
    print(f"🔍 [RAG Engine] Fetching top context documents matching topic: '{topic}'")
    relevant_docs = retriever.invoke(topic)
    
    # Map each retrieved document chunk to a source index reference map
    rag_context_blocks = []
    source_reference_map = {}
    
    for idx, doc in enumerate(relevant_docs, start=1):
        url = doc.metadata.get("source_url", "No Link Available")
        title = doc.metadata.get("source_title", "Source Reference")
        
        # Track unique links to avoid duplicating reference tables
        source_reference_map[url] = {"index": idx, "title": title}
        
        # Format the context block with a visible reference tag for the LLM
        rag_context_blocks.append(f"--- [Source Reference ID: {idx}] ---\n{doc.page_content}")
        
    rag_context = "\n\n".join(rag_context_blocks)
    
    # Format the Source Reference Guide to teach the model how to link back
    reference_guide = "\n".join([f"ID {item['index']}: {title} ({url})" for url, item in source_reference_map.items()])
    
    critic_notes = state.get("feedback", "")  # Uses key 'feedback' from your state
    if critic_notes:
        print("🔄 Integrating Critic adjustments into RAG context...")
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
    
    report = (prompt | llm).invoke({
        "topic": topic,
        "research": rag_context,
        "reference_guide": reference_guide,
        "language": state["language"]
    })
    
    print("\nDraft Generated successfully.")
    # Store the unique URL mapping dictionary inside our node response 
    # so the Finalizer Node can read it later to print the reference table!
    return {"report": report.content, "source_map": source_reference_map}

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
    feedback = chain.invoke({
        "report": state["report"],
        "language": state["language"]
    })
    
    print("\nCritic Feedback:\n", feedback.content)
    
    current_revisions = state.get("revision_count", 0)
    return {"feedback": feedback.content, "revision_count": current_revisions + 1}

def finalizing_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 5 - Finalizing Agent compiling Citation Reference Table...")
    print("=" * 50)
    
    report_text = state["report"]
    source_map = state.get("source_map", {})
    
    # Generate a professional Markdown reference list using the stored map data
    reference_footer = "\n\n### 📖 References & Source Links\n"
    if source_map:
        # Sort sources by their reference index sequence
        sorted_sources = sorted(source_map.items(), key=lambda x: x[1]["index"])
        for url, info in sorted_sources:
            reference_footer += f"- **[{info['index']}]** [{info['title']}]({url})\n"
    else:
        reference_footer += "_No external references verified._\n"
        
    # Append the link index to the bottom of the final report text
    final_output = report_text + reference_footer
    return {"report": final_output}

# --- Routing Logic ---
def route_after_critic(state: ResearchState) -> str:
    feedback = state["feedback"].strip().upper()
    revisions = state.get("revision_count", 0)
    
    if "APPROVED" in feedback or revisions >= 3:
        return "accept"
    else:
        print(f"\n🔄 [System] Critic rejected draft. Routing back to Writer. (Attempt {revisions}/3)")
        return "reject"

# --- Build Workflow ---
workflow = StateGraph(ResearchState)
workflow.add_node("search_agent", search_node)
workflow.add_node("reader_agent", reader_node)
workflow.add_node("writer_agent", writer_node)
workflow.add_node("critic_agent", critic_node)
workflow.add_node("finalizing_agent", finalizing_node)  # Added Finalizer Node

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", "reader_agent")
workflow.add_edge("reader_agent", "writer_agent")
workflow.add_edge("writer_agent", "critic_agent")

workflow.add_conditional_edges(
    "critic_agent",
    route_after_critic,
    {"accept": "finalizing_agent", "reject": "writer_agent"}  # Routes through Finalizing Node on approval
)
workflow.add_edge("finalizing_agent", END)

app = workflow.compile()