from langgraph.graph import StateGraph, START, END
from state import ResearchState
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.tools.tavily_search import TavilySearchResults # ⚡ Direct standard tool
from vector_store import create_local_vector_db  

# Initialize Gemini 3.5 Flash
llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash", temperature=0.5)

# --- Define Nodes ---

def search_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 1 - Search Agent executing query (Powered by Gemini 3.5 Flash)...")
    print("=" * 50)
    
    topic = state['topic']
    try:
        # Instantiate and run the tool directly using the topic string
        search_tool = TavilySearchResults(max_results=5)
        results = search_tool.invoke(topic)
    except Exception as e:
        print(f"Direct search tool call failed: {e}")
        results = f"Search fallback for topic: {topic}"

    return {"search_results": str(results)}

def reader_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 2 - Reader Agent extracting content & cleaning information...")
    print("=" * 50)
    
    search_results = state["search_results"]
    
    prompt = ChatPromptTemplate.from_template(
        "You are an expert Reader Agent. Analyze these web search results and extract the complete, "
        "detailed factual information to be indexed into our knowledge system.\n\nSearch Results:\n{search_results}"
    )
    
    chain = prompt | llm
    response = chain.invoke({"search_results": search_results[:2000]})
    
    return {"scraped_content": response.content}

def writer_node(state: ResearchState) -> dict:
    print("\n" + "=" * 50)
    print("Step 3 - Writer drafting report using RAG Retrieval...")
    print("=" * 50)
    
    topic = state["topic"]
    
    # RAG Step: Initialize the local vector db using the scraped content
    retriever = create_local_vector_db(state["scraped_content"])
    
    # Query the vector db only for chunks relevant to our topic
    print(f"🔍 [RAG Engine] Fetching top context documents matching topic: '{topic}'")
    relevant_docs = retriever.invoke(topic)
    
    # Combine just the retrieved chunks
    rag_context = "\n\n".join([doc.page_content for doc in relevant_docs])
    
    feedback = state.get("feedback", "")
    if feedback:
        print("🔄 Integrating Critic adjustments into RAG context...")
        rag_context += f"\n\nCRITIC FEEDBACK TO FIX:\n\n{feedback}"
        
    prompt = ChatPromptTemplate.from_template(
        "Write a highly professional and structured research report on the topic '{topic}' in the language: {language}.\n\n"
        "Base your report strictly on this semantic RAG context retrieved from our database:\n{research}\n\n"
        "Ensure clear headings, clean explanations, and an organized format."
    )
    
    chain = prompt | llm
    report = chain.invoke({
        "topic": topic,
        "research": rag_context,
        "language": state["language"]
    })
    
    print("\nDraft Generated successfully.")
    return {"report": report.content}

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

workflow.add_edge(START, "search_agent")
workflow.add_edge("search_agent", "reader_agent")
workflow.add_edge("reader_agent", "writer_agent")
workflow.add_edge("writer_agent", "critic_agent")

workflow.add_conditional_edges(
    "critic_agent",
    route_after_critic,
    {"accept": END, "reject": "writer_agent"}
)

app = workflow.compile()