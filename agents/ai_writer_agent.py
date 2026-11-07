
import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import DualReportState
from vector_store import retrieve_top_chunks

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
    max_retries=5
)

def ai_writer_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 2 — AI Writer Agent (Local FastEmbed RAG Retrieval)\n" + "=" * 60)
    
    raw_context = state.get("extracted_data", "")
    topic = state["topic"]
    
    top_4_context = retrieve_top_chunks(raw_text=raw_context, query=topic, top_k=4)
    
    feedback = state.get("critic_feedback", "").strip()
    if feedback and "APPROVED" not in feedback.upper():
        top_4_context += f"\n\nCRITIC REVISION DIRECTIVES:\n{feedback}"

    prompt = ChatPromptTemplate.from_template(
        "Draft a formal academic research report on '{topic}' using ONLY this retrieved evidence context:\n\n{context}"
    )
    
    res = (prompt | llm).invoke({"topic": topic, "context": top_4_context}).content
    
    # Ensure report string is clean
    if isinstance(res, list):
        report = " ".join([str(item.get("text", item) if isinstance(item, dict) else item) for item in res])
    else:
        report = str(res)
    
    return {
        "ai_report": report,
        "extracted_data": top_4_context
    }