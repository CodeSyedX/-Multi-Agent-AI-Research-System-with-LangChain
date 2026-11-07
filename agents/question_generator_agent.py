import os
from dotenv import load_dotenv

load_dotenv()

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from state import DualReportState

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
    max_retries=5
)

def question_generator_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 4 — Question Generator Agent\n" + "=" * 60)
    
    # Safely convert state ai_report to string
    raw_ai_report = state.get("ai_report", "")
    if isinstance(raw_ai_report, list):
        ai_report_str = " ".join([str(item.get("text", item) if isinstance(item, dict) else item) for item in raw_ai_report])
    else:
        ai_report_str = str(raw_ai_report)

    prompt = ChatPromptTemplate.from_template(
        "Based on this report for '{topic}', generate 3 open-ended synthesis questions for a student:\n{report}\n\n"
        "Output strictly 3 numbered questions."
    )

    res = (prompt | llm).invoke({"topic": state["topic"], "report": ai_report_str[:2000]}).content

    # Safely convert LLM output content to string before stripping/splitting
    if isinstance(res, list):
        raw_q = " ".join([str(item.get("text", item) if isinstance(item, dict) else item) for item in res]).strip()
    else:
        raw_q = str(res).strip()

    q_list = [q.strip() for q in raw_q.split("\n") if q.strip() and q[0].isdigit()][:3]
    return {"questions": q_list}