import re
from state import DualReportState

def critic_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 3 — Critic Agent (Python Validation Guard)\n" + "=" * 60)
    
    raw_report = state.get("ai_report", "")
    
    # Safely convert list content blocks or objects into a plain string
    if isinstance(raw_report, list):
        report = " ".join([
            str(item.get("text", item) if isinstance(item, dict) else item) 
            for item in raw_report
        ])
    else:
        report = str(raw_report)
    
    # Deterministic structural checks
    word_count = len(report.split())
    has_overview = bool(re.search(r"overview|introduction|newton|summary", report, re.IGNORECASE))
    has_tradeoffs = bool(re.search(r"trade-off|limitation|challenge|law|impact|mechanism", report, re.IGNORECASE))
    
    if word_count > 100 and (has_overview or has_tradeoffs):
        print("-> Status: APPROVED")
        return {"critic_feedback": "APPROVED", "critic_approved": True}
    else:
        issues = []
        if word_count <= 100: issues.append("Report is too short.")
        if not (has_overview or has_tradeoffs): issues.append("Missing core sections.")
        print(f"-> Status: REJECTED ({', '.join(issues)})")
        return {"critic_feedback": f"STATUS: REJECTED\nIssues: {', '.join(issues)}", "critic_approved": False}