# agents/student_writer_agent.py — Zero LLM Tokens
from state import DualReportState

def student_writer_node(state: DualReportState) -> dict:
    print("\n" + "=" * 60 + "\nSTEP 5 — Student Report Writer (Pure Python Assembly)\n" + "=" * 60)
    
    questions = state.get("questions", [])
    answers = state.get("student_answers", [])
    
    # Pure Python string interpolation — 0 LLM Tokens
    report_sections = [f"# Student Synthesis Report: {state['topic']}\n"]
    for idx, (q, a) in enumerate(zip(questions, answers), 1):
        report_sections.append(f"## Section {idx}: {q}\n{a}\n")
        
    student_report = "\n".join(report_sections)
    return {"student_report": student_report}