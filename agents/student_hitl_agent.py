from langgraph.types import interrupt
from state import DualReportState

def student_hitl_node(state: DualReportState) -> dict:
    student_responses = interrupt({
        "type": "student_questionnaire",
        "topic": state["topic"],
        "questions": state["questions"],
        "instruction": "Please answer all 3 questions thoroughly based on your understanding."
    })
    return {"student_answers": student_responses}