from typing import TypedDict, List, Dict, Optional

class DualReportState(TypedDict):
    topic: str
    raw_source_text: str
    extracted_data: str
    ai_report: str
    critic_feedback: str
    critic_approved: bool
    questions: List[str]
    student_answers: List[str]
    student_report: str
    comparative_analysis: str
    scores: Optional[Dict[str, int]]