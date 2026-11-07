# state.py
from typing import TypedDict, List, Dict, Any

class ResearchState(TypedDict):
    topic: str
    language: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    revision_count: int
    source_map: dict
    
    # Phase 2 Fields
    flagged_contradictions: List[str]     # Questions surfaced by Contradiction Detector
    student_synthesis: str                # Human student input/claim
    challenge_history: List[Dict[str, str]] # Dialogue between Challenger Agent and Student
    challenge_round: int                  # Track challenge rounds (max 3)