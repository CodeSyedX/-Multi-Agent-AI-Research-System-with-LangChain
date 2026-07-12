from typing import TypedDict

class ResearchState(TypedDict):
    topic: str
    language: str
    search_results: str
    scraped_content: str
    report: str
    feedback: str
    revision_count: int