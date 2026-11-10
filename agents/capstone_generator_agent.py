# agents/capstone_generator_agent.py
from langchain_core.prompts import ChatPromptTemplate
from state import SelfTutorState
from key_rotator import get_llm, safe_parse_json

def capstone_generator_node(state: SelfTutorState) -> dict:
    print("\n" + "="*60 + "\nSTEP 5 COMPLETED — Generating Capstone Project\n" + "="*60)

    # 1. Resolve topic — main_topic is the confirmed state key
    topic = state.get("main_topic", "General Subject")

    # 2. Normalize subtopics defensively
    raw_subtopics = state.get("subtopics") or []

    if isinstance(raw_subtopics, dict):
        # LLM mistakenly returned a wrapped dict
        subtopics = raw_subtopics.get("subtopics", [])
        if not isinstance(subtopics, list):          # Bug 1 fixed: guard non-list value
            subtopics = []
        if not subtopics:
            # Try extracting from first dict value
            first_val = next(iter(raw_subtopics.values()), [])
            subtopics = first_val if isinstance(first_val, list) else []
    elif isinstance(raw_subtopics, list):
        subtopics = raw_subtopics
    else:
        subtopics = []

    # Safely extract titles
    subtopic_titles = [
        s.get("title", "Core Concept")
        for s in subtopics
        if isinstance(s, dict)
    ]

    # 3. Include module scores to personalize capstone context (Bug 3 fix)
    module_scores = state.get("module_scores") or []
    score_summary = ""
    if module_scores:
        score_lines = [
            f"  - Module {s.get('step', i+1)}: {s.get('score', '?')}/{s.get('total', 2)}"
            for i, s in enumerate(module_scores)
            if isinstance(s, dict)
        ]
        score_summary = "\nStudent module scores:\n" + "\n".join(score_lines)

    # 4. Prompt with enriched context
    prompt = ChatPromptTemplate.from_template(
        "The student completed all 5 modules on '{topic}': {modules}.{score_summary}\n\n"
        "Design a practical capstone project integrating these topics.\n"
        "Return ONLY a raw JSON object without conversational text. Format keys exactly as:\n"
        "{{\n"
        "  \"title\": \"string\",\n"
        "  \"objective\": \"string\",\n"
        "  \"deliverables\": [\"string 1\", \"string 2\"],\n"
        "  \"criteria\": \"string\"\n"
        "}}"
    )

    llm = get_llm()
    res = (prompt | llm).invoke({
        "topic": topic,
        "modules": ", ".join(subtopic_titles) if subtopic_titles else "the core curriculum",
        "score_summary": score_summary
    }).content

    # 5. Parse with safe_parse_json and clean fallback (Bug 2 fixed)
    data = safe_parse_json(res, default={})

    if isinstance(data, dict) and "title" in data:
        deliverables_raw = data.get("deliverables", ["Final code repository", "Documentation"])
        # Guard: ensure deliverables is a list of strings
        if not isinstance(deliverables_raw, list):
            deliverables_raw = ["Final code repository", "Documentation"]
        deliverables_formatted = "\n".join([
            f"- {d}" for d in deliverables_raw if isinstance(d, str)
        ])

        project_prompt = (
            f"### 🏆 Capstone Project: {data.get('title', 'Practical Integration Project')}\n\n"
            f"**Objective**: {data.get('objective', 'Apply the concepts learned to a final project.')}\n\n"
            f"**Required Deliverables**:\n{deliverables_formatted}\n\n"
            f"**Evaluation Criteria**: {data.get('criteria', 'Completeness, accuracy, and best practices.')}"
        )
    else:
        # Bug 2 fixed: clean hardcoded fallback instead of raw LLM dump
        print("⚠️ Failed to parse Capstone JSON. Using structured fallback.")
        modules_str = ", ".join(subtopic_titles) if subtopic_titles else "the core curriculum"
        project_prompt = (
            f"### 🏆 Capstone Project: Applied {topic} Integration\n\n"
            f"**Objective**: Build a practical project demonstrating mastery of {topic}.\n\n"
            f"**Required Deliverables**:\n"
            f"- Working implementation covering: {modules_str}\n"
            f"- Written explanation of design decisions\n"
            f"- Test cases or validation results\n\n"
            f"**Evaluation Criteria**: Completeness, accuracy, and adherence to best practices."
        )

    return {"project_prompt": project_prompt}