# agents/curriculum_planner_agent.py
from langchain_core.prompts import ChatPromptTemplate
from state import SelfTutorState
from key_rotator import get_llm, safe_parse_json

def curriculum_planner_node(state: SelfTutorState) -> dict:
    print("\n" + "="*60 + "\nSTEP 1 — Curriculum Planner (Generating 5 Core Modules)\n" + "="*60)

    # 1. Safely resolve topic key
    topic = state.get("main_topic") or state.get("topic", "General Subject")

    prompt = ChatPromptTemplate.from_template(
        "Divide the main topic '{topic}' into exactly 5 sequential, progressive sub-topics for a concise course.\n"
        "Return ONLY a raw JSON array of 5 objects without markdown text or explanations. Format:\n"
        "[\n"
        "  {{\"step\": 1, \"title\": \"Module Title\", \"focus\": \"Core Focus\"}}\n"
        "]"
    )

    llm = get_llm()
    res = (prompt | llm).invoke({"topic": topic}).content

    # 2. Parse with explicit default=[] so None never reaches normalization
    parsed_data = safe_parse_json(res, default=[])

    try:
        # 3. Normalize: handle dict-wrapped response from LLM
        if isinstance(parsed_data, dict):
            subtopics = parsed_data.get("subtopics", [])
            if not isinstance(subtopics, list) or not subtopics:
                # Bug 1 fixed: guard before assigning first dict value
                first_val = next(iter(parsed_data.values()), [])
                subtopics = first_val if isinstance(first_val, list) else []
        elif isinstance(parsed_data, list):
            subtopics = parsed_data
        else:
            subtopics = []

        # 4. Validate list not empty
        if not subtopics:
            raise ValueError("Extracted subtopics list is empty.")

        # Bug 2 fixed: validate each item has required keys
        validated = []
        for i, item in enumerate(subtopics):
            if not isinstance(item, dict):
                print(f"⚠️ Subtopic item {i} is not a dict, skipping: {item}")
                continue
            validated.append({
                "step": int(item.get("step", i + 1)),
                "title": str(item.get("title", f"Module {i+1}")),
                "focus": str(item.get("focus", f"Fundamentals of {topic}"))
            })

        if len(validated) != 5:
            raise ValueError(f"Expected 5 valid subtopics, got {len(validated)}.")

        subtopics = validated

    except Exception as e:
        print(f"⚠️ Curriculum JSON Parse Error ({e}). Falling back to default structural modules.")
        subtopics = [
            {
                "step": i + 1,
                "title": f"Module {i+1}: Core Principles",
                "focus": f"Fundamentals of {topic}"
            }
            for i in range(5)
        ]

    return {
        "subtopics": subtopics,
        "current_step_index": 0,
        "module_scores": [],
        "student_mcq_answers": {}   # Bug 3 fixed: initialize here so downstream nodes never KeyError
    }