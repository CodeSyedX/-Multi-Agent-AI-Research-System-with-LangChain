
import re
from langchain_core.prompts import ChatPromptTemplate
from state import SelfTutorState
from key_rotator import get_llm, safe_parse_json

def project_evaluator_node(state: SelfTutorState) -> dict:
    print("\n" + "="*60 + "\nFINAL STEP — Project Evaluation & Mastery Certification\n" + "="*60)

    # 1. Safely resolve state keys
    topic = state.get("main_topic") or state.get("topic", "General Subject")
    project_prompt = state.get("project_prompt") or "Standard Capstone Assignment"
    submission = state.get("student_project_submission") or "No submission provided."

    # Warn if submission is missing — evaluating placeholder is misleading
    if submission == "No submission provided.":
        print("⚠️ Warning: No student submission found in state. Evaluating placeholder.")

    # Bug 5 fixed: include module scores in evaluation context
    module_scores = state.get("module_scores") or []
    score_summary = ""
    if module_scores:
        score_lines = [
            f"  - Module {s.get('step', i+1)}: {s.get('score', '?')}% "
            f"({s.get('correct', '?')}/{s.get('total', '?')} correct)"
            for i, s in enumerate(module_scores)
            if isinstance(s, dict)
        ]
        score_summary = "\nStudent module performance:\n" + "\n".join(score_lines) + "\n"

    # 2. Prompt with enriched context
    prompt = ChatPromptTemplate.from_template(
        "Evaluate this student capstone project submission for '{topic}':\n\n"
        "ASSIGNMENT:\n{project_prompt}\n\n"
        "SUBMISSION:\n{submission}\n"
        "{score_summary}\n"
        "Return ONLY a raw JSON object without conversational text or code block wrapping. Format exactly as:\n"
        "{{\n"
        "  \"score\": 85,\n"
        "  \"verdict\": \"Mastery Certified\",\n"
        "  \"summary\": \"string\",\n"
        "  \"strengths\": [\"string 1\", \"string 2\"],\n"
        "  \"improvements\": [\"string 1\", \"string 2\"]\n"
        "}}"
    )

    llm = get_llm()
    res = (prompt | llm).invoke({
        "topic": topic,
        "project_prompt": project_prompt,
        "submission": submission,
        "score_summary": score_summary
    }).content

    # Bug 1 & 7 fixed: safe_parse_json never raises; use default={} and check without raising
    data = safe_parse_json(res, default={})

    if isinstance(data, dict) and "score" in data:
        # Bug 2 fixed: cast score to int safely
        try:
            score = int(str(data.get("score", 85)).split("/")[0].strip())
        except (ValueError, TypeError):
            score = 85

        # Bug 2 fixed: clamp score to 0-100
        score = max(0, min(100, score))

        # verdict as clean string
        verdict = str(data.get("verdict", "Evaluated")).strip()   # Bug 3 partial fix

        summary = str(data.get("summary", "Project meets baseline requirements.")).strip()

        # Bug 3 fixed: validate strengths and improvements as lists
        raw_strengths = data.get("strengths", [])
        if not isinstance(raw_strengths, list):
            raw_strengths = [str(raw_strengths)] if raw_strengths else ["Solid foundational implementation."]
        strengths = "\n".join([f"- {str(s)}" for s in raw_strengths if s])

        raw_improvements = data.get("improvements", [])
        if not isinstance(raw_improvements, list):
            raw_improvements = [str(raw_improvements)] if raw_improvements else ["Consider reviewing advanced edge cases."]
        improvements = "\n".join([f"- {str(i)}" for i in raw_improvements if i])

        feedback = (
            f"### 📊 Capstone Evaluation Report\n\n"
            f"**Overall Score**: {score}/100\n"
            f"**Certification Verdict**: {verdict}\n\n"
            f"**Executive Summary**: {summary}\n\n"
            f"#### Key Strengths\n{strengths}\n\n"
            f"#### Recommendations for Enhancement\n{improvements}"
        )

    else:
        # Bug 4 fixed: strip markdown fences from raw fallback
        print("⚠️ Failed to parse Evaluation JSON. Falling back to cleaned raw response.")
        cleaned_res = re.sub(r"```(?:json)?|```", "", str(res)).strip()
        feedback = (
            f"### 📊 Capstone Evaluation Report\n\n"
            f"**Overall Score**: Pending/100\n"
            f"**Certification Verdict**: Evaluated\n\n"
            f"**Executive Summary**: "
            f"(Structured report unavailable. Raw feedback below.)\n\n"
            f"{cleaned_res}"
        )

    return {
        "project_feedback": feedback,
        "is_mastery_completed": True,
        "critic_approved": False        # Bug 6 fixed: reset critic flag cleanly at END
    }