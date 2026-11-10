# agents/module_hitl_agent.py
from langgraph.types import interrupt
from state import SelfTutorState

def module_quiz_hitl_node(state: SelfTutorState) -> dict:
    idx = state.get("current_step_index", 0)

    # 1. Safely resolve subtopics
    raw_subtopics = state.get("subtopics") or []
    if isinstance(raw_subtopics, dict):
        subtopics = raw_subtopics.get("subtopics", [])
        if not isinstance(subtopics, list) or not subtopics:
            first_val = next(iter(raw_subtopics.values()), [])
            subtopics = first_val if isinstance(first_val, list) else []
    elif isinstance(raw_subtopics, list):
        subtopics = raw_subtopics
    else:
        subtopics = []

    if not subtopics or idx >= len(subtopics):
        current_title = f"Module {idx + 1}: Core Concepts"
    else:
        current_subtopic = subtopics[idx]
        current_title = (
            current_subtopic.get("title", f"Module {idx + 1}")
            if isinstance(current_subtopic, dict)
            else str(current_subtopic)
        )

    # 2. Safely resolve module data
    module_data = state.get("current_module_data") or {}
    mcqs = module_data.get("mcqs", [])

    # Strip correct_index and explanation from interrupt payload
    # so answers are never exposed to the frontend
    safe_mcqs = []
    for q in mcqs:
        if not isinstance(q, dict):
            continue
        safe_mcqs.append({
            "id":       str(q.get("id", "")),
            "question": q.get("question", ""),
            "options":  q.get("options", [])
            # correct_index and explanation intentionally excluded
        })

    # 3. Fire interrupt with sanitized payload
    user_answers = interrupt({
        "type":          "module_quiz",
        "step":          idx + 1,
        "title":         current_title,
        "explanation":   module_data.get("explanation", "Explanation unavailable."),
        "youtube_links": module_data.get("youtube_links", []),
        "mcqs":          safe_mcqs
    })

    # 4. Score MCQs — handle both option-text and index-string answer formats
    correct    = 0
    total_mcqs = len(mcqs)

    for q in mcqs:
        if not isinstance(q, dict):
            continue

        q_id_str = str(q.get("id", ""))
        selected = user_answers.get(q_id_str) if isinstance(user_answers, dict) else None

        options = q.get("options", [])
        if not isinstance(options, list):
            options = []

        # Cast correct_index to int safely
        try:
            correct_idx = int(q.get("correct_index", 0))
        except (ValueError, TypeError):
            correct_idx = 0

        # Clamp to valid range
        if not options or not (0 <= correct_idx < len(options)):
            continue

        correct_text = options[correct_idx]

        # Handle both answer formats:
        # Format A: full option text  e.g. "Neural Networks"
        # Format B: index string      e.g. "0", "1", "2", "3"
        if selected == correct_text:
            correct += 1
        elif selected is not None:
            try:
                if int(selected) == correct_idx:
                    correct += 1
            except (ValueError, TypeError):
                pass

    score = int((correct / total_mcqs) * 100) if total_mcqs else 100

    # 5. Append score with full MCQ detail for answer review in app.py
    scores = list(state.get("module_scores") or [])
    scores.append({
        "step":    idx + 1,
        "title":   current_title,
        "score":   score,
        "correct": correct,
        "total":   total_mcqs,
        "mcq_detail": [          # full MCQ data stored for per-question breakdown
            {
                "id":            str(q.get("id", "")),
                "question":      q.get("question", ""),
                "options":       q.get("options", []),
                "correct_index": int(q.get("correct_index", 0))
                                 if str(q.get("correct_index", 0)).isdigit()
                                 else 0,
                "explanation":   q.get("explanation", "")
            }
            for q in mcqs
            if isinstance(q, dict)
        ]
    })

    # 6. Increment current_step_index so graph routing advances correctly
    next_idx = idx + 1

    return {
        "student_mcq_answers": user_answers if isinstance(user_answers, dict) else {},
        "module_scores":       scores,
        "current_step_index":  next_idx
    }