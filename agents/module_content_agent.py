# agents/module_content_agent.py
import urllib.parse
from langchain_core.prompts import ChatPromptTemplate
from state import SelfTutorState
from key_rotator import get_llm, safe_parse_json

def module_content_node(state: SelfTutorState) -> dict:
    idx = state.get("current_step_index", 0)
    topic = state.get("main_topic") or state.get("topic", "General Subject")

    # Track critic retry attempt number for this module
    critic_retries = state.get("critic_retries", 0)
    if critic_retries > 0:
        print(f"🔄 [module_content] Retry attempt {critic_retries} for Module {idx + 1} "
              f"(critic previously rejected)")

    # 1. Safely unpack subtopics
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

    # 2. Protect against IndexError on subtopic access
    if not subtopics or idx >= len(subtopics):
        current_subtopic = {
            "step": idx + 1,
            "title": f"Module {idx + 1}: Core Principles",
            "focus": f"Fundamentals of {topic}"
        }
        total_steps = 5
    else:
        current_subtopic = subtopics[idx]
        total_steps = len(subtopics)

        # Normalize if LLM hallucinated list of strings instead of dicts
        if not isinstance(current_subtopic, dict):
            current_subtopic = {
                "step": idx + 1,
                "title": str(current_subtopic)[:50] if current_subtopic else f"Module {idx + 1}",
                "focus": f"Fundamentals of {topic}"
            }

    title = current_subtopic.get("title", "Core Concepts")
    focus = current_subtopic.get("focus", "Fundamentals")

    print("\n" + "="*60 + f"\nMODULE {idx+1}/{total_steps} — {title}\n" + "="*60)

    prompt = ChatPromptTemplate.from_template(
        """You are an AI tutor teaching Step {step}/{total_steps} of '{topic}': '{title}'. Focus: {focus}.

Provide:
1. A concise 100-word explanation in 3 key bullet points.
2. Two targeted YouTube search query strings for deep-dive learning.
3. Exactly two multiple-choice questions (MCQs) testing this concept.

CRITICAL: Each MCQ must have EXACTLY 4 answer options.

Return ONLY a raw JSON object without conversational text or code block wrapping. Format keys as:
{{
  "explanation": "string",
  "youtube_queries": ["query 1", "query 2"],
  "mcqs": [
    {{
      "id": "1",
      "question": "string",
      "options": ["opt0", "opt1", "opt2", "opt3"],
      "correct_index": 0,
      "explanation": "string"
    }},
    {{
      "id": "2",
      "question": "string",
      "options": ["opt0", "opt1", "opt2", "opt3"],
      "correct_index": 0,
      "explanation": "string"
    }}
  ]
}}"""
    )

    llm = get_llm()
    res = (prompt | llm).invoke({
        "step": idx + 1,
        "total_steps": total_steps,
        "topic": topic,
        "title": title,
        "focus": focus
    }).content

    # safe_parse_json never raises — returns default={} on failure
    data = safe_parse_json(res, default={})

    # 3. Validate top-level structure
    if not isinstance(data, dict) or "mcqs" not in data:
        print("⚠️ Failed to parse Module Content JSON. Using fallback structure.")
        data = {
            "explanation": (
                f"• Core overview of {title}.\n"
                f"• Key applications and mechanisms.\n"
                f"• Practical implementation considerations."
            ),
            "youtube_queries": [
                f"{topic} {title} tutorial",
                f"{title} explained visually"
            ],
            "mcqs": [
                {
                    "id": "1",
                    "question": f"What is the primary focus of {title}?",
                    "options": [
                        focus,
                        "Unrelated behavior",
                        "Hardware assembly",
                        "Network latency"
                    ],
                    "correct_index": 0,
                    "explanation": f"{title} focuses directly on {focus}."
                },
                {
                    "id": "2",
                    "question": f"Which component best describes {title}?",
                    "options": [
                        "Core Algorithmic Module",
                        "Database Shard",
                        "Operating System Thread",
                        "CSS Layout"
                    ],
                    "correct_index": 0,
                    "explanation": "This module represents a fundamental algorithmic block."
                }
            ]
        }

    # 4. Ensure top-level explanation exists
    if "explanation" not in data or not data["explanation"]:
        data["explanation"] = (
            f"• Core overview of {title}.\n"
            f"• Key applications and mechanisms.\n"
            f"• Practical implementation considerations."
        )

    # 5. Validate and normalize each MCQ
    validated_mcqs = []
    for i, q in enumerate(data.get("mcqs", [])):
        if not isinstance(q, dict):
            continue

        # Cast correct_index to int safely
        try:
            correct_index = int(q.get("correct_index", 0))
        except (ValueError, TypeError):
            correct_index = 0

        # Validate options as list of exactly 4 strings, pad if needed
        options = q.get("options", [])
        if not isinstance(options, list):
            options = []
        options = [str(o) for o in options[:4]]
        while len(options) < 4:
            options.append(f"Option {len(options) + 1}")

        # Clamp correct_index to valid range
        correct_index = min(correct_index, len(options) - 1)

        validated_mcqs.append({
            "id": str(q.get("id", i + 1)),
            "question": str(q.get("question", f"Question {i+1} about {title}")),
            "options": options,
            "correct_index": correct_index,
            "explanation": str(q.get("explanation", f"The correct answer relates to {focus}."))
        })

    # Pad to exactly 2 MCQs if LLM returned fewer
    if len(validated_mcqs) < 2:
        for j in range(len(validated_mcqs), 2):
            validated_mcqs.append({
                "id": str(j + 1),
                "question": f"Which of the following best describes {title}?",
                "options": [
                    focus,
                    "Unrelated behavior",
                    "Hardware assembly",
                    "Network latency"
                ],
                "correct_index": 0,
                "explanation": f"{title} focuses on {focus}."
            })

    data["mcqs"] = validated_mcqs

    # 6. Generate YouTube links deterministically
    yt_links = []
    for q in data.get("youtube_queries", []):
        encoded = urllib.parse.quote(str(q))
        yt_links.append({
            "query": q,
            "url": f"https://www.youtube.com/results?search_query={encoded}"
        })
    data["youtube_links"] = yt_links

    return {
        "current_module_data": data,
        # Increment retry counter so graph.py route_critic can cap retries
        "critic_retries": critic_retries + 1
    }