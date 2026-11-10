# agents/topic_validator.py
from langchain_core.prompts import ChatPromptTemplate
from key_rotator import get_llm, safe_parse_json


def validate_topic(topic: str) -> dict:
    """
    Validates whether the entered topic is a real, teachable subject.
    Returns:
        {"valid": True,  "cleaned": "...", "reason": ""}
        {"valid": False, "cleaned": "",    "reason": "why invalid"}
    """
    # Basic length check before spending an LLM call
    if not topic or len(topic.strip()) < 3:
        return {
            "valid": False,
            "cleaned": "",
            "reason": "Topic is too short. Please enter a meaningful subject."
        }

    # Obvious gibberish check — all non-alpha characters or single repeated char
    stripped = topic.strip()
    alpha_chars = [c for c in stripped if c.isalpha()]
    if len(alpha_chars) < 3:
        return {
            "valid": False,
            "cleaned": "",
            "reason": "Topic contains no recognisable words. Please enter a real subject."
        }

    prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            # No { } braces here — avoids f-string parse errors entirely
            "You are a JSON-only API endpoint. "
            "Respond with ONLY valid JSON. "
            "No markdown, no code fences, no explanation, no preamble. "
            "Start your response directly with the opening brace."
        ),
        (
            "human",
            # All literal braces in JSON examples doubled as {{ and }}
            "Is '{topic}' a valid educational topic that can be taught "
            "in a 5-module online course?\n\n"
            "A VALID topic is any real subject, skill, technology, concept, "
            "science, art, language, or domain of knowledge.\n\n"
            "An INVALID topic is: random characters, gibberish, offensive content, "
            "meaningless words like 'thing' or 'stuff', or a single letter.\n\n"
            "Return a JSON object with EXACTLY these three keys:\n"
            "  valid   : boolean true or false\n"
            "  cleaned : the corrected and normalized topic name as a string "
            "(empty string if invalid)\n"
            "  reason  : brief reason string if invalid, empty string if valid\n\n"
            "Examples:\n"
            "Input 'python programming' -> "
            "{{\"valid\": true, \"cleaned\": \"Python Programming\", \"reason\": \"\"}}\n"
            "Input 'asdfgh' -> "
            "{{\"valid\": false, \"cleaned\": \"\", \"reason\": \"Not a recognisable topic\"}}\n"
            "Input 'machine learning' -> "
            "{{\"valid\": true, \"cleaned\": \"Machine Learning\", \"reason\": \"\"}}\n\n"
            "Now evaluate: '{topic}'"
        )
    ])

    try:
        llm = get_llm()
        res = (prompt | llm).invoke({"topic": stripped}).content
        data = safe_parse_json(res, default={})

        if isinstance(data, dict) and "valid" in data:
            return {
                "valid":   bool(data.get("valid", False)),
                "cleaned": str(data.get("cleaned", stripped)).strip(),
                "reason":  str(data.get("reason", "")).strip()
            }

        # safe_parse_json returned unexpected structure — fail open
        print(f"⚠️ [TopicValidator] Unexpected parse result: {data}. Allowing topic through.")
        return {"valid": True, "cleaned": stripped, "reason": ""}

    except Exception as e:
        # Validator itself errored — fail open so app still works
        print(f"⚠️ [TopicValidator] Validation call failed ({e}). Allowing topic through.")
        return {"valid": True, "cleaned": stripped, "reason": ""}