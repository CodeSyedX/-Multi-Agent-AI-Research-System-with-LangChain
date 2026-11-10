# agents/tutor_critic_agent.py
from state import SelfTutorState

# Required keys every MCQ dict must contain
_REQUIRED_MCQ_KEYS = {"id", "question", "options", "correct_index", "explanation"}

def tutor_critic_node(state: SelfTutorState) -> dict:
    idx = state.get("current_step_index", 0)
    critic_retries = state.get("critic_retries", 0)

    print(f"\n{'='*60}\nCRITIC — Structural Validation for Module {idx + 1} "
          f"(attempt {critic_retries})\n{'='*60}")

    # 1. Safely retrieve current module data
    data = state.get("current_module_data") or {}
    mcqs = data.get("mcqs", [])

    # 2. Enforce exactly 2 MCQs
    if not isinstance(mcqs, list) or len(mcqs) != 2:
        print(f"✗ REJECTED — Expected 2 MCQs, found: "
              f"{len(mcqs) if isinstance(mcqs, list) else type(mcqs).__name__}")
        return {
            "critic_approved": False
            # critic_retries NOT reset here — graph.py uses it to cap retries
        }

    valid = True

    for i, q in enumerate(mcqs):
        # Must be a dict
        if not isinstance(q, dict):
            print(f"✗ REJECTED — MCQ[{i}] is not a dict: {type(q).__name__}")
            valid = False
            break

        # Validate all required keys present
        missing_keys = _REQUIRED_MCQ_KEYS - q.keys()
        if missing_keys:
            print(f"✗ REJECTED — MCQ[{i}] missing required keys: {missing_keys}")
            valid = False
            break

        # Enforce exactly 4 options
        options = q.get("options", [])
        if not isinstance(options, list) or len(options) != 4:
            print(f"✗ REJECTED — MCQ[{i}] must have exactly 4 options, "
                  f"found: {len(options) if isinstance(options, list) else type(options).__name__}")
            valid = False
            break

        # Validate all options are non-empty strings
        for j, opt in enumerate(options):
            if not isinstance(opt, str) or not opt.strip():
                print(f"✗ REJECTED — MCQ[{i}] option[{j}] is empty or "
                      f"not a string: {repr(opt)}")
                valid = False
                break
        if not valid:
            break

        # Validate correct_index is int and in range
        correct_index = q.get("correct_index")
        try:
            correct_index = int(correct_index)
        except (ValueError, TypeError):
            print(f"✗ REJECTED — MCQ[{i}] correct_index not castable to int: "
                  f"{repr(correct_index)}")
            valid = False
            break

        if not (0 <= correct_index < len(options)):
            print(f"✗ REJECTED — MCQ[{i}] correct_index {correct_index} "
                  f"out of range for {len(options)} options")
            valid = False
            break

        # Validate question and explanation are non-empty strings
        for key in ("question", "explanation"):
            val = q.get(key, "")
            if not isinstance(val, str) or not val.strip():
                print(f"✗ REJECTED — MCQ[{i}] '{key}' is empty or "
                      f"not a string: {repr(val)}")
                valid = False
                break
        if not valid:
            break

        print(f"  ✓ MCQ[{i}] id={q.get('id')} — passed all checks")

    if valid:
        print("→ Status: APPROVED ✅ (All MCQ structural checks passed)")
        return {
            "critic_approved": True,
            "critic_retries": 0     # reset counter cleanly for next module
        }

    # Rejected — do NOT reset critic_retries so route_critic can enforce cap
    print("→ Status: REJECTED ❌ — graph will retry module_content or force forward")
    return {
        "critic_approved": False
        # critic_retries preserved from state — graph.py route_critic reads it
    }