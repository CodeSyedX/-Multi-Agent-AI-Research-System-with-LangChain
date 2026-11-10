# agents/project_hitl_agent.py
from langgraph.types import interrupt
from state import SelfTutorState

def project_submission_hitl_node(state: SelfTutorState) -> dict:
    print("\n" + "="*60 + "\nCAPSTONE — Awaiting Student Project Submission\n" + "="*60)

    # 1. Safely resolve the project prompt
    project_prompt = (
        state.get("project_prompt")
        or "Please submit your final capstone project code or report for evaluation."
    )

    # 2. Fire interrupt — pauses graph until app.py sends Command(resume=...)
    submission = interrupt({
        "type": "capstone_submission",
        "project_prompt": project_prompt
    })

    # 3. Bug 1 fixed: handle both plain string and dict resume payloads
    if isinstance(submission, dict):
        # app.py sent Command(resume={"submission": "student text"})
        # try common key names in order of likelihood
        raw = (
            submission.get("submission")
            or submission.get("text")
            or submission.get("content")
            or submission.get("answer")
            or next(iter(submission.values()), None)  # fallback: first value
        )
        safe_submission = str(raw).strip() if raw else ""
    elif isinstance(submission, str):
        safe_submission = submission.strip()
    else:
        # Unexpected type — cast safely
        safe_submission = str(submission).strip() if submission else ""

    # Bug 2 fixed: catch empty string after strip
    if not safe_submission:
        print("⚠️ Warning: Student submitted empty or whitespace-only content.")
        safe_submission = "No submission provided."

    # Bug 3: warn on suspiciously short submissions (non-blocking)
    if len(safe_submission) < 20 and safe_submission != "No submission provided.":
        print(f"⚠️ Warning: Submission is very short ({len(safe_submission)} chars): '{safe_submission}'")

    # Bug 5 fixed: log receipt for graph debugging
    print(f"✅ Submission received — {len(safe_submission)} characters.")

    return {"student_project_submission": safe_submission}