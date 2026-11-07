import os
import sys
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

from langgraph.types import Command
from graph import app

def save_report_deliverables(topic: str, ai_report: str, student_report: str, evaluation: str):
    """Saves generated research outputs to disk in Markdown format."""
    os.makedirs("outputs", exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    clean_topic = "".join([c if c.isalnum() else "_" for c in topic])[:30]
    filename = f"outputs/Research_Report_{clean_topic}_{timestamp}.md"

    full_markdown = (
        f"# DUAL-REPORT AI RESEARCH SYNTHESIS\n\n"
        f"**Topic**: {topic}\n"
        f"**Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
        f"---\n\n"
        f"## 🤖 1. AI GENERATED RESEARCH REPORT\n\n{ai_report}\n\n"
        f"---\n\n"
        f"## ✍️ 2. STUDENT SYNTHESIS REPORT\n\n{student_report}\n\n"
        f"---\n\n"
        f"## 📊 3. PAUL-ELDER EVALUATION & METRICS\n\n{evaluation}\n"
    )

    with open(filename, "w", encoding="utf-8") as f:
        f.write(full_markdown)

    print(f"\n📁 Complete report deliverables successfully saved to:\n   -> {os.path.abspath(filename)}")

def run_complete_pipeline():
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("GOOGLE_API_KEY"):
        print("[❌ Error] Missing TAVILY_API_KEY or GOOGLE_API_KEY in .env file.")
        sys.exit(1)

    config = {"configurable": {"thread_id": "production_run_session_01"}}

    print("\n" + "=" * 65)
    print("🚀 DUAL-REPORT ENGINE — PRODUCTION INTERACTIVE RUNNER")
    print("=" * 65)

    topic_input = input("Enter research topic (or press Enter for default): ").strip()
    if not topic_input:
        topic_input = "Newton's Laws of Motion & Propulsion Systems"

    initial_state = {"topic": topic_input, "raw_source_text": ""}

    print(f"\nTarget Topic: {topic_input}")
    print("\n[Phase 1] Executing Automated Pipeline (Extractor -> RAG Writer -> Critic -> Question Gen)...")
    app.invoke(initial_state, config)

    state_snapshot = app.get_state(config)
    if not state_snapshot.next:
        print("\n[!] Pipeline completed without reaching an interrupt point.")
        return

    # Phase 2: Human-in-the-Loop Interactivity
    pending_tasks = state_snapshot.tasks
    if pending_tasks and pending_tasks[0].interrupts:
        interrupt_payload = pending_tasks[0].interrupts[0].value

        if interrupt_payload.get("type") == "student_questionnaire":
            print("\n" + "=" * 65)
            print("📝 STUDENT SYNTHESIS QUESTIONNAIRE (Human-in-the-Loop)")
            print("=" * 65)
            print(f"Instruction: {interrupt_payload.get('instruction')}\n")

            questions = interrupt_payload.get("questions", [])
            student_answers = []

            for idx, question in enumerate(questions, 1):
                print(f"[{idx}] {question}")
                ans = input("   Your Answer: ").strip()
                while not ans:
                    print("   ⚠️ Response cannot be blank.")
                    ans = input("   Your Answer: ").strip()
                student_answers.append(ans)
                print()

            print("=" * 65)
            print("🔄 Submitting responses & executing downstream nodes (Student Writer -> Evaluator)...")
            print("=" * 65)
            app.invoke(Command(resume=student_answers), config)

    # Phase 3: Display and Save Deliverables
    final_state = app.get_state(config).values
    ai_report = final_state.get("ai_report", "No AI report generated.")
    student_report = final_state.get("student_report", "No student report generated.")
    evaluation = final_state.get("comparative_analysis", "No analysis generated.")

    print("\n" + "=" * 70)
    print("🤖 1. AI GENERATED RESEARCH REPORT")
    print("=" * 70)
    print(ai_report)

    print("\n" + "=" * 70)
    print("✍️ 2. STUDENT SYNTHESIS REPORT")
    print("=" * 70)
    print(student_report)

    print("\n" + "=" * 70)
    print("📊 3. COMPARATIVE EVALUATION & PAUL-ELDER SCORES")
    print("=" * 70)
    print(evaluation)

    # Save to outputs/ directory
    save_report_deliverables(topic_input, ai_report, student_report, evaluation)

if __name__ == "__main__":
    run_complete_pipeline()