import os
from dotenv import load_dotenv
from langgraph.types import Command
from graph import app

load_dotenv()

# We assign a static thread_id so MemorySaver knows exactly which session to pause/resume
config = {"configurable": {"thread_id": "session_003"}}

initial_state = {
    "topic": "Impact of Solid State Batteries on EV Range",
    "language": "English",
    "search_results": "",
    "scraped_content": "",
    "report": "",
    "feedback": "",
    "revision_count": 0,
    "source_map": {},
    "flagged_contradictions": [],
    "student_synthesis": "",
    "challenge_history": [],
    "challenge_round": 0
}

print("\n--- STAGE 1: Running Automated Search, RAG & Contradiction Detection ---")
# Initial execution runs until it hits the first interrupt() in contradiction_detector_node
app.invoke(initial_state, config)

# --- STAGE 2: Interactive Interrupt Loop ---
while True:
    # Pull the exact current state from the MemorySaver
    state_snapshot = app.get_state(config)
    
    # If there's no pending interrupt/next task, the graph has reached END
    if not state_snapshot.next:
        break
        
    # Extract the payload dictionary passed into interrupt() inside graph.py
    pending_interrupt = state_snapshot.tasks[0].interrupts[0].value
    msg_type = pending_interrupt.get("type")
    message = pending_interrupt.get("message")
    
    # Render the correct terminal prompt based on the interrupt type
    if msg_type == "initial_synthesis":
        print("\n" + "=" * 60)
        print("🎯 RESEARCH SPARRING WORKSPACE: REVIEW FLAGGED CONTRADICTIONS")
        print("=" * 60)
        print(message)
        user_reply = input("\n✍️ Enter your synthesis/conclusion considering the evidence above: ")
        
    elif msg_type == "off_topic":
        print(f"\n⚠️ WARNING: {message}")
        user_reply = input("✍️ Please provide an on-topic synthesis: ")
        
    elif msg_type == "challenge_question":
        current_round = pending_interrupt.get("round", 1)
        # Message here contains the pushback question from the Challenger agent
        print(f"\n[Challenger Pushback]:\n{message}")
        user_reply = input(f"\n✍️ Your response to the Challenger (Round {current_round}/3): ")
        
    else:
        user_reply = input("\n✍️ System paused. Press enter to continue... ")

    print("\n--- Processing Response... ---")
    
    # Feed the student's answer back into the graph to resume execution from the exact paused node
    app.invoke(Command(resume=user_reply), config)

# --- STAGE 3: Final Output ---
final_state = app.get_state(config).values
print("\n" + "=" * 60)
print("FINAL ANNOTATED REPORT (With Student Synthesis Integrated)")
print("=" * 60)
print(final_state.get("report", "No report generated."))