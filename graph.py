# graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from state import SelfTutorState

# Bug 1 fixed: import directly from individual agent modules
from agents.curriculum_planner_agent import curriculum_planner_node
from agents.module_content_agent import module_content_node
from agents.tutor_critic_agent import tutor_critic_node
from agents.module_hitl_agent import module_quiz_hitl_node
from agents.capstone_generator_agent import capstone_generator_node
from agents.project_hitl_agent import project_submission_hitl_node
from agents.project_evaluator_agent import project_evaluator_node

# Bug 5 & 6 fixed: singleton pattern — memory and app created once only
_memory = None
_app = None

def _get_memory() -> MemorySaver:
    global _memory
    if _memory is None:
        _memory = MemorySaver()
        print("✅ [Graph] MemorySaver checkpoint store initialized.")
    return _memory


def route_module_loop(state: SelfTutorState) -> str:
    """
    Routes after module_hitl completes.
    current_step_index is already incremented by module_hitl_agent.
    If index < total modules → next module.
    If index >= total modules → capstone.
    """
    current_idx = state.get("current_step_index", 0)
    subtopics = state.get("subtopics", [])
    total_subtopics = len(subtopics) if subtopics else 5

    # Bug 2 noted: warn if subtopics count unexpected
    if subtopics and total_subtopics != 5:
        print(f"⚠️ [Graph] Warning: Expected 5 subtopics, found {total_subtopics}. "
              f"Routing based on actual count.")

    if current_idx < total_subtopics:
        print(f"→ [Graph] Routing to next module ({current_idx + 1}/{total_subtopics})")
        return "next_module"

    print(f"→ [Graph] All {total_subtopics} modules complete. Routing to capstone.")
    return "capstone_project"


def route_critic(state: SelfTutorState) -> str:
    """
    Bug 3 fixed: conditional routing on critic_approved.
    Approved → proceed to student quiz.
    Rejected → retry module_content generation.
    Bug 4 fixed: retry counter cap prevents infinite loop.
    """
    # Track retry attempts via module_scores length as a proxy,
    # or use a dedicated critic_retries counter if added to state
    critic_approved = state.get("critic_approved", False)
    critic_retries = state.get("critic_retries", 0)

    max_retries = 2  # Maximum regeneration attempts per module

    if critic_approved:
        print("→ [Graph] Critic APPROVED — proceeding to module quiz.")
        return "approved"

    if critic_retries >= max_retries:
        print(f"⚠️ [Graph] Critic REJECTED but max retries ({max_retries}) reached. "
              f"Forcing forward to module quiz with current content.")
        return "approved"  # Force forward to avoid infinite loop

    print(f"→ [Graph] Critic REJECTED — retrying module_content "
          f"(attempt {critic_retries + 1}/{max_retries})")
    return "retry_content"


def build_graph() -> StateGraph:
    workflow = StateGraph(SelfTutorState)

    # Register nodes
    workflow.add_node("curriculum_planner", curriculum_planner_node)
    workflow.add_node("module_content", module_content_node)
    workflow.add_node("critic", tutor_critic_node)
    workflow.add_node("module_hitl", module_quiz_hitl_node)
    workflow.add_node("capstone_gen", capstone_generator_node)
    workflow.add_node("project_hitl", project_submission_hitl_node)
    workflow.add_node("project_evaluator", project_evaluator_node)

    # Edges
    workflow.add_edge(START, "curriculum_planner")
    workflow.add_edge("curriculum_planner", "module_content")
    workflow.add_edge("module_content", "critic")

    # Bug 3 fixed: conditional routing from critic
    workflow.add_conditional_edges("critic", route_critic, {
        "approved": "module_hitl",
        "retry_content": "module_content"   # loops back on rejection
    })

    # Module loop routing
    workflow.add_conditional_edges("module_hitl", route_module_loop, {
        "next_module": "module_content",
        "capstone_project": "capstone_gen"
    })

    workflow.add_edge("capstone_gen", "project_hitl")
    workflow.add_edge("project_hitl", "project_evaluator")
    workflow.add_edge("project_evaluator", END)

    return workflow


def get_app():
    """
    Bug 5 & 6 fixed: singleton app factory.
    Safe to call multiple times — returns same compiled app instance.
    """
    global _app
    if _app is None:
        print("✅ [Graph] Compiling StateGraph for first time...")
        workflow = build_graph()
        _app = workflow.compile(checkpointer=_get_memory())
        print("✅ [Graph] StateGraph compiled successfully.")
    return _app


# Module-level convenience export — still safe because get_app() is idempotent
app = get_app()