# state.py
from typing import TypedDict, List, Dict, Any, Optional

class SelfTutorState(TypedDict, total=False):
    # total=False makes all fields optional at construction time,
    # matching LangGraph's incremental state update pattern

    main_topic: str                          # Set once at app.invoke({"main_topic": topic})
    subtopics: List[Dict[str, Any]]          # Set by curriculum_planner_node
    current_step_index: int                  # Set by curriculum_planner_node, incremented each loop

    current_module_data: Optional[Dict[str, Any]]   # Set by module_content_node each cycle

    student_mcq_answers: Dict[str, str]      # String keys enforced for LangGraph Command(resume=...) safety
    module_scores: List[Dict[str, Any]]      # Appended per module by module_hitl_node

    project_prompt: Optional[str]            # Set by capstone_generator_node
    student_project_submission: Optional[str] # Set by project_hitl_node
    project_feedback: Optional[str]          # Set by project_evaluator_node

    is_mastery_completed: bool               # Set True by project_evaluator_node at END
    critic_approved: bool
    critic_retries: int                     # Set by tutor_critic_node each cycle