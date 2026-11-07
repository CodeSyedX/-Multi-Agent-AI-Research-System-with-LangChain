import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

from state import DualReportState
from agents import (
    extractor_node,
    ai_writer_node,
    critic_node,
    question_generator_node,
    student_hitl_node,
    student_writer_node,
    evaluator_node,
)

load_dotenv()

memory = MemorySaver()

def route_after_critic(state: DualReportState) -> str:
    """Routes back to AI Writer if validation fails, or proceeds to Question Generator if approved."""
    return "approved" if state.get("critic_approved", False) else "rejected"

# Initialize StateGraph with DualReportState schema
workflow = StateGraph(DualReportState)

# Add modular pipeline nodes
workflow.add_node("extractor", extractor_node)
workflow.add_node("ai_writer", ai_writer_node)
workflow.add_node("critic", critic_node)
workflow.add_node("question_generator", question_generator_node)
workflow.add_node("student_hitl", student_hitl_node)
workflow.add_node("student_writer", student_writer_node)
workflow.add_node("evaluator", evaluator_node)

# Define execution flow
workflow.add_edge(START, "extractor")
workflow.add_edge("extractor", "ai_writer")
workflow.add_edge("ai_writer", "critic")

workflow.add_conditional_edges("critic", route_after_critic, {
    "approved": "question_generator",
    "rejected": "ai_writer"
})

workflow.add_edge("question_generator", "student_hitl")
workflow.add_edge("student_hitl", "student_writer")
workflow.add_edge("student_writer", "evaluator")
workflow.add_edge("evaluator", END)

# Compile graph with checkpoint memory for interrupt/resume functionality
app = workflow.compile(checkpointer=memory)