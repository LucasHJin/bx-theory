"""Orchestrator - Assembles the full agent hierarchy for study planning."""

from google.adk.agents import SequentialAgent, LoopAgent

from .parser_agent import parser_agent
from .scheduler_agent import scheduler_agent
from .validator_agent import validator_agent
from .callbacks import before_model_callback


# Model to use for all agents
# Options: "gemini-2.0-flash" (stable), "gemini-3-flash-preview" (newer but may be overloaded)
MODEL = "gemini-3-flash-preview"


def create_root_agent() -> SequentialAgent:
    """Create the root agent that orchestrates the study planning workflow.

    Architecture:
        RootAgent (SequentialAgent)
        ├── ParserAgent (LlmAgent) - with before_model_callback
        │   └── tools: process_uploaded_files, set_preferences
        └── ScheduleLoopAgent (LoopAgent, max_iterations=3)
            ├── SchedulerAgent (LlmAgent) - with before_model_callback
            │   └── tools: determine_priorities, generate_schedule
            └── ValidatorAgent (LlmAgent) - with before_model_callback
                └── tools: validate_and_format (sets escalate=True on success)

    All LlmAgents have before_model_callback to strip file content from every
    LLM request, preventing token overflow from uploaded PDFs in session history.

    Returns:
        The root SequentialAgent
    """
    # Set model and callback for ALL agents (callback strips file content from every request)
    parser_agent.model = MODEL
    parser_agent.before_model_callback = before_model_callback

    scheduler_agent.model = MODEL
    scheduler_agent.before_model_callback = before_model_callback

    validator_agent.model = MODEL
    validator_agent.before_model_callback = before_model_callback

    # Create the scheduling loop (Scheduler -> Validator, retry up to 3 times)
    schedule_loop = LoopAgent(
        name="schedule_loop",
        description="Retry loop for schedule generation and validation",
        sub_agents=[scheduler_agent, validator_agent],
        max_iterations=3,
    )

    # Create the root sequential agent (Parser -> ScheduleLoop)
    root_agent = SequentialAgent(
        name="study_planner",
        description="Multi-agent study planner: parses course files, generates optimized schedules, and validates them",
        sub_agents=[parser_agent, schedule_loop],
    )

    return root_agent


# Export for ADK web interface
root_agent = create_root_agent()
