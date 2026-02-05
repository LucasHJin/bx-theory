"""Scheduler Agent - Generates study schedules based on course data and priorities."""

import json

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from .models import ParserOutput
from .parser import get_client
from .scheduler import (
    determine_priorities as _determine_priorities,
    generate_schedule as _generate_schedule,
    regenerate_schedule as _regenerate_schedule,
)


def _reconstruct_parser_output(parser_output_dict: dict) -> ParserOutput:
    """Reconstruct ParserOutput from state dict."""
    return ParserOutput.from_json(json.dumps(parser_output_dict))


def determine_priorities(tool_context: ToolContext) -> str:
    """Determine study priorities for each course based on exam dates, content volume, and weights.

    Use this tool FIRST before generating a schedule. It analyzes:
    - Exam date proximity (closer = higher priority)
    - Content volume (more pages = needs more time)
    - Midterm weight (higher % of grade = higher stakes)

    Requires: parser_output in state (call process_uploaded_files first)

    Args:
        tool_context: Provides access to session state with parsed course data

    Returns:
        Summary of course priorities with scores (0-10) and reasoning
    """
    parser_output_dict = tool_context.state.get("parser_output")
    if not parser_output_dict:
        return "Error: No parser output found in state. Run parser first."

    parser_output = _reconstruct_parser_output(parser_output_dict)
    client = get_client()

    print("Determining course priorities...")
    priorities = _determine_priorities(client, parser_output)

    # Store priorities in state
    tool_context.state["priorities"] = priorities

    # Format summary
    summary_parts = []
    if "course_priorities" in priorities:
        for code, info in priorities["course_priorities"].items():
            score = info.get("priority_score", "?")
            reasoning = info.get("reasoning", "")
            summary_parts.append(f"  {code}: {score}/10 — {reasoning}")
            print(f"  {code}: {score}/10 — {reasoning}")

    return "Determined priorities:\n" + "\n".join(summary_parts)


def generate_schedule(tool_context: ToolContext) -> str:
    """Generate a day-by-day study schedule based on course priorities.

    Use this tool AFTER determine_priorities. Creates a complete study plan with:
    - Learning sessions for each topic (2-4 hours based on page count)
    - Review sessions (spaced repetition: review_1 after 3-5 days, review_2 near exam)
    - Respects user preferences (max hours/day, rest days, study times)

    If validation errors exist from a previous attempt, automatically regenerates
    the schedule to fix those errors.

    Requires: parser_output and priorities in state

    Args:
        tool_context: Provides access to session state with course data and priorities

    Returns:
        Summary of generated schedule with day count, session count, and total hours
    """
    parser_output_dict = tool_context.state.get("parser_output")
    if not parser_output_dict:
        return "Error: No parser output found in state. Run parser first."

    priorities = tool_context.state.get("priorities")
    if not priorities:
        return "Error: No priorities found in state. Call determine_priorities first."

    parser_output = _reconstruct_parser_output(parser_output_dict)
    client = get_client()

    # Check if this is a retry (errors exist from previous validation)
    errors = tool_context.state.get("errors", [])
    previous_schedule = tool_context.state.get("schedule", [])

    if errors and previous_schedule:
        # Retry mode - regenerate with error feedback
        print(f"Regenerating schedule to fix {len(errors)} error(s)...")
        for e in errors:
            print(f"  {e}")

        schedule = _regenerate_schedule(
            client, parser_output, priorities, previous_schedule, errors
        )
    else:
        # Initial generation
        print("Generating study schedule...")
        schedule = _generate_schedule(client, parser_output, priorities)

    # Store schedule in state
    tool_context.state["schedule"] = schedule

    # Clear errors for fresh validation
    tool_context.state["errors"] = []

    # Calculate stats
    total_sessions = sum(len(day["sessions"]) for day in schedule)
    total_hours = sum(
        s["hours"] for day in schedule for s in day["sessions"]
    )
    print(f"  Generated {len(schedule)} study days")
    print(f"  Total sessions: {total_sessions}")
    print(f"  Total study hours: {total_hours:.1f}")

    return (
        f"Schedule generated successfully!\n"
        f"- {len(schedule)} study days\n"
        f"- {total_sessions} sessions\n"
        f"- {total_hours:.1f} total hours\n\n"
        f"The schedule is ready for validation."
    )


# Create the Scheduler Agent
scheduler_agent = LlmAgent(
    name="scheduler",
    description="Generates optimized study schedules",
    instruction="""You are a study schedule generator. Generate a study schedule by calling the tools.

STEPS:
1. Call determine_priorities() to rank courses by importance
2. Call generate_schedule() to create the day-by-day study plan

IMPORTANT: After calling BOTH tools, respond with a brief summary and STOP.
Do NOT call the tools multiple times. Do NOT ask follow-up questions.

Example response after tools complete:
"I've generated a study schedule with X days and Y sessions. The schedule is ready for validation."

Then STOP and let the validator check the schedule.""",
    tools=[determine_priorities, generate_schedule],
)
