"""Validator Agent - Validates schedules and formats output as CSV."""

import json
import os

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from .models import ParserOutput
from .validator import run_validator

# Output directory for CSV files
OUTPUT_DIR = os.path.dirname(os.path.dirname(__file__))


def _reconstruct_parser_output(parser_output_dict: dict) -> ParserOutput:
    """Reconstruct ParserOutput from state dict."""
    return ParserOutput.from_json(json.dumps(parser_output_dict))


def validate_and_format(tool_context: ToolContext) -> str:
    """Validate the generated schedule and format it as CSV.

    Use this tool to check the generated schedule for errors and produce final output.

    Validation checks:
    1. Daily study hours don't exceed user's max_hours_per_day limit
    2. All required topics have learning sessions
    3. All sessions are scheduled before their course's exam date
    4. Spaced repetition is properly applied (review sessions after learning)

    The tool automatically:
    - Saves the schedule as study_plan.csv
    - Exits the retry loop if validation passes OR max retries (3) reached
    - Returns errors to the scheduler for fixing if validation fails

    Requires: parser_output and schedule in state

    Args:
        tool_context: Provides access to session state with schedule and course data

    Returns:
        Validation summary with error/warning counts and full CSV output
    """
    parser_output_dict = tool_context.state.get("parser_output")
    if not parser_output_dict:
        return "Error: No parser output found in state."

    schedule = tool_context.state.get("schedule")
    if not schedule:
        return "Error: No schedule found in state."

    # Track validation attempts
    attempt = tool_context.state.get("validation_attempt", 0) + 1
    tool_context.state["validation_attempt"] = attempt
    max_attempts = 3

    parser_output = _reconstruct_parser_output(parser_output_dict)

    print(f"Running validation checks (attempt {attempt}/{max_attempts})...")
    csv_output, errors, warnings = run_validator(parser_output, schedule)

    # Store results in state
    tool_context.state["errors"] = errors
    tool_context.state["warnings"] = warnings
    tool_context.state["csv_output"] = csv_output

    # Build summary
    summary_parts = [f"Validation attempt {attempt}/{max_attempts}: {len(errors)} error(s), {len(warnings)} warning(s)"]

    if errors:
        summary_parts.append("\nErrors found:")
        for e in errors:
            summary_parts.append(f"  - {e}")

    if warnings:
        summary_parts.append("\nWarnings:")
        for w in warnings:
            summary_parts.append(f"  - {w}")

    # Include full CSV in response
    summary_parts.append("\n--- STUDY SCHEDULE CSV ---")
    summary_parts.append(csv_output.strip())

    # Save CSV to file
    csv_filename = "study_plan.csv"
    csv_path = os.path.join(OUTPUT_DIR, csv_filename)
    with open(csv_path, "w") as f:
        f.write(csv_output)
    print(f"CSV saved to: {csv_path}")
    summary_parts.append(f"\n📁 CSV saved to: {csv_path}")

    # Exit conditions
    if not errors:
        print("Validation passed! Exiting scheduling loop.")
        summary_parts.insert(1, "\n✓ Schedule is valid!")
        tool_context.actions.escalate = True
    elif attempt >= max_attempts:
        print(f"Max attempts ({max_attempts}) reached. Exiting with best effort schedule.")
        summary_parts.insert(1, f"\n⚠ Max retries reached. Returning schedule with {len(errors)} remaining error(s).")
        tool_context.actions.escalate = True
    else:
        summary_parts.append(f"\nRetrying... (attempt {attempt + 1}/{max_attempts})")

    return "\n".join(summary_parts)


# Create the Validator Agent
validator_agent = LlmAgent(
    name="validator",
    description="Validates schedules and outputs CSV",
    instruction="""You are a schedule validator. Validate the schedule and output CSV.

STEPS:
1. Call validate_and_format() to check the schedule

The tool will:
- Validate against constraints (hours, coverage, dates, spaced repetition)
- Save CSV to study_plan.csv
- Show the schedule in the response

IMPORTANT: After calling the tool, display the FULL CSV from the tool response.
Format it nicely so the user can see their complete study plan.

Then STOP.""",
    tools=[validate_and_format],
)
