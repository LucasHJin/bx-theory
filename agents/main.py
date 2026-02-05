import os
import asyncio
from dotenv import load_dotenv

from google.genai import types
from google.adk.runners import InMemoryRunner

from .agent import create_root_agent


async def main():
    load_dotenv()

    files_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "files")
    user_message = (
        "I can study max 6 hours per day, I prefer studying in the mornings "
        "and afternoons. I'd like Sundays off. I want to use spaced repetition."
    )

    # Create the agent hierarchy
    print("=" * 60)
    print("STUDY PLANNER (ADK)")
    print("=" * 60)

    root_agent = create_root_agent()
    runner = InMemoryRunner(agent=root_agent, app_name="study_planner")
    runner.auto_create_session = True

    # Prepare the initial message with files_dir context
    initial_message = f"""Create a study plan for me.

Files directory: {files_dir}

My preferences: {user_message}"""

    # Run the agent and collect events
    final_state = {}
    async for event in runner.run_async(
        user_id="student",
        session_id="planning_session",
        new_message=types.Content(
            role="user",
            parts=[types.Part(text=initial_message)]
        ),
    ):
        # Track the latest state from events
        if hasattr(event, "actions") and event.actions.state_delta:
            final_state.update(event.actions.state_delta)

        # Print agent activity
        if hasattr(event, "author") and event.author:
            print(f"\n[{event.author}]", end=" ")
        if hasattr(event, "content") and event.content:
            for part in event.content.parts:
                if hasattr(part, "text") and part.text:
                    print(part.text[:200] + "..." if len(part.text) > 200 else part.text)

    # Extract final outputs from state
    csv_output = final_state.get("csv_output", "")
    errors = final_state.get("errors", [])
    warnings = final_state.get("warnings", [])

    # Write CSV to file
    if csv_output:
        output_file = "study_plan.csv"
        with open(output_file, "w") as f:
            f.write(csv_output)
        print(f"\nStudy plan written to {output_file}")

        print("\n" + "=" * 60)
        print("FINAL CSV OUTPUT")
        print("=" * 60)
        print(csv_output)
    else:
        print("\nNo CSV output generated.")

    if errors:
        print(f"\nRemaining errors: {len(errors)}")
        for e in errors:
            print(f"  {e}")

    if warnings:
        print(f"\nWarnings: {len(warnings)}")
        for w in warnings:
            print(f"  {w}")


if __name__ == "__main__":
    asyncio.run(main())
