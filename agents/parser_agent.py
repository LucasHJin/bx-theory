"""Parser Agent - Extracts course data from PDFs and parses user preferences.

Handles file uploads from ADK web interface by saving them locally and
processing with pdfplumber (not sending raw content to the model).
"""

import os
import json
import tempfile
from dataclasses import asdict
from pathlib import Path

from google.adk.agents import LlmAgent
from google.adk.tools import ToolContext

from .models import Topic, Course, UserPreferences, ParserOutput
from .parser import (
    get_client,
    classify_files,
    extract_pdf_text,
    parse_midterm_date,
    parse_midterm_chapters,
    parse_midterm_topics,
    parse_midterm_weight,
    parse_textbook_toc,
    parse_user_preferences,
)


# Session-specific directory for uploaded files (shared with callbacks.py)
UPLOAD_DIR = os.path.join(tempfile.gettempdir(), "study_planner_uploads")


def _get_upload_dir(tool_context: ToolContext) -> str:
    """Get or create the upload directory for this session."""
    # Check state first (set by callback)
    session_id = tool_context.state.get("session_id", "default")

    # Fallback to context attributes
    if session_id == "default":
        if hasattr(tool_context, 'session') and tool_context.session:
            session_id = getattr(tool_context.session, 'id', 'default')

    upload_dir = os.path.join(UPLOAD_DIR, str(session_id))
    os.makedirs(upload_dir, exist_ok=True)
    return upload_dir


def upload_file(tool_context: ToolContext, file_name: str, file_content_base64: str) -> str:
    """Save an uploaded file for processing.

    Args:
        file_name: Name of the file (e.g., "PHYS 234 - Syllabus.pdf")
        file_content_base64: Base64-encoded file content

    Returns:
        Confirmation message
    """
    import base64

    upload_dir = _get_upload_dir(tool_context)
    file_path = os.path.join(upload_dir, file_name)

    # Decode and save
    content = base64.b64decode(file_content_base64)
    with open(file_path, 'wb') as f:
        f.write(content)

    # Track uploaded files in state
    uploaded = tool_context.state.get("uploaded_files", [])
    if file_path not in uploaded:
        uploaded.append(file_path)
    tool_context.state["uploaded_files"] = uploaded

    return f"Saved {file_name} ({len(content)} bytes)"


def process_uploaded_files(tool_context: ToolContext) -> str:
    """Process all uploaded PDF files to extract course information.

    Use this tool FIRST when the user uploads course files (syllabi, midterm
    overviews, textbooks). This tool:
    1. Classifies each PDF by course code and type (syllabus/midterm/textbook)
    2. Extracts midterm dates, weights, topics, and page counts
    3. Stores parsed data in state for the scheduler to use

    Args:
        tool_context: Provides access to session state and uploaded files

    Returns:
        Summary of parsed courses including topic counts, exam dates, and weights
    """
    # Check if we already have parsed data in state
    if "parser_output" in tool_context.state:
        existing = tool_context.state["parser_output"]
        if existing.get("courses"):
            return f"Already parsed {len(existing['courses'])} courses: {list(existing['courses'].keys())}"

    # Get upload directory
    upload_dir = _get_upload_dir(tool_context)

    # Check for files saved by callback
    uploaded_paths = tool_context.state.get("uploaded_files", [])
    if uploaded_paths:
        # Copy callback-saved files to our upload dir if they're elsewhere
        for path in uploaded_paths:
            if os.path.exists(path) and path != upload_dir:
                src = Path(path)
                if src.suffix.lower() == ".pdf":
                    dest = Path(upload_dir) / src.name
                    if not dest.exists():
                        dest.write_bytes(src.read_bytes())
                        print(f"Copied {src.name} to upload directory")

    # Check for uploaded files
    pdf_files = list(Path(upload_dir).glob("*.pdf"))
    if not pdf_files:
        return "No PDF files found. Please upload your course files (syllabi, midterm overviews, textbooks)."

    print(f"Processing {len(pdf_files)} PDF files from {upload_dir}...")

    client = get_client()

    # Classify files by course
    print("Classifying files...")
    course_files = classify_files(upload_dir)
    print(f"Found {len(course_files)} courses: {list(course_files.keys())}")

    if not course_files:
        return f"Could not identify courses from uploaded files: {[f.name for f in pdf_files]}"

    # Parse each course
    courses: dict[str, Course] = {}
    for course_code, files in course_files.items():
        print(f"\nParsing {course_code}...")

        # Parse midterm overview
        midterm_date = ""
        chapters_covered: list[str] = []
        topics: list[Topic] = []

        if "midterm" in files:
            print("  Parsing midterm overview...")
            midterm_text = extract_pdf_text(str(files["midterm"]))
            midterm_date = parse_midterm_date(midterm_text)
            chapters_covered = parse_midterm_chapters(midterm_text)
            print(f"  Date: {midterm_date}, Chapters: {chapters_covered}")

            raw_topics = parse_midterm_topics(client, midterm_text)
            topics = [
                Topic(name=t["name"], chapters=t["chapters"], pages=0)
                for t in raw_topics
            ]
            print(f"  Found {len(topics)} topics")

        # Parse syllabus for midterm weight (and date if no midterm file)
        midterm_weight = 0
        syllabus_text = ""
        if "syllabus" in files:
            print("  Parsing syllabus...")
            syllabus_text = extract_pdf_text(str(files["syllabus"]))
            midterm_weight = parse_midterm_weight(client, syllabus_text)
            print(f"  Midterm #1 weight: {midterm_weight}%")

            # Fallback: extract midterm date from syllabus if no midterm file
            if not midterm_date and syllabus_text:
                print("  No midterm file found, extracting date from syllabus...")
                midterm_date = parse_midterm_date_from_syllabus(client, syllabus_text)
                if midterm_date:
                    print(f"  Found midterm date in syllabus: {midterm_date}")
                else:
                    print("  Could not find midterm date in syllabus")

        # Parse textbook for page counts
        total_pages = 0
        if "textbook" in files and chapters_covered:
            print("  Parsing textbook TOC...")
            page_counts = parse_textbook_toc(str(files["textbook"]), chapters_covered)
            print(f"  Page counts: {page_counts}")

            for topic in topics:
                topic.pages = sum(page_counts.get(ch, 0) for ch in topic.chapters)
            total_pages = sum(page_counts.values())

        courses[course_code] = Course(
            name=course_code,
            midterm_date=midterm_date,
            midterm_weight=midterm_weight,
            topics=topics,
            total_pages=total_pages,
        )

    # Store in state
    parser_output = ParserOutput(courses=courses, preferences=UserPreferences())
    tool_context.state["parser_output"] = json.loads(parser_output.to_json())

    # Build summary
    summary = f"Parsed {len(courses)} courses:\n"
    for code, course in courses.items():
        summary += f"  - {code}: {len(course.topics)} topics, exam {course.midterm_date}, weight {course.midterm_weight}%\n"

    return summary


def set_preferences(
    tool_context: ToolContext,
    max_hours_per_day: int = 6,
    preferred_times: str = "morning,afternoon",
    rest_days: str = "",
    study_style: str = "spaced_repetition"
) -> str:
    """Set user study preferences.

    Args:
        max_hours_per_day: Maximum study hours per day (default: 6)
        preferred_times: Comma-separated times: morning,afternoon,evening (default: morning,afternoon)
        rest_days: Comma-separated rest days: Sunday,Saturday (default: none)
        study_style: One of: spaced_repetition, intensive, balanced (default: spaced_repetition)

    Returns:
        Confirmation of preferences
    """
    times_list = [t.strip() for t in preferred_times.split(",") if t.strip()]
    rest_list = [d.strip() for d in rest_days.split(",") if d.strip()]

    preferences = UserPreferences(
        max_hours_per_day=max_hours_per_day,
        preferred_study_times=times_list,
        rest_days=rest_list,
        study_style=study_style,
    )

    # Update state
    parser_output_dict = tool_context.state.get("parser_output", {"courses": {}, "preferences": {}})
    parser_output_dict["preferences"] = asdict(preferences)
    tool_context.state["parser_output"] = parser_output_dict

    return (
        f"Set preferences: max {max_hours_per_day} hrs/day, "
        f"times: {times_list}, rest days: {rest_list}, style: {study_style}"
    )


# Create the Parser Agent
parser_agent = LlmAgent(
    name="parser",
    description="Parses uploaded course files and user preferences",
    instruction="""You are a course file parser. Parse files and set preferences.

STEPS:
1. Call process_uploaded_files() to parse the uploaded PDFs
2. Call set_preferences() with user's preferences (extract from their message, use defaults if not specified)

IMPORTANT: After calling BOTH tools, give a brief summary and STOP.
Do NOT call tools multiple times. Do NOT ask follow-up questions.

Example:
- User says: "Create a study plan. I prefer mornings, max 5 hours, Sundays off"
- You call: process_uploaded_files()
- You call: set_preferences(max_hours_per_day=5, preferred_times="morning", rest_days="Sunday")
- You respond: "Found X courses. Set preferences: 5 hrs/day, mornings, Sundays off. Ready for scheduling."
- Then STOP.""",
    tools=[process_uploaded_files, set_preferences],
)
