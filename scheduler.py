import json
from dataclasses import asdict
from datetime import datetime, timedelta

from models import ParserOutput
from parser import get_client, _llm_call


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_valid_dates(start: str, end_exclusive: str, rest_days: list[str]) -> list[str]:
    """Return all valid study dates between start and end (exclusive),
    skipping any days whose weekday name is in rest_days."""
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    rest_set = {d.lower() for d in rest_days}
    dates = []
    current = datetime.strptime(start, "%Y-%m-%d")
    end_dt = datetime.strptime(end_exclusive, "%Y-%m-%d")
    while current < end_dt:
        if DAY_NAMES[current.weekday()].lower() not in rest_set:
            dates.append(current.strftime("%Y-%m-%d"))
        current += timedelta(days=1)
    return dates


# ---------------------------------------------------------------------------
# Step 1: Determine course importance / priority
# ---------------------------------------------------------------------------

def determine_priorities(client, parser_output: ParserOutput) -> dict:
    """Use LLM to rank courses by study priority."""
    courses_summary = {}
    for code, course in parser_output.courses.items():
        courses_summary[code] = {
            "midterm_date": course.midterm_date,
            "midterm_weight": course.midterm_weight,
            "total_pages": course.total_pages,
            "num_topics": len(course.topics),
        }

    today = datetime.now().strftime("%Y-%m-%d")
    course_codes = list(courses_summary.keys())

    prompt = f"""You are a study planner. Rank these courses by study priority.

## INPUT
today: "{today}"
courses: {json.dumps(courses_summary)}

## RANKING FACTORS (in order of weight)
1. exam_date_proximity (HIGH) — fewer days until exam = higher priority
2. content_volume (MEDIUM) — more total_pages = needs more time
3. midterm_weight (MEDIUM) — higher % of grade = higher stakes

## OUTPUT SCHEMA
Return exactly this JSON structure. No extra keys, no markdown.
The top-level key MUST be "course_priorities".
Each entry MUST use one of these exact course codes: {json.dumps(course_codes)}

{{
  "course_priorities": {{
    "<COURSE_CODE>": {{
      "priority_score": <float 0.0-10.0>,
      "reasoning": "<1 sentence citing specific numbers>"
    }}
  }}
}}"""

    raw = _llm_call(client, prompt)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Step 2: Generate the daily schedule
# ---------------------------------------------------------------------------

def generate_schedule(client, parser_output: ParserOutput, priorities: dict) -> list[dict]:
    """Use LLM to generate a day-by-day study schedule."""
    today = datetime.now().strftime("%Y-%m-%d")
    start_date = (datetime.now() + timedelta(days=1)).strftime("%Y-%m-%d")

    prefs = asdict(parser_output.preferences)

    # Pre-compute per-course data with strict constraints
    courses_spec = {}
    earliest_exam = None
    latest_exam = None
    for code, course in parser_output.courses.items():
        exam_dt = datetime.strptime(course.midterm_date, "%Y-%m-%d")
        last_study = (exam_dt - timedelta(days=1)).strftime("%Y-%m-%d")
        if earliest_exam is None or course.midterm_date < earliest_exam:
            earliest_exam = course.midterm_date
        if latest_exam is None or course.midterm_date > latest_exam:
            latest_exam = course.midterm_date

        courses_spec[code] = {
            "midterm_date": course.midterm_date,
            "last_valid_study_date": last_study,
            "midterm_weight": course.midterm_weight,
            "total_pages": course.total_pages,
            "topics": [
                {"name": t.name, "pages": t.pages}
                for t in course.topics
            ],
        }

    # Build list of all valid topic names per course for strict validation
    valid_topics = {}
    for code, course in parser_output.courses.items():
        valid_topics[code] = [t.name for t in course.topics]

    # Compute valid scheduling dates
    valid_dates = _get_valid_dates(start_date, latest_exam, prefs.get("rest_days", []))

    prompt = f"""You are a study schedule generator. Produce a day-by-day study plan.

## CONTEXT
today: "{today}"
scheduling_window: "{start_date}" to "{valid_dates[-1]}" (inclusive)

## COURSES
{json.dumps(courses_spec, indent=2)}

## PRIORITIES
{json.dumps(priorities, indent=2)}

## USER PREFERENCES
{json.dumps(prefs)}

## VALID DATES (use ONLY these dates, no others)
{json.dumps(valid_dates)}

## VALID TOPIC NAMES (use ONLY these exact strings for each course)
{json.dumps(valid_topics)}

## STRICT RULES
1. "date" values MUST come from the VALID DATES list above. No other dates allowed.
2. "course" values MUST be one of: {json.dumps(list(courses_spec.keys()))}
3. "topic" values MUST exactly match one of the VALID TOPIC NAMES for that course. Do NOT invent new topic names like "Comprehensive Review" or "Final Review".
4. "type" values MUST be exactly one of: "learning", "review_1", "review_2". No other values.
5. "hours" must be a number > 0, with at most one decimal place.
6. Total hours per day MUST NOT exceed {prefs['max_hours_per_day']}.
7. Every topic for every course MUST appear exactly once with type "learning".
8. Every topic MUST appear at least once with type "review_1", scheduled 3-5 days after its "learning" session.
9. High-page topics (>= 40 pages) should also get a "review_2" session, scheduled closer to the exam.
10. All sessions for a course MUST have a date STRICTLY BEFORE that course's midterm_date (i.e. on or before last_valid_study_date).
11. The last study session for each course should be on or near last_valid_study_date (1-2 days before exam).
12. Learning sessions: 2-4 hours based on topic page count. Review sessions: 0.5-1.5 hours.
13. Spread topics across days. Avoid scheduling all topics for one course on the same day.
14. Higher priority courses get more total study hours.

## OUTPUT FORMAT
Return a JSON array. Each element is a day object. Only include days with sessions.

[
  {{
    "date": "YYYY-MM-DD",
    "sessions": [
      {{
        "course": "COURSE_CODE",
        "topic": "Exact Topic Name",
        "hours": 2.5,
        "type": "learning"
      }}
    ]
  }}
]

Generate the complete schedule now."""

    raw = _llm_call(client, prompt)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_scheduler(parser_output: ParserOutput) -> list[dict]:
    """Run the scheduler agent pipeline."""
    client = get_client()

    # Step 1: Determine priorities
    print("Determining course priorities...")
    priorities = determine_priorities(client, parser_output)

    if "course_priorities" in priorities:
        for code, info in priorities["course_priorities"].items():
            score = info.get("priority_score", "?")
            reasoning = info.get("reasoning", "")
            print(f"  {code}: {score}/10 — {reasoning}")

    # Step 2: Generate schedule
    print("\nGenerating study schedule...")
    schedule = generate_schedule(client, parser_output, priorities)
    print(f"  Generated {len(schedule)} study days")

    # Print summary stats
    total_sessions = sum(len(day["sessions"]) for day in schedule)
    total_hours = sum(
        s["hours"] for day in schedule for s in day["sessions"]
    )
    print(f"  Total sessions: {total_sessions}")
    print(f"  Total study hours: {total_hours:.1f}")

    return schedule
