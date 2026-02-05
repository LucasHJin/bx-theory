import csv
import io
from datetime import datetime

from .models import ParserOutput


# ---------------------------------------------------------------------------
# Validation checks
# ---------------------------------------------------------------------------

def _check_daily_hours(schedule: list[dict], max_hours: int) -> list[str]:
    """Check 1: Reasonable daily hours."""
    issues = []
    for day in schedule:
        date = day["date"]
        total = sum(s["hours"] for s in day["sessions"])
        if total > 8:
            issues.append(f"ERROR: Day {date} has {total} hours (exceeds 8 hour maximum)")
        elif total > max_hours:
            issues.append(f"WARNING: Day {date} has {total} hours (exceeds user preference of {max_hours} hours)")
    return issues


def _check_topic_coverage(schedule: list[dict], parser_output: ParserOutput) -> list[str]:
    """Check 2: Every required topic has at least one learning session."""
    issues = []

    # Collect scheduled topics per course (learning sessions only)
    scheduled: dict[str, set[str]] = {}
    for day in schedule:
        for s in day["sessions"]:
            course = s["course"]
            if s["type"] == "learning":
                scheduled.setdefault(course, set()).add(s["topic"])

    for code, course in parser_output.courses.items():
        required = {t.name for t in course.topics}
        found = scheduled.get(code, set())
        missing = required - found
        if missing:
            issues.append(f"ERROR: Course {code} missing topics: {sorted(missing)}")

    return issues


def _check_study_before_exam(schedule: list[dict], parser_output: ParserOutput) -> list[str]:
    """Check 3: All sessions before exam; last session close to exam."""
    issues = []

    # Collect all session dates per course
    course_dates: dict[str, list[str]] = {}
    for day in schedule:
        for s in day["sessions"]:
            course_dates.setdefault(s["course"], []).append(day["date"])

    for code, course in parser_output.courses.items():
        # Skip courses without midterm dates (can't validate schedule timing)
        if not course.midterm_date:
            continue

        dates = course_dates.get(code, [])
        if not dates:
            issues.append(f"ERROR: Course {code} has no scheduled sessions")
            continue

        midterm_dt = datetime.strptime(course.midterm_date, "%Y-%m-%d")

        # Check for sessions on or after exam date
        violations = [d for d in dates if datetime.strptime(d, "%Y-%m-%d") >= midterm_dt]
        if violations:
            issues.append(
                f"ERROR: Course {code} has study sessions on/after exam date "
                f"({course.midterm_date}): {sorted(set(violations))}"
            )

        # Check last session proximity
        last_date = max(d for d in dates if datetime.strptime(d, "%Y-%m-%d") < midterm_dt) if \
            any(datetime.strptime(d, "%Y-%m-%d") < midterm_dt for d in dates) else None

        if last_date:
            gap = (midterm_dt - datetime.strptime(last_date, "%Y-%m-%d")).days
            if gap > 3:
                issues.append(
                    f"WARNING: Last study session for {code} is {gap} days before exam"
                )

    return issues


def _check_spaced_repetition(schedule: list[dict], parser_output: ParserOutput) -> list[str]:
    """Check 4: Spaced repetition — learning + review with proper gaps."""
    issues = []

    # Build per-course, per-topic session lists with dates
    topic_sessions: dict[str, dict[str, list[tuple[str, str]]]] = {}  # course -> topic -> [(date, type)]
    for day in schedule:
        for s in day["sessions"]:
            course = s["course"]
            topic = s["topic"]
            topic_sessions.setdefault(course, {}).setdefault(topic, []).append(
                (day["date"], s["type"])
            )

    for code, course in parser_output.courses.items():
        for topic in course.topics:
            sessions = topic_sessions.get(code, {}).get(topic.name, [])

            learning = [(d, t) for d, t in sessions if t == "learning"]
            reviews = [(d, t) for d, t in sessions if t in ("review_1", "review_2")]

            if not learning:
                issues.append(f"ERROR: Topic \"{topic.name}\" ({code}) has no learning session")
                continue

            if not reviews:
                issues.append(
                    f"WARNING: Topic \"{topic.name}\" ({code}) has no review sessions "
                    f"(no spaced repetition)"
                )
                continue

            # Check gap between first learning and first review
            learn_date = datetime.strptime(learning[0][0], "%Y-%m-%d")
            review_date = datetime.strptime(reviews[0][0], "%Y-%m-%d")
            gap = (review_date - learn_date).days

            if gap < 2:
                issues.append(
                    f"WARNING: Review too soon for \"{topic.name}\" ({code}) "
                    f"({gap} day(s) after learning, expected >= 2)"
                )
            elif gap > 7:
                issues.append(
                    f"WARNING: Review too late for \"{topic.name}\" ({code}) "
                    f"({gap} days after learning, expected <= 7)"
                )

    return issues


# ---------------------------------------------------------------------------
# CSV formatting
# ---------------------------------------------------------------------------

NOTES_MAP = {
    "learning": "Initial learning session",
    "review_1": "First review (spaced repetition)",
    "review_2": "Final review before exam",
}


def _format_csv(schedule: list[dict], issues: list[str]) -> str:
    """Format the schedule as CSV with optional validation header."""
    buf = io.StringIO()

    # Write validation issues as comment header
    if issues:
        buf.write("# VALIDATION WARNINGS:\n")
        for issue in issues:
            buf.write(f"# - {issue}\n")
        buf.write("#\n")

    writer = csv.writer(buf)
    writer.writerow(["Date", "Course", "Topic", "Hours", "Type", "Notes"])

    # Sort schedule by date
    sorted_schedule = sorted(schedule, key=lambda d: d["date"])

    for day in sorted_schedule:
        for s in day["sessions"]:
            note = NOTES_MAP.get(s["type"], "")
            writer.writerow([
                day["date"],
                s["course"],
                s["topic"],
                s["hours"],
                s["type"],
                note,
            ])

    return buf.getvalue()


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_validator(parser_output: ParserOutput, schedule: list[dict]) -> tuple[str, list[str], list[str]]:
    """Run all validation checks and format the final CSV.

    Returns (csv_output, errors, warnings).
    """

    all_issues: list[str] = []

    # Check 1: Daily hours
    print("Check 1: Daily hours...")
    issues = _check_daily_hours(schedule, parser_output.preferences.max_hours_per_day)
    all_issues.extend(issues)
    print(f"  {len(issues)} issue(s)")

    # Check 2: Topic coverage
    print("Check 2: Topic coverage...")
    issues = _check_topic_coverage(schedule, parser_output)
    all_issues.extend(issues)
    print(f"  {len(issues)} issue(s)")

    # Check 3: Study before exam
    print("Check 3: Study before exam...")
    issues = _check_study_before_exam(schedule, parser_output)
    all_issues.extend(issues)
    print(f"  {len(issues)} issue(s)")

    # Check 4: Spaced repetition
    print("Check 4: Spaced repetition...")
    issues = _check_spaced_repetition(schedule, parser_output)
    all_issues.extend(issues)
    print(f"  {len(issues)} issue(s)")

    # Separate errors and warnings
    errors = [i for i in all_issues if i.startswith("ERROR")]
    warnings = [i for i in all_issues if i.startswith("WARNING")]
    print(f"\nValidation complete: {len(errors)} error(s), {len(warnings)} warning(s)")

    # Format CSV
    csv_output = _format_csv(schedule, all_issues)
    return csv_output, errors, warnings
