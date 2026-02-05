import os
import re
import json
import time
from pathlib import Path
from datetime import datetime

import pdfplumber
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError

from .models import Topic, Course, UserPreferences, ParserOutput


def get_client() -> genai.Client:
    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable is not set")
    # Disable the SDK's built-in retry to avoid compounding rate-limit hits
    return genai.Client(
        api_key=api_key,
        http_options=types.HttpOptions(
            retryOptions=types.HttpRetryOptions(attempts=1),
        ),
    )


MODEL = "gemini-3-flash-preview"


def _llm_call(client: genai.Client, prompt: str, max_retries: int = 5) -> str:
    """Call Gemini with automatic retry on rate-limit (429) and overload (503) errors.

    The SDK's internal retries are disabled so we control the pacing here.
    """
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.0,
                ),
            )
            return response.text
        except (ClientError, ServerError) as e:
            err_str = str(e)
            is_retryable = "429" in err_str or "503" in err_str
            if is_retryable and attempt < max_retries - 1:
                # Parse retry delay from error if available
                retry_match = re.search(r"retry in ([\d.]+)s", err_str)
                if retry_match:
                    wait = int(float(retry_match.group(1))) + 5
                else:
                    wait = 15
                print(f"    Retryable error, waiting {wait}s before retry...")
                time.sleep(wait)
            else:
                raise
    return ""


# ---------------------------------------------------------------------------
# File classification
# ---------------------------------------------------------------------------

def _classify_file_by_content(client: genai.Client, pdf_path: Path) -> dict | None:
    """Use LLM to classify a PDF by reading its first few pages.

    Returns dict with 'course_code' and 'file_type' or None if can't classify.
    """
    try:
        text = extract_pdf_text(str(pdf_path), max_pages=3)
        if not text or len(text) < 100:
            return None

        # Truncate to avoid token limits
        text = text[:4000]

        prompt = f"""Analyze this document excerpt and identify:
1. The course code (e.g., "PHYS 234", "HLTH 204", "SYSD 300")
2. The document type: "syllabus", "midterm" (midterm overview/study guide), or "textbook"

Return ONLY a JSON object with these keys:
- "course_code": the course code as a string (e.g., "PHYS 234"), or null if this is a textbook
- "file_type": one of "syllabus", "midterm", or "textbook"

If this is a textbook, set course_code to null (we'll match it to a course later).

Document excerpt:
{text}"""

        raw = _llm_call(client, prompt)
        result = json.loads(raw)
        return result
    except Exception as e:
        print(f"  Warning: Could not classify {pdf_path.name}: {e}")
        return None


def classify_files(files_dir: str) -> dict[str, dict[str, Path]]:
    """Group PDF files by course code and type (syllabus/midterm/textbook).

    Works with both named files (e.g., "PHYS 234 - Syllabus.pdf") and
    generic uploaded files (e.g., "uploaded_1.pdf") by using content-based classification.
    """
    courses: dict[str, dict[str, Path]] = {}
    pdf_files = sorted(Path(files_dir).glob("*.pdf"))

    # Check if files have informative names or generic upload names
    has_generic_names = any(
        pdf.name.startswith("uploaded_") or not re.match(r"^[A-Z]{2,5}\s*\d{2,4}", pdf.name)
        for pdf in pdf_files
    )

    # If all files have generic names, use content-based classification
    if has_generic_names and all(
        pdf.name.startswith("uploaded_") or not re.match(r"^[A-Z]{2,5}\s*\d{2,4}", pdf.name)
        for pdf in pdf_files
    ):
        print("Using content-based file classification...")
        client = get_client()
        textbooks: list[Path] = []

        for pdf_path in pdf_files:
            print(f"  Classifying {pdf_path.name}...")
            classification = _classify_file_by_content(client, pdf_path)

            if classification:
                course_code = classification.get("course_code")
                file_type = classification.get("file_type", "").lower()

                if file_type == "textbook" or not course_code:
                    textbooks.append(pdf_path)
                elif course_code:
                    if course_code not in courses:
                        courses[course_code] = {}

                    if file_type == "syllabus":
                        courses[course_code]["syllabus"] = pdf_path
                    elif file_type in ("midterm", "overview"):
                        courses[course_code]["midterm"] = pdf_path
                    print(f"    -> {course_code} ({file_type})")

        # Match textbooks to courses
        if textbooks and courses:
            print("Matching textbooks to courses...")
            for textbook in textbooks:
                best_match = _match_textbook_to_course(client, textbook, courses)
                if best_match and "textbook" not in courses[best_match]:
                    courses[best_match]["textbook"] = textbook
                    print(f"    {textbook.name} -> {best_match}")

        return courses

    # Original filename-based classification
    for pdf_path in pdf_files:
        name = pdf_path.name
        # Extract course code like "PHYS 234" or "HLTH 204"
        code_match = re.match(r"^([A-Z]{2,5}\s*\d{2,4})", name)

        if code_match:
            course_code = code_match.group(1).strip()
            if course_code not in courses:
                courses[course_code] = {}

            lower_name = name.lower()
            if "syllabus" in lower_name:
                courses[course_code]["syllabus"] = pdf_path
            elif "midterm" in lower_name or "overview" in lower_name:
                courses[course_code]["midterm"] = pdf_path
            else:
                courses[course_code]["textbook"] = pdf_path
        else:
            # Textbook files don't start with a course code — match them
            # to courses by checking if any course doesn't have a textbook yet.
            # We'll handle this in a second pass.
            pass

    # Second pass: assign unmatched PDFs (textbooks) to courses using
    # global best-match scoring to avoid cross-matching.
    unmatched = [p for p in pdf_files if not re.match(r"^[A-Z]{2,5}\s*\d{2,4}", p.name)]
    courses_missing_textbook = [c for c, files in courses.items() if "textbook" not in files]

    if unmatched and courses_missing_textbook:
        # Score every (course, textbook) pair, then assign greedily by best score
        scores: list[tuple[int, str, Path]] = []
        for course_code in courses_missing_textbook:
            syllabus_path = courses[course_code].get("syllabus")
            if not syllabus_path:
                continue
            syllabus_text = extract_pdf_text(str(syllabus_path))
            for candidate in unmatched:
                score = _score_textbook_match(syllabus_text, candidate)
                if score > 0:
                    scores.append((score, course_code, candidate))

        # Assign highest-scoring pairs first, each textbook/course used once
        scores.sort(key=lambda x: x[0], reverse=True)
        assigned_courses: set[str] = set()
        assigned_textbooks: set[Path] = set()
        for score, course_code, candidate in scores:
            if course_code not in assigned_courses and candidate not in assigned_textbooks:
                courses[course_code]["textbook"] = candidate
                assigned_courses.add(course_code)
                assigned_textbooks.add(candidate)

    return courses


def _match_textbook_to_course(client: genai.Client, textbook: Path, courses: dict[str, dict[str, Path]]) -> str | None:
    """Use LLM to match a textbook to the most likely course based on syllabus content."""
    # Get textbook info from first few pages
    textbook_text = extract_pdf_text(str(textbook), max_pages=5)[:2000]

    # Build course summaries from syllabi
    course_info = {}
    for code, files in courses.items():
        if "syllabus" in files:
            syllabus_text = extract_pdf_text(str(files["syllabus"]), max_pages=2)[:1500]
            course_info[code] = syllabus_text

    if not course_info:
        return None

    prompt = f"""Match this textbook to the correct course.

TEXTBOOK (first pages):
{textbook_text}

COURSES AND THEIR SYLLABI:
"""
    for code, syllabus in course_info.items():
        prompt += f"\n--- {code} ---\n{syllabus[:800]}\n"

    prompt += f"""

Which course code does this textbook belong to? Return ONLY a JSON object:
{{"course_code": "<CODE>"}}

Choose from: {list(course_info.keys())}"""

    try:
        raw = _llm_call(client, prompt)
        result = json.loads(raw)
        return result.get("course_code")
    except Exception:
        return None


def _score_textbook_match(syllabus_text: str, candidate: Path) -> int:
    """Score how well a textbook PDF filename matches a syllabus.

    Returns the number of distinctive filename words found in the syllabus text.
    """
    syllabus_lower = syllabus_text.lower()
    name_parts = re.split(r"[\s_\-,]+", candidate.stem.lower())
    common_words = {"edition", "higher", "press", "pages", "chapter"}
    return sum(
        1 for part in name_parts
        if len(part) > 4 and part not in common_words and part in syllabus_lower
    )


# ---------------------------------------------------------------------------
# PDF text extraction
# ---------------------------------------------------------------------------

def extract_pdf_text(pdf_path: str, max_pages: int | None = None) -> str:
    """Extract all text from a PDF using pdfplumber."""
    text_parts = []
    with pdfplumber.open(pdf_path) as pdf:
        pages = pdf.pages[:max_pages] if max_pages else pdf.pages
        for page in pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    return "\n\n".join(text_parts)


# ---------------------------------------------------------------------------
# Midterm overview parsing
# ---------------------------------------------------------------------------

def parse_midterm_date(text: str) -> str:
    """Extract midterm date from overview text using regex.

    Args:
        text: The full text content of a midterm overview document

    Returns:
        Date string in YYYY-MM-DD format, or empty string if not found
    """
    match = re.search(r"Date:\s*(.+)", text)
    if match:
        date_str = match.group(1).strip()
        # Parse various date formats
        for fmt in ["%B %d, %Y", "%b %d, %Y", "%Y-%m-%d", "%m/%d/%Y"]:
            try:
                dt = datetime.strptime(date_str, fmt)
                return dt.strftime("%Y-%m-%d")
            except ValueError:
                continue
    return ""


def parse_midterm_chapters(text: str) -> list[str]:
    """Extract chapter numbers from 'Coverage: Chapters ...' line."""
    match = re.search(r"Coverage:\s*Chapters?\s*(.+)", text, re.IGNORECASE)
    if match:
        chapters_str = match.group(1).strip()
        # Extract all numbers (handles "1, 2, 3, 4, 5, 6" and "1, 2, 3, 4, and 5")
        return re.findall(r"\d+", chapters_str)
    return []


def parse_midterm_topics(client: genai.Client, text: str) -> list[dict]:
    """Use LLM to extract structured topics from midterm overview text."""
    prompt = f"""Extract all study topics from this midterm overview document.

For each chapter mentioned, create a topic entry with:
- "name": The chapter title/topic name (e.g., "Stern-Gerlach Experiments & Quantum State Vectors")
- "chapters": A list of chapter numbers as strings (e.g., ["1"])

Return ONLY a JSON array of objects. Example:
[
  {{"name": "Stern-Gerlach Experiments & Quantum State Vectors", "chapters": ["1"]}},
  {{"name": "Operators and Measurement", "chapters": ["2"]}}
]

If a single topic spans multiple chapters (e.g., "Chapters 2 & 3: The Modelling Process"),
combine them into one topic with multiple chapter numbers.

Document:
{text}"""

    raw = _llm_call(client, prompt)
    return json.loads(raw)


# ---------------------------------------------------------------------------
# Syllabus parsing
# ---------------------------------------------------------------------------

def parse_midterm_weight(client: genai.Client, text: str) -> int:
    """Extract midterm #1 weight percentage from syllabus text using LLM."""
    prompt = f"""From this course syllabus, find the weight/percentage of Midterm Examination #1
(the first midterm only, not midterm #2 or the final exam).

Return ONLY a JSON object with a single key "weight" containing the integer percentage.
For example: {{"weight": 25}}

If the syllabus says "midterm #1 15%", return {{"weight": 15}}.
If it says "Midterm Examination 1 (M1) 40%", return {{"weight": 40}}.

Syllabus text:
{text}"""

    raw = _llm_call(client, prompt)
    result = json.loads(raw)
    return int(result.get("weight", 0))


# ---------------------------------------------------------------------------
# Textbook TOC parsing (page counts per chapter)
# ---------------------------------------------------------------------------

def parse_textbook_toc(pdf_path: str, chapters_needed: list[str]) -> dict[str, int]:
    """Extract page counts per chapter from textbook TOC.

    Returns a dict mapping chapter number (str) to page count (int).
    """
    # Extract first 20 pages where TOC typically lives
    toc_text = extract_pdf_text(pdf_path, max_pages=20)

    # Try to find chapter page numbers from TOC
    # Common patterns: "Chapter 1 Title ... 15" or "1 Title 15" or "Chapter 1. Title 15"
    chapter_pages: list[tuple[str, int]] = []

    # Pattern: lines with chapter numbers and page numbers
    # Look for lines like "Chapter N" or "N " followed by a title and ending with a page number
    patterns = [
        # "Chapter 1 Title ... 45"
        r"(?:Chapter|CHAPTER)\s+(\d+)[.\s:].*?(\d+)\s*$",
        # "1 Title 45" at start of line
        r"^(\d+)\s+[A-Z].*?(\d+)\s*$",
        # "1. Title ... 45"
        r"^(\d+)\.\s+.*?(\d+)\s*$",
    ]

    for line in toc_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                ch_num = match.group(1)
                page_num = int(match.group(2))
                # Sanity check: page numbers should be reasonable
                if 1 <= page_num <= 2000:
                    chapter_pages.append((ch_num, page_num))
                break

    # Calculate page counts for needed chapters
    page_counts: dict[str, int] = {}
    if chapter_pages:
        # Sort by page number
        chapter_pages.sort(key=lambda x: x[1])

        for i, (ch_num, start_page) in enumerate(chapter_pages):
            if ch_num in chapters_needed:
                if i + 1 < len(chapter_pages):
                    end_page = chapter_pages[i + 1][1]
                else:
                    # Last chapter — estimate based on total PDF pages
                    end_page = start_page + 30  # default estimate
                page_counts[ch_num] = end_page - start_page

    # If regex-based TOC parsing didn't find all chapters, try LLM fallback
    missing = [ch for ch in chapters_needed if ch not in page_counts]
    if missing and toc_text:
        llm_counts = _parse_toc_with_llm(toc_text, chapters_needed)
        for ch, count in llm_counts.items():
            if ch not in page_counts:
                page_counts[ch] = count

    return page_counts


def _parse_toc_with_llm(toc_text: str, chapters_needed: list[str]) -> dict[str, int]:
    """Fallback: use LLM to extract chapter page ranges from TOC text."""
    client = get_client()
    prompt = f"""From this textbook's table of contents, find the starting page number for each chapter.

I need page counts for these chapters: {chapters_needed}

From the TOC text, identify the page number where each chapter starts.
Then calculate the approximate number of pages per chapter (next chapter start - this chapter start).

Return ONLY a JSON object mapping chapter number (as string) to page count (as integer).
Example: {{"1": 35, "2": 42, "3": 28}}

If you cannot determine the page count for a chapter, use 30 as a default estimate.

Table of Contents text:
{toc_text}"""

    raw = _llm_call(client, prompt)
    result = json.loads(raw)
    return {str(k): int(v) for k, v in result.items()}


# ---------------------------------------------------------------------------
# User preferences parsing
# ---------------------------------------------------------------------------

def parse_user_preferences(client: genai.Client, user_message: str) -> UserPreferences:
    """Use LLM to parse natural language preferences into structured data."""
    prompt = f"""Convert this student's study preferences into structured parameters.

Extract the following (use defaults if not mentioned):
- max_hours_per_day: integer, how many hours max they can study per day (default: 6)
- preferred_study_times: list of strings from ["morning", "afternoon", "evening"] (default: ["morning", "afternoon"])
- rest_days: list of day names like ["Sunday"] (default: [])
- study_style: one of "intensive", "spaced_repetition", "balanced" (default: "spaced_repetition")

Return ONLY a JSON object with these four keys.

Student message:
"{user_message}"
"""

    raw = _llm_call(client, prompt)
    data = json.loads(raw)
    return UserPreferences(
        max_hours_per_day=int(data.get("max_hours_per_day", 6)),
        preferred_study_times=data.get("preferred_study_times", ["morning", "afternoon"]),
        rest_days=data.get("rest_days", []),
        study_style=data.get("study_style", "spaced_repetition"),
    )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_parser(files_dir: str, user_message: str) -> ParserOutput:
    """Run the full parser agent pipeline."""
    client = get_client()

    # Step 1: Classify files by course
    print("Classifying files...")
    course_files = classify_files(files_dir)
    print(f"Found {len(course_files)} courses: {list(course_files.keys())}")

    # Step 2: Parse each course
    courses: dict[str, Course] = {}
    for course_code, files in course_files.items():
        print(f"\nParsing {course_code}...")

        # 2a: Parse midterm overview
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
                Topic(
                    name=t["name"],
                    chapters=t["chapters"],
                    pages=0,  # filled in after textbook parsing
                )
                for t in raw_topics
            ]
            print(f"  Found {len(topics)} topics")

        # 2b: Parse syllabus for midterm weight
        midterm_weight = 0
        if "syllabus" in files:
            print("  Parsing syllabus...")
            syllabus_text = extract_pdf_text(str(files["syllabus"]))
            midterm_weight = parse_midterm_weight(client, syllabus_text)
            print(f"  Midterm #1 weight: {midterm_weight}%")

        # 2c: Parse textbook for page counts
        total_pages = 0
        if "textbook" in files and chapters_covered:
            print("  Parsing textbook TOC...")
            page_counts = parse_textbook_toc(str(files["textbook"]), chapters_covered)
            print(f"  Page counts: {page_counts}")

            # Assign page counts to topics
            for topic in topics:
                topic_pages = sum(page_counts.get(ch, 0) for ch in topic.chapters)
                topic.pages = topic_pages

            total_pages = sum(page_counts.values())
        else:
            if "textbook" not in files:
                print(f"  No textbook found for {course_code}")

        courses[course_code] = Course(
            name=course_code,
            midterm_date=midterm_date,
            midterm_weight=midterm_weight,
            topics=topics,
            total_pages=total_pages,
        )

    # Step 3: Parse user preferences
    print("\nParsing user preferences...")
    preferences = parse_user_preferences(client, user_message)
    print(f"  Preferences: max_hours={preferences.max_hours_per_day}, "
          f"times={preferences.preferred_study_times}, "
          f"rest_days={preferences.rest_days}, "
          f"style={preferences.study_style}")

    return ParserOutput(courses=courses, preferences=preferences)
