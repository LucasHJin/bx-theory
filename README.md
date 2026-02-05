# Exam Study Planner

A multi-agent system built with Google ADK that generates personalized study schedules from course materials.

## How To

Upload your course files (syllabi, midterm overviews, textbooks) and the system will:
1. Extract exam dates, topics, and content volume from PDFs
2. Generate an optimized day-by-day study schedule with spaced repetition
3. Validate the schedule and output as CSV

## Architecture

```
root_agent (SequentialAgent)
├── parser_agent (LlmAgent)
│   └── tools: process_uploaded_files, set_preferences
└── schedule_loop (LoopAgent, max_iterations=3)
    ├── scheduler_agent (LlmAgent)
    │   └── tools: determine_priorities, generate_schedule
    └── validator_agent (LlmAgent)
        └── tools: validate_and_format
```

### Agent 1 - Parser
**Input:** PDF files (textbook, syllabus, midterm overview) + user preferences

**Processing:**
- Classify files by course and type (two-pass LLM classification)
- Parse midterm overview for date, chapters, topics
- Parse syllabus for midterm weight
- Parse textbook TOC for page counts per chapter
- Extract user preferences (max_hours_per_day, rest_days, study_style)

**Output:** `state["parser_output"]` containing courses and preferences

### Agent 2 - Scheduler
**Input:** Parsed course data from state

**Processing:**
- Determine course priorities (exam proximity, content volume, grade weight)
- Generate day-by-day schedule with learning + review sessions
- Apply spaced repetition (review_1 after 3-5 days, review_2 near exam)

**Output:** `state["schedule"]` array of daily study sessions

### Agent 3 - Validator
**Input:** Parser output + schedule from state

**Validation checks:**
- Daily hours reasonable (error if >8h, warning if >user max)
- All topics have learning sessions
- All sessions before exam date
- Spaced repetition properly applied

**Output:** `study_plan.csv` with format: Date, Course, Topic, Hours, Type, Notes

If validation fails, the LoopAgent retries (up to 3 times) with error feedback.
