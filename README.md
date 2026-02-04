**Architecture**

SEQUENTIAL
Agent 1 - parser
- INPUT: files (textbook, syllabus, midterm topics) + user message of preferences
    - Parse midterm topics for date, all topics + subtopics (which chapters)
    - Parse syllabus for midterm worth (note -> only parse for midterm #1)
    - Parse textbook for stats per chapter (number of pages)
- OUTPUT: structured output in json
    - For each course:
        - Topic + number of pages per topic
        - Total pages 
        - Percentage worth of midterm
        - Date of midterm
        - *Note* -> only use LLMs for parsing if necessary, don't use LLMs for any decision making
    - Preferences part -> use LLM to decipher user message into staitstical preferences (i.e. max_hours_per_day)
Agent 2 - scheduler
- INPUT: structured JSON from agent 1
    - Feed data to LLM 
        - Determine importance of different classes (based on exam date, difficulty, percentage worth)
        - Generate a schedule day-by-day to balance workload between classes using the above importance
            - Also pay attention to user preference
- OUTPUT: array of daily study sessions in JSON
    - Should include date, course, topic, number of hours, type (learning / review 1 / review 2)
        - Can have multiple topics per day
Agent 3 - validator
- INPUT: outputs from agent 1 and 2
    - Validate amount of hours per day is reasonable (i.e. 8 hours)
    - Valid all topics covered
    - Check for studying before the exam + spaced repetition
- OUTPUT:
    - Final formatted schedule as CSV (Date,Course,Topic,Hours,Type,Notes)