import os
import json
from dotenv import load_dotenv

from models import ParserOutput
from parser import run_parser
from scheduler import run_scheduler, run_scheduler_retry
from validator import run_validator

CACHE_FILE = "parser_output.json"


def main():
    load_dotenv()

    files_dir = os.path.join(os.path.dirname(__file__), "files")
    user_message = (
        "I can study max 6 hours per day, I prefer studying in the mornings "
        "and afternoons. I'd like Sundays off. I want to use spaced repetition."
    )

    # AGENT 1: Parser (skip if cached output exists)
    if os.path.exists(CACHE_FILE):
        print("=" * 60)
        print("AGENT 1: Parser (loading from cache)")
        print("=" * 60)
        with open(CACHE_FILE) as f:
            parser_output = ParserOutput.from_json(f.read())
        print(f"Loaded {len(parser_output.courses)} courses from {CACHE_FILE}")
    else:
        print("=" * 60)
        print("AGENT 1: Parser")
        print("=" * 60)

        parser_output = run_parser(files_dir, user_message)

        # Cache for subsequent runs
        with open(CACHE_FILE, "w") as f:
            f.write(parser_output.to_json())
        print(f"\nCached parser output to {CACHE_FILE}")

    print("\n" + "=" * 60)
    print("PARSER OUTPUT")
    print("=" * 60)
    print(parser_output.to_json())

    # AGENT 2: Scheduler
    print("\n" + "=" * 60)
    print("AGENT 2: Scheduler")
    print("=" * 60)

    schedule, priorities = run_scheduler(parser_output)

    # AGENT 3 + retry loop: Validate → feed errors back → regenerate
    MAX_RETRIES = 3

    for attempt in range(1, MAX_RETRIES + 1):
        print("\n" + "=" * 60)
        print(f"AGENT 3: Validator & Formatter (attempt {attempt}/{MAX_RETRIES})")
        print("=" * 60)

        csv_output, errors, warnings = run_validator(parser_output, schedule)

        if not errors:
            print("\nNo errors found — schedule is valid.")
            break

        if attempt < MAX_RETRIES:
            print(f"\n{len(errors)} error(s) found. Feeding back to scheduler...")
            for e in errors:
                print(f"  {e}")

            print("\n" + "=" * 60)
            print(f"AGENT 2: Scheduler (retry {attempt})")
            print("=" * 60)

            schedule = run_scheduler_retry(
                parser_output, priorities, schedule, errors
            )
        else:
            print(f"\n{len(errors)} error(s) remain after {MAX_RETRIES} attempts.")
            print("Outputting best schedule with warnings.")

    # Write CSV to file
    output_file = "study_plan.csv"
    with open(output_file, "w") as f:
        f.write(csv_output)
    print(f"\nStudy plan written to {output_file}")

    print("\n" + "=" * 60)
    print("FINAL CSV OUTPUT")
    print("=" * 60)
    print(csv_output)


if __name__ == "__main__":
    main()
