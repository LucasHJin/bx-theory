import os
import json
from dotenv import load_dotenv

from models import ParserOutput
from parser import run_parser
from scheduler import run_scheduler

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

    schedule = run_scheduler(parser_output)

    print("\n" + "=" * 60)
    print("SCHEDULER OUTPUT")
    print("=" * 60)
    print(json.dumps(schedule, indent=2))


if __name__ == "__main__":
    main()
