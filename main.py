import os
from dotenv import load_dotenv

from parser import run_parser


def main():
    load_dotenv()

    files_dir = os.path.join(os.path.dirname(__file__), "files")
    user_message = (
        "I can study max 6 hours per day, I prefer studying in the mornings "
        "and afternoons. I'd like Sundays off. I want to use spaced repetition."
    )

    print("=" * 60)
    print("AGENT 1: Parser")
    print("=" * 60)

    result = run_parser(files_dir, user_message)

    print("\n" + "=" * 60)
    print("PARSER OUTPUT")
    print("=" * 60)
    print(result.to_json())


if __name__ == "__main__":
    main()
