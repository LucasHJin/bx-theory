from dataclasses import dataclass, field, asdict
import json


@dataclass
class Topic:
    name: str
    chapters: list[str]
    pages: int = 0


@dataclass
class Course:
    name: str
    midterm_date: str  # "YYYY-MM-DD"
    midterm_weight: int  # Percentage (0-100)
    topics: list[Topic] = field(default_factory=list)
    total_pages: int = 0


@dataclass
class UserPreferences:
    max_hours_per_day: int = 6
    preferred_study_times: list[str] = field(default_factory=lambda: ["morning", "afternoon"])
    rest_days: list[str] = field(default_factory=list)
    study_style: str = "spaced_repetition"


@dataclass
class ParserOutput:
    courses: dict[str, Course] = field(default_factory=dict)
    preferences: UserPreferences = field(default_factory=UserPreferences)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)
