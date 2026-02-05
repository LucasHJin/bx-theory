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
    max_hours_per_day: int | None = None
    preferred_study_times: list[str] = field(default_factory=lambda: ["morning", "afternoon"])
    rest_days: list[str] = field(default_factory=list)
    study_style: str = "spaced_repetition"


@dataclass
class ParserOutput:
    courses: dict[str, Course] = field(default_factory=dict)
    preferences: UserPreferences = field(default_factory=UserPreferences)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)

    @classmethod
    def from_json(cls, json_str: str) -> "ParserOutput":
        data = json.loads(json_str)
        courses = {}
        for code, c in data["courses"].items():
            topics = [Topic(**t) for t in c["topics"]]
            courses[code] = Course(
                name=c["name"],
                midterm_date=c["midterm_date"],
                midterm_weight=c["midterm_weight"],
                topics=topics,
                total_pages=c["total_pages"],
            )
        prefs = UserPreferences(**data["preferences"])
        return cls(courses=courses, preferences=prefs)
