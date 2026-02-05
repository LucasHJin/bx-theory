"""ADK Agents for Study Planner.

Multi-agent system that:
1. Parses course files (syllabi, midterm overviews, textbooks)
2. Generates optimized study schedules
3. Validates schedules and outputs CSV

Usage with ADK web:
    adk web

The root_agent is automatically discovered by ADK.
"""

from .agent import root_agent, create_root_agent
from .parser_agent import parser_agent
from .scheduler_agent import scheduler_agent
from .validator_agent import validator_agent

__all__ = [
    "root_agent",
    "create_root_agent",
    "parser_agent",
    "scheduler_agent",
    "validator_agent",
]
