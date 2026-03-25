"""Context randomizer for generating fictional user contexts.

Produces randomized but deterministic user contexts used to enrich
synthetic training data with diverse scenarios.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field

# ---------------------------------------------------------------------------
# Pools of randomized values
# ---------------------------------------------------------------------------

_NAMES = [
    "Alex", "Jordan", "Morgan", "Casey", "Riley", "Taylor", "Quinn",
    "Avery", "Dakota", "Reese", "Skyler", "Jamie", "Drew", "Hayden",
    "Cameron", "Sage", "Rowan", "Finley", "Emerson", "Parker",
]

_ROLES = [
    "backend developer", "frontend developer", "full-stack developer",
    "data scientist", "ML engineer", "DevOps engineer", "product manager",
    "designer", "QA engineer", "security analyst", "mobile developer",
    "site reliability engineer", "CTO", "tech lead", "student",
]

_PROJECT_NAMES = [
    "web-app", "api-gateway", "ml-pipeline", "mobile-client", "dashboard",
    "auth-service", "data-lake", "chatbot", "cli-tool", "scheduler",
    "analytics-engine", "notification-hub", "search-service", "cms",
    "inventory-tracker", "payment-gateway", "user-portal", "etl-runner",
]

_LANGUAGES = [
    "Python", "TypeScript", "JavaScript", "Rust", "Go", "Java",
    "Kotlin", "Swift", "C++", "Ruby", "Elixir", "Scala",
]

_PROJECT_STATUSES = ["active", "maintenance", "planning", "archived"]

_SERVICES = [
    "GitHub", "Slack", "Jira", "Notion", "Google Calendar", "Gmail",
    "VS Code", "Docker", "AWS", "GCP", "Azure", "Confluence",
    "Linear", "Figma", "Sentry", "Datadog", "PagerDuty", "Vercel",
]

_VERBOSITY = ["concise", "detailed", "moderate"]
_FORMALITY = ["casual", "formal", "neutral"]
_TONE = ["friendly", "professional", "playful", "direct"]
_CODE_STYLE = ["documented", "minimal", "verbose"]

_TIMES_OF_DAY = ["morning", "afternoon", "evening"]

_DAYS_OF_WEEK = [
    "Monday", "Tuesday", "Wednesday", "Thursday",
    "Friday", "Saturday", "Sunday",
]

_ACTIVITIES = [
    "pushed 3 commits to main",
    "reviewed a pull request",
    "closed 2 issues",
    "deployed to staging",
    "merged a feature branch",
    "ran the test suite",
    "updated dependencies",
    "wrote documentation",
    "refactored the auth module",
    "fixed a CI pipeline failure",
    "created a new branch",
    "attended a standup meeting",
    "paired on a bug fix",
    "set up monitoring alerts",
]

_TASKS = [
    "fix flaky integration test",
    "review PR #42",
    "update API documentation",
    "deploy v2.1 to production",
    "migrate database schema",
    "add rate limiting to API",
    "write unit tests for auth module",
    "investigate memory leak",
    "set up CI/CD pipeline",
    "onboard new team member",
    "plan next sprint",
    "upgrade Python to 3.12",
    "add logging to payment service",
    "configure SSL certificates",
]


# ---------------------------------------------------------------------------
# Dataclass
# ---------------------------------------------------------------------------


@dataclass
class FictionalContext:
    """A fictional user context for synthetic data generation."""

    user_name: str
    role: str
    projects: list[dict] = field(default_factory=list)
    preferences: dict[str, str] = field(default_factory=dict)
    connected_services: list[str] = field(default_factory=list)
    time_of_day: str = "morning"
    day_of_week: str = "Monday"
    recent_activity: list[str] = field(default_factory=list)
    pending_tasks: list[str] = field(default_factory=list)

    def to_system_prompt(self) -> str:
        """Build a system prompt describing this fictional user context."""
        lines = [
            f"You are Homie, a personal AI assistant for {self.user_name}, "
            f"a {self.role}.",
            f"It is currently {self.time_of_day} on {self.day_of_week}.",
        ]

        if self.connected_services:
            lines.append(
                f"Connected services: {', '.join(self.connected_services)}."
            )

        if self.preferences:
            pref_parts = [f"{k}: {v}" for k, v in self.preferences.items()]
            lines.append(f"User preferences — {'; '.join(pref_parts)}.")

        if self.projects:
            proj_parts = []
            for p in self.projects:
                proj_parts.append(
                    f"{p['name']} ({p['language']}, {p['status']})"
                )
            lines.append(f"Active projects: {', '.join(proj_parts)}.")

        if self.recent_activity:
            lines.append(
                f"Recent activity: {'; '.join(self.recent_activity)}."
            )

        if self.pending_tasks:
            lines.append(
                f"Pending tasks: {'; '.join(self.pending_tasks)}."
            )

        return "\n".join(lines)

    def to_detail(self) -> str:
        """Return a detail string for template {detail} substitution."""
        if self.projects:
            return self.projects[0]["name"]
        return self.role


# ---------------------------------------------------------------------------
# Randomizer
# ---------------------------------------------------------------------------


class ContextRandomizer:
    """Generate randomized fictional user contexts.

    Parameters
    ----------
    seed:
        Optional seed for deterministic output via ``random.Random``.
    """

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def generate(self) -> FictionalContext:
        """Produce a single :class:`FictionalContext` with random values."""
        num_projects = self._rng.randint(1, 4)
        projects = []
        used_names: set[str] = set()
        for _ in range(num_projects):
            name = self._rng.choice(
                [n for n in _PROJECT_NAMES if n not in used_names] or _PROJECT_NAMES
            )
            used_names.add(name)
            projects.append({
                "name": name,
                "language": self._rng.choice(_LANGUAGES),
                "status": self._rng.choice(_PROJECT_STATUSES),
            })

        num_services = self._rng.randint(2, 6)
        services = self._rng.sample(
            _SERVICES, min(num_services, len(_SERVICES))
        )

        num_activities = self._rng.randint(1, 4)
        activities = self._rng.sample(
            _ACTIVITIES, min(num_activities, len(_ACTIVITIES))
        )

        num_tasks = self._rng.randint(1, 4)
        tasks = self._rng.sample(
            _TASKS, min(num_tasks, len(_TASKS))
        )

        return FictionalContext(
            user_name=self._rng.choice(_NAMES),
            role=self._rng.choice(_ROLES),
            projects=projects,
            preferences={
                "verbosity": self._rng.choice(_VERBOSITY),
                "formality": self._rng.choice(_FORMALITY),
                "tone": self._rng.choice(_TONE),
                "code_style": self._rng.choice(_CODE_STYLE),
            },
            connected_services=services,
            time_of_day=self._rng.choice(_TIMES_OF_DAY),
            day_of_week=self._rng.choice(_DAYS_OF_WEEK),
            recent_activity=activities,
            pending_tasks=tasks,
        )
