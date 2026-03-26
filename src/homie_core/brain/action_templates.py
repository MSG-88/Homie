"""Action-first response templates for Jarvis-like behavior.

Instead of telling the user how to do something, Homie should DO it
when it has the tools. These templates map common intents to tool
sequences and response patterns.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ActionTemplate:
    """Maps an intent to a tool-call sequence."""
    intent: str                    # "check_email", "send_email", "schedule_meeting", etc.
    description: str               # Human-readable description
    required_tools: list[str]      # Tools needed: ["email_search", "email_read"]
    tool_sequence: list[dict]      # Ordered tool calls: [{"tool": "email_search", "args": {...}}]
    confirmation_required: bool = False  # Needs user OK before executing?
    response_template: str = ""    # How to present results


# Pre-built action templates for common Jarvis-like operations
ACTION_TEMPLATES: dict[str, ActionTemplate] = {
    "check_email": ActionTemplate(
        intent="check_email",
        description="Check and summarize unread emails",
        required_tools=["email_unread", "email_summary"],
        tool_sequence=[
            {"tool": "email_summary", "args": {}},
            {"tool": "email_unread", "args": {"group_by": "priority"}},
        ],
        response_template="Here's your email summary:\n{result}",
    ),
    "search_email": ActionTemplate(
        intent="search_email",
        description="Search emails by query",
        required_tools=["email_search"],
        tool_sequence=[
            {"tool": "email_search", "args": {"query": "{query}"}},
        ],
        response_template="Found these emails:\n{result}",
    ),
    "draft_email": ActionTemplate(
        intent="draft_email",
        description="Draft an email",
        required_tools=["email_draft"],
        tool_sequence=[
            {"tool": "email_draft", "args": {"to": "{to}", "subject": "{subject}", "body": "{body}"}},
        ],
        confirmation_required=True,
        response_template="I've drafted this email. Want me to send it?\n{result}",
    ),
    "check_weather": ActionTemplate(
        intent="check_weather",
        description="Get current weather",
        required_tools=["web_search"],
        tool_sequence=[
            {"tool": "web_search", "args": {"query": "current weather {location}"}},
        ],
        response_template="{result}",
    ),
    "git_status": ActionTemplate(
        intent="git_status",
        description="Check git repository status",
        required_tools=["git_status"],
        tool_sequence=[
            {"tool": "git_status", "args": {}},
        ],
        response_template="Git status:\n{result}",
    ),
    "remember_fact": ActionTemplate(
        intent="remember_fact",
        description="Store a fact about the user",
        required_tools=["remember"],
        tool_sequence=[
            {"tool": "remember", "args": {"fact": "{fact}"}},
        ],
        response_template="Got it, I'll remember that. {result}",
    ),
    "recall_info": ActionTemplate(
        intent="recall_info",
        description="Recall stored information",
        required_tools=["recall"],
        tool_sequence=[
            {"tool": "recall", "args": {"query": "{query}"}},
        ],
        response_template="{result}",
    ),
    "system_info": ActionTemplate(
        intent="system_info",
        description="Get system information",
        required_tools=["system_info"],
        tool_sequence=[
            {"tool": "system_info", "args": {}},
        ],
        response_template="System info:\n{result}",
    ),
    "list_files": ActionTemplate(
        intent="list_files",
        description="Search for files",
        required_tools=["find_files"],
        tool_sequence=[
            {"tool": "find_files", "args": {"pattern": "{pattern}"}},
        ],
        response_template="Found:\n{result}",
    ),
    "read_file": ActionTemplate(
        intent="read_file",
        description="Read a file's contents",
        required_tools=["read_file"],
        tool_sequence=[
            {"tool": "read_file", "args": {"path": "{path}"}},
        ],
        response_template="{result}",
    ),
}


def get_intent_keywords() -> dict[str, list[str]]:
    """Map keywords to intents for fast intent detection."""
    return {
        "check_email": ["check email", "check my email", "inbox", "unread", "new emails", "email summary"],
        "search_email": ["search email", "find email", "email from", "email about"],
        "draft_email": ["draft email", "write email", "compose email", "send email", "email to"],
        "check_weather": ["weather", "temperature", "forecast", "rain"],
        "git_status": ["git status", "repo status", "what changed", "uncommitted"],
        "remember_fact": ["remember that", "note that", "keep in mind", "don't forget"],
        "recall_info": ["what do you know about", "recall", "do you remember", "what did i tell you"],
        "system_info": ["system info", "system status", "cpu", "memory usage", "disk space"],
        "list_files": ["find file", "search file", "where is", "list files"],
        "read_file": ["read file", "show file", "open file", "cat file"],
    }


def detect_intent(user_message: str) -> str | None:
    """Detect the user's intent from their message."""
    msg_lower = user_message.lower().strip()
    keywords = get_intent_keywords()

    best_match = None
    best_score = 0

    for intent, kws in keywords.items():
        for kw in kws:
            if kw in msg_lower:
                score = len(kw)  # Longer keyword match = more specific
                if score > best_score:
                    best_score = score
                    best_match = intent

    return best_match


def get_action_template(intent: str) -> ActionTemplate | None:
    """Get the action template for an intent."""
    return ACTION_TEMPLATES.get(intent)
