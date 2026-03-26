"""Response conciseness optimizer — matches depth to question complexity.

Analyzes the user's question to determine expected response length,
then provides guidance to the model via system prompt injection.
"""
from __future__ import annotations

import re
from enum import Enum


class ResponseDepth(str, Enum):
    BRIEF = "brief"        # 1-3 sentences: greetings, yes/no, simple facts
    MODERATE = "moderate"   # 1-2 paragraphs: explanations, summaries
    DETAILED = "detailed"   # Full structured response: tutorials, plans, code


# Patterns that indicate brief responses are appropriate
_BRIEF_PATTERNS = [
    r"^(hi|hey|hello|good\s+(morning|afternoon|evening)|thanks|thank you|bye|goodbye)",
    r"^(yes|no|ok|okay|sure|got it|understood)",
    r"^what\s+time",
    r"^how\s+are\s+you",
    r"^what.s\s+(your\s+name|up)",
]

# Patterns that indicate detailed responses
_DETAILED_PATTERNS = [
    r"(write|create|build|implement|design|develop)\s+(a|an|the|me)",
    r"(explain|describe|compare|analyze|review)\s+.{10,}",
    r"(help\s+me\s+(prepare|plan|draft|build|write|create))",
    r"(step.by.step|tutorial|guide|roadmap|outline)",
    r"(code|function|class|script|dockerfile|sql|api)",
]


def detect_depth(user_message: str) -> ResponseDepth:
    """Detect the appropriate response depth for a user message."""
    msg = user_message.strip().lower()

    # Check brief patterns
    for pattern in _BRIEF_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return ResponseDepth.BRIEF

    # Check detailed patterns
    for pattern in _DETAILED_PATTERNS:
        if re.search(pattern, msg, re.IGNORECASE):
            return ResponseDepth.DETAILED

    # Word count heuristic
    words = len(msg.split())
    if words <= 4:
        return ResponseDepth.BRIEF
    if words >= 15:
        return ResponseDepth.DETAILED

    return ResponseDepth.MODERATE


def get_depth_instruction(depth: ResponseDepth) -> str:
    """Get a system prompt instruction for the target response depth."""
    instructions = {
        ResponseDepth.BRIEF: "Respond in 1-3 sentences. Be warm but brief. No bullet points or headers needed.",
        ResponseDepth.MODERATE: "Respond in 1-2 short paragraphs. Use bullets if it helps clarity. Stay focused.",
        ResponseDepth.DETAILED: "Provide a thorough, structured response. Use headers, bullets, code blocks as needed. Be comprehensive but avoid padding.",
    }
    return instructions[depth]


def optimize_prompt_for_conciseness(user_message: str, system_prompt: str) -> str:
    """Inject conciseness guidance into the system prompt based on the user's message."""
    depth = detect_depth(user_message)
    instruction = get_depth_instruction(depth)
    return f"{system_prompt}\n\n[Response guidance: {instruction}]"
