from __future__ import annotations


def estimate_tokens(text: str) -> int:
    return len(text) // 4


def estimate_conversation_tokens(conversation: list[dict]) -> int:
    return sum(estimate_tokens(msg.get("content", "")) for msg in conversation)
