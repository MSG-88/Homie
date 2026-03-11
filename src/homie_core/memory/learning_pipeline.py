"""Memory Learning Pipeline — makes Homie learn from every interaction.

Three learning mechanisms:
1. EXPLICIT: User says "remember X" → stored via tool call
2. IMPLICIT: Auto-extract facts from conversation via pattern matching
3. CONSOLIDATION: End-of-session summarization → episodic memory

The pipeline runs lightweight pattern extraction after every response
(no model call needed), and full consolidation at session end.
"""
from __future__ import annotations

import re
from typing import Optional

from homie_core.memory.working import WorkingMemory
from homie_core.memory.semantic import SemanticMemory
from homie_core.memory.episodic import EpisodicMemory


# Patterns that indicate the user is sharing a fact about themselves
_PREFERENCE_PATTERNS = [
    # "I prefer/like/love/hate X"
    re.compile(r"\bi\s+(prefer|like|love|enjoy|hate|dislike|avoid|use|work with|always|never)\s+(.{5,80})", re.IGNORECASE),
    # "my favorite/preferred X is Y"
    re.compile(r"\bmy\s+(favorite|preferred|usual|go-to|default)\s+(\w+)\s+is\s+(.{3,60})", re.IGNORECASE),
    # "I am a/an X" or "I'm a/an X"
    re.compile(r"\bi(?:'m| am)\s+(?:a|an)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    # "I work at/on/with X"
    re.compile(r"\bi\s+work\s+(?:at|on|with|for|in)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    # "My name is X" or "Call me X"
    re.compile(r"(?:my name is|call me|i'm called)\s+(\w+)", re.IGNORECASE),
]

# Patterns indicating correction/update of existing knowledge
_CORRECTION_PATTERNS = [
    re.compile(r"\bactually,?\s+(?:i|my)\s+(.{5,80})", re.IGNORECASE),
    re.compile(r"\bno,?\s+(?:i|my|it's)\s+(.{5,80})", re.IGNORECASE),
    re.compile(r"\bthat's (?:wrong|incorrect|not right|outdated)", re.IGNORECASE),
]

# Things NOT to learn (too vague or transient)
_SKIP_PATTERNS = [
    re.compile(r"^i\s+(want|need|have|think|know|see|feel|guess)\s+", re.IGNORECASE),
    re.compile(r"^i\s+(am|'m)\s+(not sure|confused|wondering|trying)", re.IGNORECASE),
]


class LearningPipeline:
    """Extracts and stores learnable facts from conversations.

    Runs after each user message (lightweight, no model call).
    Stores extracted facts in semantic memory with appropriate confidence.
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
        min_confidence: float = 0.5,
    ):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._min_confidence = min_confidence
        self._session_facts: list[str] = []
        self._interaction_count = 0

    def process_user_message(self, text: str) -> list[str]:
        """Extract and store facts from a user message.

        Returns list of facts that were learned (for transparency).
        """
        self._interaction_count += 1

        if not self._sm:
            return []

        if len(text.strip()) < 10:
            return []

        # Check if this is a correction
        for pattern in _CORRECTION_PATTERNS:
            if pattern.search(text):
                # Don't auto-learn from corrections — the tool call
                # or manual extraction should handle this
                return []

        # Skip vague statements
        for pattern in _SKIP_PATTERNS:
            if pattern.match(text):
                return []

        # Extract learnable facts
        learned = []
        for pattern in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                fact = self._format_fact(match, pattern)
                if fact and not self._already_known(fact):
                    self._sm.learn(fact, confidence=self._min_confidence, tags=["auto_extracted"])
                    self._session_facts.append(fact)
                    learned.append(fact)

        return learned

    def _format_fact(self, match: re.Match, pattern: re.Pattern) -> Optional[str]:
        """Format a regex match into a clean fact string."""
        groups = match.groups()
        full = match.group(0).strip()

        # Clean up the fact
        fact = full.rstrip(".,!?")

        # Normalize to third person for storage
        fact = re.sub(r"^[Ii]\s+", "User ", fact)
        fact = re.sub(r"^[Ii]'m\s+", "User is ", fact)
        fact = re.sub(r"^[Mm]y\s+", "User's ", fact)

        # Skip if too short or too long
        if len(fact) < 10 or len(fact) > 200:
            return None

        return fact

    def _already_known(self, fact: str) -> bool:
        """Check if a similar fact already exists in semantic memory."""
        if not self._sm:
            return False
        existing = self._sm.get_facts(min_confidence=0.0)
        fact_lower = fact.lower()
        for f in existing:
            existing_lower = f["fact"].lower()
            # Simple overlap check — if >60% of words match, it's a duplicate
            fact_words = set(fact_lower.split())
            existing_words = set(existing_lower.split())
            if fact_words and existing_words:
                overlap = len(fact_words & existing_words) / max(len(fact_words), len(existing_words))
                if overlap > 0.6:
                    # Reinforce existing fact instead
                    self._sm.reinforce(f["fact"], boost=0.05)
                    return True
        return False

    def consolidate_session(
        self,
        working_memory: WorkingMemory,
        mood: Optional[str] = None,
        outcome: Optional[str] = None,
    ) -> Optional[str]:
        """Consolidate the current session into episodic memory.

        Called at end of session. Creates a session summary and stores
        it as an episode for future recall.

        Returns the episode summary, or None if nothing to consolidate.
        """
        conversation = working_memory.get_conversation()
        if len(conversation) < 2:
            return None

        # Build a session summary from conversation
        user_msgs = [m["content"] for m in conversation if m["role"] == "user"]
        assistant_msgs = [m["content"] for m in conversation if m["role"] == "assistant"]

        if not user_msgs:
            return None

        # Create a concise summary
        topics = self._extract_topics(user_msgs)
        n_turns = len(user_msgs)
        activity = working_memory.get("activity_type", "general")

        summary_parts = [f"{n_turns}-turn {activity} session"]
        if topics:
            summary_parts.append(f"Topics: {', '.join(topics[:5])}")
        if self._session_facts:
            summary_parts.append(f"Learned: {', '.join(self._session_facts[:3])}")

        summary = ". ".join(summary_parts)

        # Store in episodic memory
        if self._em:
            try:
                context_tags = [activity] + topics[:3]
                self._em.record(
                    summary=summary,
                    mood=mood or working_memory.get("sentiment", "neutral"),
                    outcome=outcome or "completed",
                    context_tags=context_tags,
                )
            except Exception:
                pass

        return summary

    def _extract_topics(self, messages: list[str]) -> list[str]:
        """Extract key topics from user messages using keyword extraction.

        Uses a simple TF approach: find words that appear more than once
        across messages, excluding common stop words.
        """
        stop_words = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
            "have", "has", "had", "do", "does", "did", "will", "would", "could",
            "should", "may", "might", "can", "shall", "to", "of", "in", "for",
            "on", "with", "at", "by", "from", "as", "into", "through", "during",
            "before", "after", "above", "below", "between", "and", "but", "or",
            "not", "no", "so", "if", "then", "than", "too", "very", "just",
            "about", "up", "out", "that", "this", "it", "its", "my", "me", "i",
            "you", "your", "we", "our", "they", "their", "what", "which", "who",
            "how", "when", "where", "why", "all", "each", "every", "both",
            "some", "any", "other", "more", "most", "such", "here", "there",
        }

        word_counts: dict[str, int] = {}
        for msg in messages:
            words = re.findall(r"\w+", msg.lower())
            for word in words:
                if word not in stop_words and len(word) > 2:
                    word_counts[word] = word_counts.get(word, 0) + 1

        # Sort by frequency, return top keywords
        sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
        return [word for word, count in sorted_words[:10] if count >= 1]

    def get_session_stats(self) -> dict:
        """Return statistics about the current session's learning."""
        return {
            "interactions": self._interaction_count,
            "facts_learned": len(self._session_facts),
            "facts": list(self._session_facts),
        }
