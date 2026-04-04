"""Memory Learning Pipeline — makes Homie learn from every interaction.

Three learning mechanisms:
1. EXPLICIT: User says "remember X" → stored via tool call
2. IMPLICIT: Auto-extract facts from conversation via pattern matching
3. CONSOLIDATION: End-of-session summarization → episodic memory

The pipeline runs lightweight pattern extraction after every response
(no model call needed), and full consolidation at session end.
"""
from __future__ import annotations

import logging
import re
from typing import Optional

logger = logging.getLogger(__name__)

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
    # "I usually/typically/normally X"
    re.compile(r"\bi\s+(?:usually|typically|normally|tend to|often)\s+(.{5,80})", re.IGNORECASE),
    # "I've been X-ing for Y" (experience)
    re.compile(r"\bi(?:'ve| have)\s+been\s+(\w+ing)\s+(?:for|since)\s+(.{3,40})", re.IGNORECASE),
    # "I speak/know/study X" (skills/languages)
    re.compile(r"\bi\s+(?:speak|know|study|learned|learn|practice)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    # "I live in X" / "I'm from X" / "I'm based in X"
    re.compile(r"\bi(?:'m| am)\s+(?:from|based in|living in|located in)\s+(.{3,40}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    re.compile(r"\bi\s+live\s+in\s+(.{3,40}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    # "I'm interested in X" / "I'm passionate about X"
    re.compile(r"\bi(?:'m| am)\s+(?:interested in|passionate about|into|focused on)\s+(.{3,60}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
    # "My X is Y" (possessives)
    re.compile(r"\bmy\s+(dog|cat|pet|car|phone|laptop|os|editor|ide|language|framework|stack|setup)\s+is\s+(.{3,40})", re.IGNORECASE),
    # "I switched from X to Y" / "I moved from X to Y"
    re.compile(r"\bi\s+(?:switched|moved|migrated|transitioned)\s+(?:from\s+\w+\s+)?to\s+(.{3,40}?)(?:\.|,|!|\?|$)", re.IGNORECASE),
]

# Patterns for detecting project names in conversation
_PROJECT_PATTERNS = [
    # "working on project X" / "the X project"
    re.compile(r"\b(?:working on|building|developing|the)\s+(?:project\s+)?([A-Z][\w-]{2,30})\s+(?:project|app|service|api|tool|system|platform)", re.IGNORECASE),
    re.compile(r"\bproject\s+([A-Z][\w-]{2,30})\b", re.IGNORECASE),
    # "for X" at end of technical statement
    re.compile(r"\b(?:implementing|designing|fixing|adding|updating)\s+.{5,40}\s+(?:for|in)\s+([A-Z][\w-]{2,30})\b"),
    # "let's continue with X"
    re.compile(r"\b(?:continue|resume|back to|switch to|work on)\s+([A-Z][\w-]{2,30})\b", re.IGNORECASE),
]

# Patterns indicating correction/update of existing knowledge
_CORRECTION_PATTERNS = [
    re.compile(r"\bactually,?\s+(?:i|my)\s+(.{5,80})", re.IGNORECASE),
    re.compile(r"\bno,?\s+(?:i|my|it's)\s+(.{5,80})", re.IGNORECASE),
    re.compile(r"\bthat's (?:wrong|incorrect|not right|outdated)", re.IGNORECASE),
    re.compile(r"\bnot anymore", re.IGNORECASE),
    re.compile(r"\bi\s+(?:stopped|quit|no longer|don't|don't)\s+(.{5,60})", re.IGNORECASE),
]

# Things NOT to learn (too vague or transient)
_SKIP_PATTERNS = [
    re.compile(r"^i\s+(want|need|have|think|know|see|feel|guess)\s+", re.IGNORECASE),
    re.compile(r"^i\s+(am|'m)\s+(not sure|confused|wondering|trying)", re.IGNORECASE),
    re.compile(r"^i\s+(just|was just|am just)\s+", re.IGNORECASE),
    re.compile(r"^can\s+(you|i)\s+", re.IGNORECASE),
]


def _tokenize_simple(text: str) -> list[str]:
    """Simple word tokenizer for overlap comparison."""
    return re.findall(r"\w+", text.lower())


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
        self._active_project: Optional[str] = None
        self._project_history: list[str] = []

    def process_user_message(self, text: str) -> list[str]:
        """Extract and store facts from a user message.

        Returns list of facts that were learned (for transparency).
        """
        self._interaction_count += 1

        # Detect project context from message
        self._detect_project(text)

        if not self._sm:
            return []

        if len(text.strip()) < 10:
            return []

        # Handle corrections — update existing facts rather than ignoring
        for pattern in _CORRECTION_PATTERNS:
            match = pattern.search(text)
            if match:
                correction = self._handle_correction(text, match)
                return correction

        # Skip vague statements
        for pattern in _SKIP_PATTERNS:
            if pattern.match(text):
                return []

        # Build tags: auto_extracted + project tag if active
        tags = ["auto_extracted"]
        if self._active_project:
            tags.append(f"project:{self._active_project}")

        # Extract learnable facts
        learned = []
        for pattern in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if match:
                fact = self._format_fact(match, pattern)
                if fact and not self._already_known(fact):
                    self._sm.learn(fact, confidence=self._min_confidence, tags=tags)
                    self._session_facts.append(fact)
                    learned.append(fact)

        return learned

    def _detect_project(self, text: str) -> None:
        """Detect and track project names mentioned in conversation."""
        for pattern in _PROJECT_PATTERNS:
            match = pattern.search(text)
            if match:
                project = match.group(1).strip()
                # Filter out common false positives
                if project.lower() in {"the", "this", "that", "my", "our", "your", "some"}:
                    continue
                if len(project) < 3:
                    continue
                self._active_project = project
                if project not in self._project_history:
                    self._project_history.append(project)
                break

    @property
    def active_project(self) -> Optional[str]:
        """Currently detected project context."""
        return self._active_project

    def recall_project(self, project_name: str) -> dict:
        """Recall all facts and episodes tagged with a project.

        Returns dict with 'facts' and 'episodes' keys.
        """
        result: dict = {"facts": [], "episodes": []}

        if self._sm:
            try:
                all_facts = self._sm.get_facts(min_confidence=0.0)
                tag = f"project:{project_name}"
                result["facts"] = [
                    f for f in all_facts
                    if tag in f.get("tags", [])
                ]
            except Exception as e:
                logger.warning("Failed to retrieve project facts for '%s': %s", project_name, e)

        if self._em:
            try:
                episodes = self._em.recall(project_name, n=20)
                result["episodes"] = [
                    ep for ep in episodes
                    if project_name.lower() in " ".join(
                        ep.get("context_tags", [])
                    ).lower()
                ]
            except Exception as e:
                logger.warning("Failed to retrieve project episodes for '%s': %s", project_name, e)

        return result

    def _handle_correction(self, text: str, match: re.Match) -> list[str]:
        """Handle a correction — try to find and update the contradicted fact."""
        if not self._sm:
            return []

        # Extract the correction content
        correction_text = match.group(0).strip()
        correction_text = re.sub(r"^(actually,?\s*|no,?\s*)", "", correction_text, flags=re.IGNORECASE).strip()

        if len(correction_text) < 10:
            return []

        # Find existing facts that might be contradicted
        existing = self._sm.get_facts(min_confidence=0.0)
        correction_words = set(_tokenize_simple(correction_text))

        for f in existing:
            fact_words = set(_tokenize_simple(f["fact"]))
            # If there's topic overlap, this might be a correction
            overlap = len(correction_words & fact_words)
            if overlap >= 2:
                # Reduce confidence of the old fact
                self._sm._db._conn.execute(
                    "UPDATE semantic_memory SET confidence = MAX(0.1, confidence - 0.3) WHERE id = ?",
                    (f["id"],),
                )
                self._sm._db._conn.commit()

        # Store the correction as a new fact with high confidence
        formatted = re.sub(r"^[Ii]\s+", "User ", correction_text)
        formatted = re.sub(r"^[Ii]'m\s+", "User is ", formatted)
        formatted = re.sub(r"^[Mm]y\s+", "User's ", formatted)
        formatted = formatted.rstrip(".,!?")

        if len(formatted) >= 10 and not self._already_known(formatted):
            self._sm.learn(formatted, confidence=0.8, tags=["correction"])
            self._session_facts.append(formatted)
            return [formatted]

        return []

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
                # Add project tags for cross-session continuity
                for project in self._project_history:
                    context_tags.append(f"project:{project}")
                self._em.record(
                    summary=summary,
                    mood=mood or working_memory.get("sentiment", "neutral"),
                    outcome=outcome or "completed",
                    context_tags=context_tags,
                )
            except Exception as e:
                logger.warning("Failed to record session episode in episodic memory: %s", e)

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
