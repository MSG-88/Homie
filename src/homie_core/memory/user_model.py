"""Cross-Session User Model — unified profile from all memory sources.

Aggregates facts from SemanticMemory, session patterns from EpisodicMemory,
and behavioral signals into a queryable user profile. The profile is
materialized on-demand and cached briefly to avoid redundant computation.

This replaces scattered fact lookups with a coherent user understanding
that the cognitive architecture can inject into prompts.
"""
from __future__ import annotations

import logging
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)

from homie_core.memory.semantic import SemanticMemory
from homie_core.memory.episodic import EpisodicMemory


# Cache TTL in seconds — user model doesn't change mid-conversation
_CACHE_TTL = 120.0


@dataclass
class UserProfile:
    """Structured user profile synthesized from all memory sources."""

    # Identity
    name: str = ""
    role: str = ""
    location: str = ""

    # Preferences & skills
    preferences: list[str] = field(default_factory=list)
    skills: list[str] = field(default_factory=list)
    tools_and_tech: list[str] = field(default_factory=list)

    # Behavioral patterns
    common_topics: list[str] = field(default_factory=list)
    preferred_hours: list[int] = field(default_factory=list)
    avg_session_turns: float = 0.0
    dominant_mood: str = "neutral"

    # Interaction style
    verbosity_preference: str = "moderate"  # terse / moderate / detailed
    asks_follow_ups: bool = False

    # Expertise tracking
    expertise_level: str = "unknown"  # beginner / intermediate / advanced / expert
    communication_style: str = "balanced"  # formal / casual / balanced / technical

    # Raw facts for prompt injection
    all_facts: list[str] = field(default_factory=list)

    # Synthesis metadata
    last_updated: str = ""
    update_count: int = 0

    def to_context_block(self, max_chars: int = 800) -> str:
        """Render as a structured block for prompt injection."""
        lines: list[str] = []

        if self.name:
            identity = f"User: {self.name}"
            if self.role:
                identity += f" ({self.role})"
            if self.location:
                identity += f", {self.location}"
            lines.append(identity)

        if self.skills:
            lines.append(f"Skills: {', '.join(self.skills[:5])}")

        if self.tools_and_tech:
            lines.append(f"Tech: {', '.join(self.tools_and_tech[:5])}")

        if self.preferences:
            lines.append(f"Preferences: {', '.join(self.preferences[:4])}")

        if self.common_topics:
            lines.append(f"Frequent topics: {', '.join(self.common_topics[:5])}")

        if self.dominant_mood != "neutral":
            lines.append(f"Usual mood: {self.dominant_mood}")

        if self.verbosity_preference != "moderate":
            lines.append(f"Prefers {self.verbosity_preference} responses")

        result = "\n".join(lines)
        if len(result) > max_chars:
            result = result[:max_chars - 3] + "..."
        return result

    def is_empty(self) -> bool:
        return not self.name and not self.all_facts


# ------------------------------------------------------------------
# Fact categorization patterns
# ------------------------------------------------------------------

_NAME_PATTERN = re.compile(
    r"user(?:'s name is| is called| goes by)\s+(\w+)", re.IGNORECASE
)
_ROLE_PATTERN = re.compile(
    r"user\s+is\s+(?:a|an)\s+(.{3,50}?)(?:\.|$)", re.IGNORECASE
)
_LOCATION_PATTERN = re.compile(
    r"user\s+(?:lives? in|is (?:from|based in|located in))\s+(.{3,40}?)(?:\.|$)",
    re.IGNORECASE,
)
_SKILL_PATTERN = re.compile(
    r"user\s+(?:knows?|speaks?|studies|learned|practices?|has expertise in)\s+(.{3,50}?)(?:\.|$)",
    re.IGNORECASE,
)
_TECH_PATTERN = re.compile(
    r"user\s+(?:uses?|works? with|prefers?|switched to)\s+(.{3,50}?)(?:\.|$)",
    re.IGNORECASE,
)
_PREFERENCE_PATTERN = re.compile(
    r"user\s+(?:prefers?|likes?|loves?|enjoys?|hates?|dislikes?|avoids?)\s+(.{3,60}?)(?:\.|$)",
    re.IGNORECASE,
)


def _categorize_fact(fact_text: str) -> tuple[str, str]:
    """Categorize a fact into (category, extracted_value)."""
    for pattern, category in [
        (_NAME_PATTERN, "name"),
        (_ROLE_PATTERN, "role"),
        (_LOCATION_PATTERN, "location"),
        (_SKILL_PATTERN, "skill"),
        (_TECH_PATTERN, "tech"),
        (_PREFERENCE_PATTERN, "preference"),
    ]:
        m = pattern.search(fact_text)
        if m:
            return category, m.group(1).strip()
    return "other", fact_text


class UserModelSynthesizer:
    """Builds a unified user profile from all memory sources.

    Materializes the profile on-demand and caches it briefly.
    Only queries memory stores — never writes.
    """

    def __init__(
        self,
        semantic_memory: Optional[SemanticMemory] = None,
        episodic_memory: Optional[EpisodicMemory] = None,
    ):
        self._sm = semantic_memory
        self._em = episodic_memory
        self._cache: Optional[UserProfile] = None
        self._cache_time: float = 0.0

    def get_profile(self, force_refresh: bool = False) -> UserProfile:
        """Get the current user profile, using cache if fresh."""
        now = time.monotonic()
        if (
            not force_refresh
            and self._cache is not None
            and (now - self._cache_time) < _CACHE_TTL
        ):
            return self._cache

        profile = self._build_profile()
        self._cache = profile
        self._cache_time = now
        return profile

    def get_relevant_context(self, query: str, max_chars: int = 600) -> str:
        """Get user context relevant to a specific query.

        Returns a formatted string for prompt injection, filtered
        by relevance to the query.
        """
        profile = self.get_profile()
        if profile.is_empty():
            return ""

        # For now, return the full context block (already concise)
        # Future: TF-IDF filter facts by query relevance
        return profile.to_context_block(max_chars=max_chars)

    def _build_profile(self) -> UserProfile:
        """Synthesize profile from semantic + episodic memory."""
        profile = UserProfile()

        # 1. Extract structured info from semantic facts
        if self._sm:
            try:
                facts = self._sm.get_facts(min_confidence=0.3)
                profile.all_facts = [f["fact"] for f in facts]
                self._extract_from_facts(profile, facts)
            except Exception as e:
                logger.warning("Failed to extract user profile from semantic memory: %s", e)

        # 2. Extract patterns from episodic memory
        if self._em:
            try:
                self._extract_from_episodes(profile)
            except Exception as e:
                logger.warning("Failed to extract user profile from episodic memory: %s", e)

        # 3. Infer interaction preferences from facts
        self._infer_preferences(profile)

        return profile

    def _extract_from_facts(
        self, profile: UserProfile, facts: list[dict]
    ) -> None:
        """Categorize facts into profile fields."""
        for fact in facts:
            category, value = _categorize_fact(fact["fact"])

            if category == "name" and not profile.name:
                profile.name = value
            elif category == "role" and not profile.role:
                profile.role = value
            elif category == "location" and not profile.location:
                profile.location = value
            elif category == "skill":
                profile.skills.append(value)
            elif category == "tech":
                profile.tools_and_tech.append(value)
            elif category == "preference":
                profile.preferences.append(value)

        # Deduplicate
        profile.skills = list(dict.fromkeys(profile.skills))
        profile.tools_and_tech = list(dict.fromkeys(profile.tools_and_tech))
        profile.preferences = list(dict.fromkeys(profile.preferences))

    def _extract_from_episodes(self, profile: UserProfile) -> None:
        """Extract behavioral patterns from episodic memory."""
        try:
            # Get recent episodes for pattern analysis
            episodes = self._em.recall("", n=50)
        except Exception:
            return

        if not episodes:
            return

        # Topic frequency
        topic_counts: Counter = Counter()
        mood_counts: Counter = Counter()
        turn_counts: list[int] = []

        for ep in episodes:
            summary = ep.get("summary", "")
            mood = ep.get("mood", "neutral")

            mood_counts[mood] += 1

            # Extract turn count from summary like "5-turn coding session"
            turn_match = re.search(r"(\d+)-turn", summary)
            if turn_match:
                turn_counts.append(int(turn_match.group(1)))

            # Extract topics from context tags
            tags = ep.get("context_tags", [])
            if isinstance(tags, list):
                for tag in tags:
                    if tag and len(tag) > 2:
                        topic_counts[tag] += 1

        # Populate profile
        profile.common_topics = [
            topic for topic, _ in topic_counts.most_common(10)
        ]

        if mood_counts:
            profile.dominant_mood = mood_counts.most_common(1)[0][0]

        if turn_counts:
            profile.avg_session_turns = sum(turn_counts) / len(turn_counts)

    def _infer_preferences(self, profile: UserProfile) -> None:
        """Infer interaction preferences from accumulated signals."""
        # Verbosity: if user has many technical skills, they probably
        # prefer concise responses
        if len(profile.skills) > 3 or len(profile.tools_and_tech) > 3:
            profile.verbosity_preference = "terse"
        elif not profile.skills and not profile.tools_and_tech:
            profile.verbosity_preference = "detailed"

        # Expertise level: infer from skills and tech stack breadth
        total_signals = len(profile.skills) + len(profile.tools_and_tech)
        if total_signals >= 8:
            profile.expertise_level = "expert"
        elif total_signals >= 5:
            profile.expertise_level = "advanced"
        elif total_signals >= 2:
            profile.expertise_level = "intermediate"
        elif total_signals >= 1:
            profile.expertise_level = "beginner"

        # Communication style: infer from facts
        fact_text = " ".join(profile.all_facts).lower()
        if any(w in fact_text for w in ["formal", "professional", "business"]):
            profile.communication_style = "formal"
        elif any(w in fact_text for w in ["casual", "chill", "relaxed"]):
            profile.communication_style = "casual"
        elif any(w in fact_text for w in ["technical", "code", "engineer", "developer"]):
            profile.communication_style = "technical"

    def incremental_update(self, new_facts: list[str]) -> UserProfile:
        """Incrementally update the user profile with newly learned facts.

        More efficient than a full rebuild — only processes new facts
        against the existing profile and merges changes.

        Args:
            new_facts: List of fact strings just learned from conversation.

        Returns:
            Updated UserProfile.
        """
        profile = self.get_profile()

        for fact_text in new_facts:
            category, value = _categorize_fact(fact_text)

            if category == "name" and not profile.name:
                profile.name = value
            elif category == "role":
                profile.role = value  # Role can be updated
            elif category == "location":
                profile.location = value  # Location can be updated
            elif category == "skill" and value not in profile.skills:
                profile.skills.append(value)
            elif category == "tech" and value not in profile.tools_and_tech:
                profile.tools_and_tech.append(value)
            elif category == "preference" and value not in profile.preferences:
                profile.preferences.append(value)

            if fact_text not in profile.all_facts:
                profile.all_facts.append(fact_text)

        # Re-infer preferences with updated data
        self._infer_preferences(profile)

        profile.update_count += 1
        profile.last_updated = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

        # Update cache
        self._cache = profile
        self._cache_time = time.monotonic()

        # Persist to semantic memory profile store
        self._persist_profile(profile)

        return profile

    def _persist_profile(self, profile: UserProfile) -> None:
        """Persist the synthesized profile to semantic memory's profile store."""
        if not self._sm:
            return
        try:
            data = {
                "name": profile.name,
                "role": profile.role,
                "location": profile.location,
                "preferences": profile.preferences[:20],
                "skills": profile.skills[:20],
                "tools_and_tech": profile.tools_and_tech[:20],
                "common_topics": profile.common_topics[:10],
                "expertise_level": profile.expertise_level,
                "communication_style": profile.communication_style,
                "verbosity_preference": profile.verbosity_preference,
                "dominant_mood": profile.dominant_mood,
                "update_count": profile.update_count,
                "last_updated": profile.last_updated,
            }
            self._sm.set_profile("user_model", data)
        except Exception:
            pass

    def load_persisted_profile(self) -> Optional[UserProfile]:
        """Load previously persisted profile from semantic memory.

        Returns None if no persisted profile exists.
        """
        if not self._sm:
            return None
        try:
            data = self._sm.get_profile("user_model")
            if not data:
                return None
            profile = UserProfile(
                name=data.get("name", ""),
                role=data.get("role", ""),
                location=data.get("location", ""),
                preferences=data.get("preferences", []),
                skills=data.get("skills", []),
                tools_and_tech=data.get("tools_and_tech", []),
                common_topics=data.get("common_topics", []),
                expertise_level=data.get("expertise_level", "unknown"),
                communication_style=data.get("communication_style", "balanced"),
                verbosity_preference=data.get("verbosity_preference", "moderate"),
                dominant_mood=data.get("dominant_mood", "neutral"),
                update_count=data.get("update_count", 0),
                last_updated=data.get("last_updated", ""),
            )
            return profile
        except Exception:
            return None

    def invalidate_cache(self) -> None:
        """Force next get_profile() to rebuild."""
        self._cache = None
        self._cache_time = 0.0
