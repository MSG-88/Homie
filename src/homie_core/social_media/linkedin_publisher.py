"""LinkedIn AI Content Publisher — research, generate, and publish AI/LLM posts.

Researches trending topics in AI/ML/LLM space, generates professional
LinkedIn posts tailored to the user's profile, and publishes with approval.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from typing import Callable, Optional

import requests

logger = logging.getLogger(__name__)


@dataclass
class PostDraft:
    """A generated LinkedIn post draft."""
    topic: str
    title: str
    content: str            # Full post text (LinkedIn max 3000 chars)
    hashtags: list[str]
    category: str           # "thought_leadership", "tutorial", "news", "opinion", "case_study"
    estimated_engagement: str = "medium"  # "low", "medium", "high"
    sources: list[str] = field(default_factory=list)  # URLs or references
    created_at: float = 0.0

    def to_linkedin_text(self) -> str:
        """Format for LinkedIn posting."""
        text = self.content
        if self.hashtags:
            text += "\n\n" + " ".join(f"#{h}" for h in self.hashtags)
        return text[:3000]  # LinkedIn character limit


# Trending AI/LLM topics to rotate through
_AI_TOPICS = [
    "Local LLMs and privacy-first AI assistants",
    "Fine-tuning open-source models for enterprise use",
    "Agentic AI: when models use tools autonomously",
    "RAG (Retrieval-Augmented Generation) best practices",
    "Small language models vs large: when less is more",
    "AI in software engineering: beyond code completion",
    "Responsible AI development in production systems",
    "The rise of on-device AI: running models on edge",
    "Multi-modal AI: combining text, vision, and audio",
    "Open-source AI ecosystem: Hugging Face, Ollama, and beyond",
    "LLM evaluation and benchmarking strategies",
    "AI agents in enterprise workflows",
    "Vector databases and semantic search in production",
    "Prompt engineering: from craft to science",
    "Building AI products that users actually want",
    "The economics of running AI locally vs cloud",
    "Knowledge graphs meet LLMs: structured + unstructured intelligence",
    "AI safety and alignment in practical applications",
    "From chatbot to companion: the evolution of AI assistants",
    "Developer experience in the age of AI-assisted coding",
]


class LinkedInPublisher:
    """Research, generate, and publish AI/LLM content on LinkedIn.

    Usage:
        publisher = LinkedInPublisher(inference_fn=router.generate, profile_name="Muthu Subramanian G")
        draft = publisher.generate_post(topic="Local LLMs")
        # User reviews...
        publisher.publish(draft, access_token=token)
    """

    def __init__(self, inference_fn: Callable[..., str], profile_name: str = "",
                 profile_headline: str = ""):
        self._inference_fn = inference_fn
        self._profile_name = profile_name
        self._profile_headline = profile_headline
        self._post_history: list[str] = []  # Track published topics to avoid repeats

    def get_topic_suggestions(self, count: int = 5) -> list[str]:
        """Get topic suggestions avoiding recently posted ones."""
        available = [t for t in _AI_TOPICS if t not in self._post_history]
        if not available:
            self._post_history.clear()
            available = list(_AI_TOPICS)
        # Rotate through topics
        return available[:count]

    def research_topic(self, topic: str) -> str:
        """Research a topic using the LLM to generate key points and trends."""
        prompt = f"""You are a tech content researcher. Research this topic for a LinkedIn post:

Topic: {topic}

Provide:
1. 3-4 key trends or insights about this topic (current, not outdated)
2. A compelling angle for a LinkedIn post (what would make professionals engage?)
3. 2-3 real-world examples or use cases
4. Suggested tone: thought-provoking but accessible

Keep the research concise — bullet points, not essays."""

        try:
            return self._inference_fn(prompt=prompt, max_tokens=600, temperature=0.7)
        except Exception as exc:
            logger.warning("Research failed: %s", exc)
            return ""

    def generate_post(self, topic: str, style: str = "thought_leadership",
                     research: str = "") -> PostDraft:
        """Generate a LinkedIn post draft on the given topic.

        Args:
            topic: The AI/LLM topic to write about
            style: "thought_leadership", "tutorial", "news", "opinion", "case_study"
            research: Optional pre-researched content to base the post on
        """
        if not research:
            research = self.research_topic(topic)

        author_context = ""
        if self._profile_name:
            author_context = f"Author: {self._profile_name}"
            if self._profile_headline:
                author_context += f" ({self._profile_headline})"

        prompt = f"""Write a professional LinkedIn post about: {topic}

Style: {style}
{author_context}

Research/context:
{research}

Requirements:
- Start with a hook that grabs attention (question, bold statement, or surprising fact)
- Keep it under 2500 characters (LinkedIn best practice for visibility)
- Use short paragraphs (2-3 sentences max)
- Include a personal perspective or experience angle
- End with a question or call-to-action to drive engagement
- Suggest 3-5 relevant hashtags at the end (just the words, no # symbol)
- Professional but approachable tone — not corporate-speak
- Do NOT use placeholder brackets like [your experience] — write concrete content

Return the post text only, followed by a line "HASHTAGS: tag1, tag2, tag3"."""

        try:
            raw = self._inference_fn(prompt=prompt, max_tokens=800, temperature=0.8)
        except Exception as exc:
            logger.error("Post generation failed: %s", exc)
            return PostDraft(topic=topic, title=topic, content="Generation failed.",
                           hashtags=[], category=style)

        # Parse hashtags from the response
        content = raw
        hashtags = []
        if "HASHTAGS:" in raw:
            parts = raw.rsplit("HASHTAGS:", 1)
            content = parts[0].strip()
            tag_str = parts[1].strip()
            hashtags = [t.strip().replace("#", "") for t in tag_str.split(",") if t.strip()]

        if not hashtags:
            hashtags = ["AI", "MachineLearning", "LLM", "TechInnovation"]

        return PostDraft(
            topic=topic,
            title=topic,
            content=content,
            hashtags=hashtags[:5],
            category=style,
            estimated_engagement="medium",
            created_at=time.time(),
        )

    def publish(self, draft: PostDraft, access_token: str, person_id: str) -> dict:
        """Publish a draft to LinkedIn.

        Tries the versioned /rest/posts API first, falls back to /v2/ugcPosts.
        """

        text = draft.to_linkedin_text()
        person_urn = f"urn:li:person:{person_id}"

        # Try versioned API first (Community Management)
        try:
            resp = requests.post(
                "https://api.linkedin.com/rest/posts",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "LinkedIn-Version": "202503",
                    "Content-Type": "application/json",
                    "X-Restli-Protocol-Version": "2.0.0",
                },
                json={
                    "author": person_urn,
                    "commentary": text,
                    "visibility": "PUBLIC",
                    "distribution": {
                        "feedDistribution": "MAIN_FEED",
                        "targetEntities": [],
                        "thirdPartyDistributionChannels": [],
                    },
                    "lifecycleState": "PUBLISHED",
                },
                timeout=30,
            )
            if resp.status_code in (200, 201):
                self._post_history.append(draft.topic)
                post_id = resp.headers.get("x-restli-id", resp.json().get("id", ""))
                return {"status": "published", "id": post_id, "api": "rest"}
            logger.info("Versioned API failed (%d), trying legacy", resp.status_code)
        except Exception as exc:
            logger.info("Versioned API error: %s, trying legacy", exc)

        # Fall back to legacy UGC API
        try:
            resp = requests.post(
                "https://api.linkedin.com/v2/ugcPosts",
                headers={
                    "Authorization": f"Bearer {access_token}",
                    "X-Restli-Protocol-Version": "2.0.0",
                    "Content-Type": "application/json",
                },
                json={
                    "author": person_urn,
                    "lifecycleState": "PUBLISHED",
                    "specificContent": {
                        "com.linkedin.ugc.ShareContent": {
                            "shareCommentary": {"text": text},
                            "shareMediaCategory": "NONE",
                        }
                    },
                    "visibility": {"com.linkedin.ugc.MemberNetworkVisibility": "PUBLIC"},
                },
                timeout=30,
            )
            if resp.status_code in (200, 201):
                self._post_history.append(draft.topic)
                return {"status": "published", "id": resp.json().get("id", ""), "api": "ugc"}
            return {"status": "failed", "error": resp.text[:300], "code": resp.status_code}
        except Exception as exc:
            return {"status": "failed", "error": str(exc)}

    def generate_weekly_schedule(self, weeks: int = 4) -> list[dict]:
        """Generate a content schedule for the next N weeks."""
        topics = self.get_topic_suggestions(count=weeks)
        schedule = []
        for i, topic in enumerate(topics):
            schedule.append({
                "week": i + 1,
                "topic": topic,
                "style": ["thought_leadership", "tutorial", "opinion", "case_study"][i % 4],
                "status": "planned",
            })
        return schedule
