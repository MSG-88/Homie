"""EmailAgent — autonomous email intelligence: classify, extract knowledge, draft replies."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── LLM prompts ──────────────────────────────────────────────────────────────

_CLASSIFY_PROMPT = """\
Classify this email. Return ONLY a JSON object.

From: {sender}
Subject: {subject}
Body (truncated):
{body}

Return: {{"priority": "<critical|high|normal|low|spam>", "categories": [<from: work, personal, bill, newsletter, social, security, travel, order, promotion>], "action_needed": <true|false>, "summary": "<1-2 sentences>"}}"""

_EXTRACT_KNOWLEDGE_PROMPT = """\
Extract structured knowledge from this email. Return ONLY a JSON object.

{content}

Return: {{"facts": [<list of factual statements>], "action_items": [<list of things the reader must do>], "deadlines": [<list of dates/deadlines mentioned>], "people": [<list of people/orgs mentioned with roles>], "amounts": [<monetary amounts mentioned>], "relationships": [<connections between entities, e.g. "Alice manages Project X">]}}"""

_DIGEST_PROMPT = """\
Create a concise email digest from these {count} emails. Use bullet points.

{emails_block}

Sections:
1. **Action Required** — emails needing response/action
2. **Important Updates** — notable but no action needed
3. **Low Priority** — count of promotional/social/newsletters

Keep it brief and scannable."""

_DRAFT_REPLY_PROMPT = """\
Draft a reply to this email.

Original email:
From: {sender}
Subject: {subject}
Body:
{body}

{instructions}

Write a professional, concise reply. Return ONLY the reply text (no subject line, no metadata)."""


def _parse_json(raw: str) -> dict | list | None:
    """Extract JSON from LLM output, handling markdown fences."""
    import re
    cleaned = re.sub(r"```(?:json)?\s*", "", raw).strip().rstrip("`")
    for i, c in enumerate(cleaned):
        if c in "{[":
            cleaned = cleaned[i:]
            break
    try:
        return json.loads(cleaned)
    except (json.JSONDecodeError, TypeError):
        return None


class EmailAgent(BaseAgent):
    """Autonomous email intelligence agent.

    Reads, classifies, extracts knowledge from, and drafts replies to emails.
    Works gracefully when email_service is None (returns empty/default results).
    All LLM calls go through inference_fn.
    """

    def __init__(
        self,
        agent_bus: AgentBus,
        inference_fn: Callable,
        email_service=None,
    ) -> None:
        super().__init__(name="email", agent_bus=agent_bus, inference_fn=inference_fn)
        self._email = email_service

    # ── BaseAgent interface ──────────────────────────────────────────

    async def process(self, message: AgentMessage) -> AgentMessage:
        """Route incoming bus messages to the appropriate method."""
        action = message.content.get("action", "process_inbox")

        if action == "process_inbox":
            result = self.process_inbox()
        elif action == "extract_knowledge":
            result = self.extract_knowledge(message.content.get("email_content", ""))
        elif action == "generate_digest":
            result = {"digest": self.generate_digest(message.content.get("emails", []))}
        elif action == "draft_reply":
            result = {
                "draft": self.draft_reply(
                    message.content.get("email", {}),
                    message.content.get("instructions", ""),
                )
            }
        elif action == "classify_priority":
            result = {"priority": self.classify_priority(message.content.get("email", {}))}
        else:
            result = {"error": f"unknown action: {action}"}

        return AgentMessage(
            from_agent=self.name,
            to_agent=message.from_agent,
            message_type="result",
            content=result if isinstance(result, dict) else {"result": result},
            parent_goal_id=message.parent_goal_id,
        )

    # ── Public API ───────────────────────────────────────────────────

    def process_inbox(self) -> dict:
        """Process unread emails: classify, extract knowledge, identify actions.

        Returns a summary dict. Degrades gracefully if email_service is None.
        """
        if self._email is None:
            return {
                "status": "no_email_service",
                "processed": 0,
                "action_items": [],
                "summary": "Email service not configured.",
            }

        try:
            unread = self._email.get_unread()
        except Exception as exc:
            logger.warning("Failed to fetch unread emails: %s", exc)
            return {
                "status": "error",
                "processed": 0,
                "action_items": [],
                "summary": f"Error fetching emails: {exc}",
            }

        all_emails = []
        for priority_bucket in unread.values():
            if isinstance(priority_bucket, list):
                all_emails.extend(priority_bucket)

        if not all_emails:
            return {
                "status": "empty",
                "processed": 0,
                "action_items": [],
                "summary": "No unread emails.",
            }

        classified = []
        action_items = []

        for email_entry in all_emails:
            classification = self.classify_priority(email_entry)
            email_entry["classified_priority"] = classification

            # Extract knowledge from snippet/body if available
            content = email_entry.get("body") or email_entry.get("snippet", "")
            if content:
                knowledge = self.extract_knowledge(content)
                email_entry["knowledge"] = knowledge
                action_items.extend(knowledge.get("action_items", []))

            classified.append(email_entry)

        return {
            "status": "ok",
            "processed": len(classified),
            "action_items": action_items,
            "emails": classified,
            "summary": f"Processed {len(classified)} emails, found {len(action_items)} action items.",
        }

    def extract_knowledge(self, email_content: str) -> dict:
        """Extract structured knowledge from email content using the LLM.

        Returns dict with: facts, action_items, deadlines, people, amounts, relationships.
        """
        if not email_content:
            return self._empty_knowledge()

        prompt = _EXTRACT_KNOWLEDGE_PROMPT.format(content=email_content[:3000])
        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict):
                # Ensure all expected keys exist
                for key in ("facts", "action_items", "deadlines", "people", "amounts", "relationships"):
                    if key not in parsed:
                        parsed[key] = []
                return parsed
        except Exception as exc:
            logger.debug("Knowledge extraction failed: %s", exc)

        return self._empty_knowledge()

    def generate_digest(self, emails: list) -> str:
        """Generate a natural language email digest from a list of email dicts."""
        if not emails:
            return "No emails to summarize."

        # Build compact representation
        parts = []
        for i, em in enumerate(emails[:30]):
            sender = em.get("sender", "unknown")
            subject = em.get("subject", "(no subject)")
            snippet = (em.get("snippet") or em.get("body", ""))[:200]
            priority = em.get("priority", "normal")
            parts.append(
                f"- [{priority}] From: {sender} | Subject: {subject} | {snippet}"
            )

        prompt = _DIGEST_PROMPT.format(
            count=len(emails),
            emails_block="\n".join(parts),
        )

        try:
            return self.inference_fn(prompt)
        except Exception as exc:
            logger.debug("Digest generation failed: %s", exc)
            return f"Failed to generate digest: {exc}"

    def draft_reply(self, email: dict, instructions: str = "") -> str:
        """Draft a reply to an email dict.

        The email dict should have at minimum: sender, subject, body or snippet.
        """
        if not email:
            return ""

        sender = email.get("sender", "unknown")
        subject = email.get("subject", "(no subject)")
        body = email.get("body") or email.get("snippet", "")

        instruction_block = (
            f"Additional instructions: {instructions}" if instructions else
            "Write a helpful and professional reply."
        )

        prompt = _DRAFT_REPLY_PROMPT.format(
            sender=sender,
            subject=subject,
            body=body[:3000],
            instructions=instruction_block,
        )

        try:
            return self.inference_fn(prompt)
        except Exception as exc:
            logger.debug("Draft reply failed: %s", exc)
            return f"Failed to draft reply: {exc}"

    def classify_priority(self, email: dict) -> str:
        """Classify email priority using LLM: critical/high/normal/low/spam.

        Falls back to 'normal' if LLM fails or email is empty.
        """
        if not email:
            return "normal"

        sender = email.get("sender", "unknown")
        subject = email.get("subject", "(no subject)")
        body = (email.get("body") or email.get("snippet", ""))[:1500]

        prompt = _CLASSIFY_PROMPT.format(sender=sender, subject=subject, body=body)

        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict):
                priority = parsed.get("priority", "normal")
                valid_priorities = {"critical", "high", "normal", "low", "spam"}
                if priority in valid_priorities:
                    return priority
        except Exception as exc:
            logger.debug("Priority classification failed: %s", exc)

        return "normal"

    # ── Private helpers ──────────────────────────────────────────────

    @staticmethod
    def _empty_knowledge() -> dict:
        return {
            "facts": [],
            "action_items": [],
            "deadlines": [],
            "people": [],
            "amounts": [],
            "relationships": [],
        }
