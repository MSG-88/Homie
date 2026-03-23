"""CodeAgent — analyze, generate, review, and refactor code."""

from __future__ import annotations

import json
import logging
from typing import Callable

from ..communication.agent_bus import AgentBus, AgentMessage
from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

# ── LLM prompts ──────────────────────────────────────────────────────────────

_ANALYZE_PROMPT = """\
Analyze the following code file. Return ONLY a JSON object.

File: {file_path}
```
{code}
```

Return: {{"language": "<detected language>", "summary": "<what this code does>", "structure": {{"classes": [<class names>], "functions": [<function names>], "imports": [<imported modules>]}}, "issues": [<potential bugs, anti-patterns, or concerns>], "suggestions": [<improvement suggestions>], "complexity": "<low|medium|high>"}}"""

_GENERATE_PROMPT = """\
Generate code based on the following specification.

Language: {language}
Specification:
{specification}

Return ONLY the code, no explanations or markdown fences. Write clean, well-documented code with proper error handling."""

_REVIEW_PROMPT = """\
Review the following code. Return ONLY a JSON object.

{context_block}
```
{code}
```

Return: {{"quality_score": <float 0.0-1.0>, "issues": [{{"severity": "<critical|warning|info>", "line": "<approximate line or description>", "message": "<what's wrong>", "suggestion": "<how to fix>"}}], "strengths": [<what the code does well>], "summary": "<overall assessment>"}}"""

_REFACTOR_PROMPT = """\
Refactor the following code according to the instruction.

Instruction: {instruction}

Original code:
```
{code}
```

Return ONLY the refactored code, no explanations or markdown fences."""


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


def _strip_fences(raw: str) -> str:
    """Strip markdown code fences from LLM output."""
    import re
    stripped = re.sub(r"^```\w*\n?", "", raw.strip())
    stripped = re.sub(r"\n?```$", "", stripped)
    return stripped.strip()


class CodeAgent(BaseAgent):
    """Code intelligence agent: analyze, generate, review, and refactor code.

    All LLM calls go through inference_fn.
    """

    def __init__(
        self,
        agent_bus: AgentBus,
        inference_fn: Callable,
    ) -> None:
        super().__init__(name="code", agent_bus=agent_bus, inference_fn=inference_fn)

    # ── BaseAgent interface ──────────────────────────────────────────

    async def process(self, message: AgentMessage) -> AgentMessage:
        action = message.content.get("action", "analyze_code")

        if action == "analyze_code":
            result = self.analyze_code(
                message.content.get("file_path", ""),
                message.content.get("code", ""),
            )
        elif action == "generate_code":
            result = {
                "code": self.generate_code(
                    message.content.get("specification", ""),
                    message.content.get("language", "python"),
                )
            }
        elif action == "review_code":
            result = self.review_code(
                message.content.get("code", ""),
                message.content.get("context", ""),
            )
        elif action == "refactor":
            result = {
                "code": self.refactor(
                    message.content.get("code", ""),
                    message.content.get("instruction", ""),
                )
            }
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

    def analyze_code(self, file_path: str, code: str = "") -> dict:
        """Analyze a code file: structure, issues, suggestions.

        If code is not provided, attempts to read from file_path.
        """
        if not code and file_path:
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    code = f.read()
            except (OSError, IOError) as exc:
                return {"error": f"Cannot read file: {exc}"}

        if not code:
            return {"error": "No code provided and file_path is empty or unreadable."}

        prompt = _ANALYZE_PROMPT.format(
            file_path=file_path or "<inline>",
            code=code[:8000],
        )

        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and "summary" in parsed:
                parsed.setdefault("language", "unknown")
                parsed.setdefault("structure", {"classes": [], "functions": [], "imports": []})
                parsed.setdefault("issues", [])
                parsed.setdefault("suggestions", [])
                parsed.setdefault("complexity", "medium")
                return parsed
        except Exception as exc:
            logger.debug("Code analysis failed: %s", exc)

        return {
            "language": "unknown",
            "summary": "Analysis failed.",
            "structure": {"classes": [], "functions": [], "imports": []},
            "issues": [],
            "suggestions": [],
            "complexity": "unknown",
        }

    def generate_code(self, specification: str, language: str = "python") -> str:
        """Generate code from a specification."""
        if not specification:
            return ""

        prompt = _GENERATE_PROMPT.format(
            language=language,
            specification=specification,
        )

        try:
            raw = self.inference_fn(prompt)
            return _strip_fences(raw)
        except Exception as exc:
            logger.debug("Code generation failed: %s", exc)
            return f"# Code generation failed: {exc}"

    def review_code(self, code: str, context: str = "") -> dict:
        """Review code for issues, suggest improvements.

        Returns dict with quality_score, issues, strengths, summary.
        """
        if not code:
            return {
                "quality_score": 0.0,
                "issues": [],
                "strengths": [],
                "summary": "No code provided.",
            }

        context_block = f"Context: {context}\n" if context else ""

        prompt = _REVIEW_PROMPT.format(code=code[:8000], context_block=context_block)

        try:
            raw = self.inference_fn(prompt)
            parsed = _parse_json(raw)
            if isinstance(parsed, dict) and "quality_score" in parsed:
                parsed["quality_score"] = max(0.0, min(1.0, float(parsed["quality_score"])))
                parsed.setdefault("issues", [])
                parsed.setdefault("strengths", [])
                parsed.setdefault("summary", "")
                return parsed
        except Exception as exc:
            logger.debug("Code review failed: %s", exc)

        return {
            "quality_score": 0.0,
            "issues": [],
            "strengths": [],
            "summary": "Review failed.",
        }

    def refactor(self, code: str, instruction: str) -> str:
        """Refactor code based on instruction.

        Returns the refactored code string.
        """
        if not code:
            return ""
        if not instruction:
            return code  # Nothing to do

        prompt = _REFACTOR_PROMPT.format(
            code=code[:8000],
            instruction=instruction,
        )

        try:
            raw = self.inference_fn(prompt)
            return _strip_fences(raw)
        except Exception as exc:
            logger.debug("Code refactoring failed: %s", exc)
            return code  # Return original on failure
