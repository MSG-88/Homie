"""Deep analyzer — LLM-powered knowledge extraction from file content."""

import json
import logging
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)

_EXTRACTION_PROMPT = """Analyze this source code/document and extract structured knowledge.

File: {file_path}
Content:
{content}

Extract as JSON:
{{
  "entities": [{{"name": "...", "type": "class|function|module|concept|technology"}}],
  "relationships": [{{"subject": "...", "predicate": "uses|depends_on|implements|handles|contains", "object": "..."}}]
}}

Be concise. Only extract clearly stated facts, not speculation."""


class DeepAnalyzer:
    """LLM-powered deep extraction of entities and relationships."""

    def __init__(
        self,
        inference_fn: Optional[Callable[[str], str]] = None,
        max_content_chars: int = 8000,
    ) -> None:
        self._infer = inference_fn
        self._max_chars = max_content_chars

    def analyze(self, content: str, file_path: str = "") -> dict[str, Any]:
        """Analyze content and extract entities and relationships."""
        if self._infer is None:
            return {"entities": [], "relationships": []}

        # Truncate if needed
        if len(content) > self._max_chars:
            content = content[:self._max_chars] + "\n... (truncated)"

        prompt = _EXTRACTION_PROMPT.format(file_path=file_path, content=content)

        try:
            response = self._infer(prompt)
            return self._parse_response(response)
        except Exception:
            logger.warning("Deep analysis failed for %s", file_path)
            return {"entities": [], "relationships": []}

    def _parse_response(self, response: str) -> dict[str, Any]:
        """Parse LLM response as JSON."""
        # Try to find JSON in the response
        try:
            start = response.find("{")
            end = response.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(response[start:end])
                return {
                    "entities": data.get("entities", []),
                    "relationships": data.get("relationships", []),
                }
        except (json.JSONDecodeError, ValueError):
            pass
        return {"entities": [], "relationships": []}
