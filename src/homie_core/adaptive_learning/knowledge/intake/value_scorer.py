"""Value scorer — determines which files deserve deep LLM analysis."""

import math
from typing import Any, Optional


class ValueScorer:
    """Scores files by value to determine which get deep analysis."""

    def __init__(self, top_percent: int = 20) -> None:
        self._top_percent = top_percent

    def score_files(
        self,
        extractions: list[dict[str, Any]],
        import_counts: Optional[dict[str, int]] = None,
    ) -> dict[str, float]:
        """Score each file. Higher = more valuable for deep analysis."""
        if not extractions:
            return {}

        import_counts = import_counts or {}
        scores = {}

        for ext in extractions:
            file_key = ext.get("file", "")
            score = 0.0

            # Size score — larger files have more to extract
            line_count = ext.get("line_count", 0)
            score += min(math.log(line_count + 1) / 6.0, 1.0)  # caps at ~400 lines

            # Class/function density — more definitions = more architecture
            classes = len(ext.get("classes", []))
            functions = len(ext.get("functions", []))
            score += min((classes + functions) / 10.0, 1.0)

            # Import reference count — how many other files import this one
            basename = file_key.rsplit("/", 1)[-1].rsplit("\\", 1)[-1].replace(".py", "")
            ref_count = import_counts.get(basename, 0)
            score += min(ref_count / 5.0, 1.0)

            scores[file_key] = score

        return scores

    def select_for_deep_pass(
        self,
        extractions: list[dict[str, Any]],
        import_counts: Optional[dict[str, int]] = None,
    ) -> list[dict]:
        """Select the top N% of files for deep analysis."""
        if not extractions:
            return []

        scores = self.score_files(extractions, import_counts)
        sorted_files = sorted(extractions, key=lambda e: scores.get(e.get("file", ""), 0), reverse=True)
        count = max(1, len(sorted_files) * self._top_percent // 100)
        return sorted_files[:count]
