"""Prompt optimizer — compresses context and history to reduce token waste."""

from typing import Optional

from homie_core.middleware.base import HomieMiddleware

# Character budget by complexity (approximate — 4 chars ≈ 1 token)
_CHAR_BUDGETS = {
    "trivial":  1500,
    "simple":   2500,
    "moderate": 5000,
    "complex":  8000,
    "deep":     12000,
}

_HISTORY_LIMITS = {
    "trivial":  2,
    "simple":   5,
    "moderate": 10,
    "complex":  20,
    "deep":     999,  # effectively unlimited
}


class PromptOptimizer(HomieMiddleware):
    """Middleware that compresses prompts based on query complexity."""

    name = "prompt_optimizer"
    order = 50  # Run early — before other middleware adds more

    def __init__(self) -> None:
        self._current_complexity = "moderate"

    def set_complexity(self, complexity: str) -> None:
        """Set current query complexity for next prompt optimization."""
        self._current_complexity = complexity

    def modify_prompt(self, prompt: str) -> str:
        """Compress the prompt based on current complexity."""
        budget = _CHAR_BUDGETS.get(self._current_complexity, 5000)
        return self.compress(prompt, self._current_complexity, max_chars=budget)

    def compress(self, prompt: str, complexity: str, max_chars: int = 5000) -> str:
        """Compress a prompt to fit within a character budget."""
        if len(prompt) <= max_chars:
            return prompt
        # Strategy: keep the first part (system prompt) and last part (recent context)
        # Trim the middle (older context, less relevant facts)
        separator = "\n...(compressed)...\n"
        available = max_chars - len(separator)
        half = available // 2
        return prompt[:half] + separator + prompt[-half:]

    def trim_history(self, history: list[dict], complexity: str) -> list[dict]:
        """Trim conversation history based on complexity budget."""
        limit = _HISTORY_LIMITS.get(complexity, 10)
        if len(history) <= limit:
            return history
        # Keep most recent turns
        return history[-limit:]

    def deduplicate_facts(self, facts: list[str]) -> list[str]:
        """Remove semantically similar facts using simple word overlap."""
        if len(facts) <= 1:
            return facts

        unique = [facts[0]]
        for fact in facts[1:]:
            is_dup = False
            fact_words = set(fact.lower().split())
            for existing in unique:
                existing_words = set(existing.lower().split())
                if not fact_words or not existing_words:
                    continue
                overlap = len(fact_words & existing_words) / len(fact_words | existing_words)
                if overlap > 0.4:
                    is_dup = True
                    break
            if not is_dup:
                unique.append(fact)
        return unique
