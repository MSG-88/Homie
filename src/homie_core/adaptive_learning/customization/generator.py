"""Customization generator — creates code from natural language requests."""

import logging
from pathlib import Path
from typing import Callable, Optional

logger = logging.getLogger(__name__)

_ANALYSIS_PROMPT = """Analyze this customization request and describe what code needs to be generated.
Request: {request}

Describe:
1. What type of code is needed (middleware, tool, prompt modification, scheduled task)
2. What conditions trigger it
3. What behavior it should produce
4. What existing systems it needs to integrate with

Be specific and concise."""

_GENERATION_PROMPT = """Generate Python code for the following customization.

Analysis: {analysis}

Requirements:
- Must be a complete, self-contained Python module
- Follow existing patterns in the Homie codebase
- Include necessary imports
- Include a brief module docstring

Generate only the Python code, nothing else."""


class CustomizationGenerator:
    """Generates code from natural language customization requests."""

    def __init__(
        self,
        inference_fn: Callable[[str], str],
        evolver,
        rollback,
        project_root: str | Path,
    ) -> None:
        self._infer = inference_fn
        self._evolver = evolver
        self._rollback = rollback
        self._root = Path(project_root)

    def analyze_request(self, request: str) -> str:
        """Analyze a customization request to understand what to build."""
        prompt = _ANALYSIS_PROMPT.format(request=request)
        return self._infer(prompt)

    def generate_code(self, request: str, analysis: str = "") -> str:
        """Generate code for a customization."""
        if not analysis:
            analysis = self.analyze_request(request)
        prompt = _GENERATION_PROMPT.format(analysis=analysis)
        return self._infer(prompt)

    def apply(self, file_path: str, code: str, reason: str = "") -> str:
        """Apply generated code via ArchitectureEvolver. Returns version_id."""
        full_path = self._root / file_path if not Path(file_path).is_absolute() else Path(file_path)
        return self._evolver.create_module(
            file_path=full_path,
            content=code,
            reason=reason,
        )
