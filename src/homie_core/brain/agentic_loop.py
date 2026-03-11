"""Agentic Loop — multi-turn reasoning with tool use.

The loop:
1. Build prompt (cognitive architecture) + tool descriptions
2. Generate response
3. Check for tool calls in output
4. If tool calls found: execute tools, append results, re-generate
5. Repeat until no more tool calls or max iterations reached
6. Return final response (with tool call markers stripped)

This transforms Homie from a one-shot chatbot into an agent that can
reason, act, observe results, and reason again.
"""
from __future__ import annotations

import re
from typing import Iterator, Optional

from homie_core.brain.tool_registry import ToolRegistry, ToolResult, parse_tool_calls


# Strip tool call markers from final output shown to user
_TOOL_MARKER_PATTERN = re.compile(r"<tool>.*?</tool>", re.DOTALL)
_JSON_TOOL_MARKER = re.compile(r'\{"tool"\s*:.*?\}', re.DOTALL)

# Maximum tool-use iterations to prevent infinite loops
_MAX_ITERATIONS = 5


def _strip_tool_markers(text: str) -> str:
    """Remove tool call syntax from text shown to user."""
    text = _TOOL_MARKER_PATTERN.sub("", text)
    text = _JSON_TOOL_MARKER.sub("", text)
    # Clean up extra whitespace left behind
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _format_tool_results(results: list[ToolResult]) -> str:
    """Format tool results as context for the next generation round."""
    parts = []
    for r in results:
        if r.success:
            parts.append(f"[Tool: {r.tool_name}] Result: {r.output}")
        else:
            parts.append(f"[Tool: {r.tool_name}] Error: {r.error}")
    return "\n".join(parts)


class AgenticLoop:
    """Multi-turn agentic reasoning loop with tool use.

    Wraps a model engine and tool registry. Each call to process()
    may trigger multiple generation rounds if the model invokes tools.
    """

    def __init__(
        self,
        model_engine,
        tool_registry: ToolRegistry,
        max_iterations: int = _MAX_ITERATIONS,
    ):
        self._engine = model_engine
        self._tools = tool_registry
        self._max_iterations = max_iterations

    def process(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> str:
        """Run the agentic loop — blocking mode.

        Returns the final response with tool markers stripped.
        """
        current_prompt = prompt
        all_text_parts = []

        for iteration in range(self._max_iterations):
            # Generate
            response = self._engine.generate(
                current_prompt, max_tokens=max_tokens, temperature=temperature,
            )

            # Check for tool calls
            tool_calls = parse_tool_calls(response)

            if not tool_calls:
                # No tool calls — this is the final response
                all_text_parts.append(response)
                break

            # Execute tools
            results = [self._tools.execute(call) for call in tool_calls]

            # Collect the text before/between tool calls
            clean_text = _strip_tool_markers(response)
            if clean_text:
                all_text_parts.append(clean_text)

            # Append tool results and re-prompt
            tool_output = _format_tool_results(results)
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {response}\n\n{tool_output}\n\n"
                f"Continue your response based on the tool results above.\nAssistant:"
            )
        else:
            # Max iterations reached
            all_text_parts.append(
                "(Reached maximum reasoning steps. Here's what I have so far.)"
            )

        final = "\n".join(all_text_parts)
        return _strip_tool_markers(final)

    def process_stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
    ) -> Iterator[str]:
        """Run the agentic loop — streaming mode.

        Yields tokens in real-time. When a tool call is detected in accumulated
        output, pauses streaming, executes tools, then continues generation
        with a new streaming round.
        """
        current_prompt = prompt

        for iteration in range(self._max_iterations):
            # Stream tokens, accumulating for tool call detection
            accumulated = []
            tool_detected = False

            for token in self._engine.stream(
                current_prompt, max_tokens=max_tokens, temperature=temperature,
            ):
                accumulated.append(token)

                # Check if we've seen a complete tool call
                full_text = "".join(accumulated)
                if "<tool>" in full_text and "</tool>" in full_text:
                    tool_detected = True
                    break

                # Only yield tokens that are clearly not part of a tool call
                if "<tool>" not in full_text:
                    yield token

            full_response = "".join(accumulated)

            if not tool_detected:
                # No tool calls — yield any remaining buffered tokens
                # (tokens after last yield point)
                break

            # Parse and execute tool calls
            tool_calls = parse_tool_calls(full_response)
            if not tool_calls:
                break

            results = [self._tools.execute(call) for call in tool_calls]

            # Yield a brief indicator that tools are working
            yield "\n"

            # Build continuation prompt
            tool_output = _format_tool_results(results)
            current_prompt = (
                f"{current_prompt}\n\nAssistant: {full_response}\n\n{tool_output}\n\n"
                f"Continue your response based on the tool results above.\nAssistant:"
            )
