"""Auto-generation pipeline for proactive reports, alerts, and summaries.

Uses LLM inference to produce human-readable output from proactive task
triggers and gathered data.
"""

from __future__ import annotations

import json
import logging
from typing import Callable

from homie_core.neural.proactive.trigger_engine import ProactiveTask

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_REPORT_PROMPT = """\
Generate a concise {domain} report for the following proactive task.

Task: {action}
Priority: {priority}
Trigger: {trigger_type}

Data:
{data}

Write a well-structured report with:
1. Title
2. Executive Summary (2-3 sentences)
3. Key Findings (bullet points)
4. Detailed Analysis (if data supports it)
5. Recommendations

Be specific and actionable. Use the data provided.
"""

_ALERT_PROMPT = """\
Generate a concise alert for the following proactive trigger.

Task: {action}
Domain: {domain}
Priority: {priority} (1=critical, 10=low)
Trigger type: {trigger_type}

Data:
{data}

Write an alert with:
1. Severity (Critical / Warning / Info) — based on priority
2. What happened (1-2 sentences)
3. Why it matters
4. Recommended immediate action

Be concise and direct. This is an alert, not a full report.
"""

_SUMMARY_PROMPT = """\
Generate a concise summary on the topic: {topic}

Data:
{data}

Write a clear summary with:
1. Overview (2-3 sentences)
2. Key Points (3-5 bullet points)
3. Notable Items (anything that stands out)
4. Suggested Next Steps (if applicable)

Be specific, using the data provided.
"""


class AutoGenerator:
    """Generates reports, alerts, and summaries for proactive tasks."""

    def __init__(self, inference_fn: Callable[[str], str]) -> None:
        self._infer = inference_fn

    def generate_report(self, task: ProactiveTask, data: dict) -> str:
        """Generate a structured report for a triggered proactive task."""
        prompt = _REPORT_PROMPT.format(
            domain=task.domain,
            action=task.action,
            priority=task.priority,
            trigger_type=task.trigger_type,
            data=json.dumps(data, indent=2, default=str)[:4000],
        )
        return self._infer(prompt)

    def generate_alert(self, task: ProactiveTask, data: dict) -> str:
        """Generate a concise alert for a triggered proactive task."""
        prompt = _ALERT_PROMPT.format(
            action=task.action,
            domain=task.domain,
            priority=task.priority,
            trigger_type=task.trigger_type,
            data=json.dumps(data, indent=2, default=str)[:4000],
        )
        return self._infer(prompt)

    def generate_summary(self, topic: str, data: dict) -> str:
        """Generate a general-purpose summary on *topic* using *data*."""
        prompt = _SUMMARY_PROMPT.format(
            topic=topic,
            data=json.dumps(data, indent=2, default=str)[:4000],
        )
        return self._infer(prompt)
