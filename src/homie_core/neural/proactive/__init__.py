"""Proactive autonomous intelligence for the Neural Reasoning Engine."""

from homie_core.neural.proactive.trigger_engine import ProactiveTask, TriggerEngine
from homie_core.neural.proactive.auto_generator import AutoGenerator
from homie_core.neural.proactive.learned_triggers import LearnedTriggerManager

__all__ = [
    "ProactiveTask",
    "TriggerEngine",
    "AutoGenerator",
    "LearnedTriggerManager",
]
