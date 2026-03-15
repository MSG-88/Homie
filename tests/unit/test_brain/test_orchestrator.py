import pytest
from unittest.mock import MagicMock, patch, call
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_core.memory.working import WorkingMemory
from homie_core.middleware import HomieMiddleware, MiddlewareStack, HookRegistry


@pytest.fixture
def brain():
    engine = MagicMock()
    engine.generate.return_value = "Hello! I'm Homie, your AI assistant."
    wm = WorkingMemory()
    return BrainOrchestrator(model_engine=engine, working_memory=wm), engine, wm


def test_process_returns_response(brain):
    br, engine, wm = brain
    response = br.process("Hello")
    assert response == "Hello! I'm Homie, your AI assistant."
    engine.generate.assert_called_once()


def test_process_adds_to_conversation(brain):
    br, _, wm = brain
    br.process("Hello")
    msgs = wm.get_conversation()
    assert len(msgs) == 2
    assert msgs[0]["role"] == "user"
    assert msgs[1]["role"] == "assistant"


def test_prompt_includes_facts(brain):
    br, _, _ = brain
    sm = MagicMock()
    sm.get_facts.return_value = [{"fact": "user likes Python", "confidence": 0.9}]
    br._sm = sm
    prompt = br._build_optimized_prompt("test")
    assert "user likes Python" in prompt


def test_prompt_includes_episodes(brain):
    br, _, _ = brain
    em = MagicMock()
    em.recall.return_value = [{"summary": "Debugged auth module", "mood": "focused"}]
    br._em = em
    prompt = br._build_optimized_prompt("debugging")
    assert "Debugged auth module" in prompt


def test_process_stream_yields_tokens(brain):
    br, engine, wm = brain
    engine.stream.return_value = iter(["Hello", " world", "!"])
    tokens = list(br.process_stream("Hi"))
    assert tokens == ["Hello", " world", "!"]
    msgs = wm.get_conversation()
    assert msgs[-1]["content"] == "Hello world!"


def test_optimized_prompt_respects_budget(brain):
    br, _, _ = brain
    br.set_system_prompt("A" * 2500)
    prompt = br._build_optimized_prompt("test query")
    # Should still include system prompt and user query even with tight budget
    assert "A" * 2500 in prompt
    assert "test query" in prompt


def test_set_system_prompt(brain):
    br, _, _ = brain
    br.set_system_prompt("You are a coding assistant.")
    assert br._system_prompt == "You are a coding assistant."


def test_orchestrator_accepts_middleware_stack():
    """Orchestrator stores a provided MiddlewareStack and exposes a hooks property."""
    engine = MagicMock()
    engine.generate.return_value = "ok"
    wm = WorkingMemory()
    stack = MiddlewareStack()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    assert br._middleware is stack
    assert isinstance(br.hooks, HookRegistry)


def test_orchestrator_works_without_middleware():
    """Orchestrator creates a default MiddlewareStack when none is provided."""
    engine = MagicMock()
    engine.generate.return_value = "response"
    wm = WorkingMemory()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm)
    assert isinstance(br._middleware, MiddlewareStack)
    result = br.process("hello")
    assert result == "response"


def test_orchestrator_state_kwarg_backward_compat():
    """process() accepts optional state kwarg without breaking callers that omit it."""
    engine = MagicMock()
    engine.generate.return_value = "hi"
    wm = WorkingMemory()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm)
    # Call without state — backward compat
    assert br.process("hello") == "hi"
    # Call with explicit state
    assert br.process("hello", state={"key": "value"}) == "hi"


def test_orchestrator_middleware_before_turn_blocks():
    """Middleware that returns None from before_turn causes process() to return ''."""
    engine = MagicMock()
    engine.generate.return_value = "should not appear"
    wm = WorkingMemory()

    class BlockingMiddleware(HomieMiddleware):
        def before_turn(self, message: str, state: dict):
            return None  # Block

    stack = MiddlewareStack([BlockingMiddleware()])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    result = br.process("hello")
    assert result == ""
    engine.generate.assert_not_called()


def test_process_stream_fires_middleware():
    """process_stream() runs before/after turn middleware hooks."""
    engine = MagicMock()
    engine.stream.return_value = iter(["hello", " world"])
    wm = WorkingMemory()

    before_calls = []
    after_calls = []

    class TrackingMiddleware(HomieMiddleware):
        def before_turn(self, message: str, state: dict):
            before_calls.append(message)
            return message

        def after_turn(self, response: str, state: dict):
            after_calls.append(response)
            return response

    stack = MiddlewareStack([TrackingMiddleware()])
    br = BrainOrchestrator(model_engine=engine, working_memory=wm, middleware_stack=stack)
    tokens = list(br.process_stream("hi"))
    assert len(before_calls) == 1
    assert len(after_calls) == 1
    assert "".join(tokens) in ("hello world", after_calls[0])


# ---------------------------------------------------------------------------
# Context-overflow recovery tests
# ---------------------------------------------------------------------------

def test_process_handles_context_overflow():
    """Context-overflow error from engine is caught; recovery runs and retries."""
    engine = MagicMock()
    wm = WorkingMemory()

    # First call raises overflow; second call succeeds
    engine.generate.side_effect = [
        RuntimeError("context length exceeded"),
        "recovered response",
    ]

    br = BrainOrchestrator(model_engine=engine, working_memory=wm)

    # Populate some conversation so middleware has something to compress
    for i in range(10):
        wm.add_message("user" if i % 2 == 0 else "assistant", "x" * 50)

    result = br.process("hello")
    assert result == "recovered response"
    assert engine.generate.call_count == 2


def test_process_handles_context_overflow_too_long():
    """'too long' in error message is also detected as context overflow."""
    engine = MagicMock()
    wm = WorkingMemory()

    engine.generate.side_effect = [
        RuntimeError("prompt is too long for the model"),
        "ok",
    ]

    br = BrainOrchestrator(model_engine=engine, working_memory=wm)
    result = br.process("test")
    assert result == "ok"


def test_non_overflow_errors_propagate():
    """Non-overflow runtime errors are re-raised without retry."""
    engine = MagicMock()
    wm = WorkingMemory()
    engine.generate.side_effect = RuntimeError("model crashed unexpectedly")

    br = BrainOrchestrator(model_engine=engine, working_memory=wm)

    with pytest.raises(RuntimeError, match="model crashed"):
        br.process("hello")

    # Only one attempt — no retry
    assert engine.generate.call_count == 1


def test_process_stream_handles_context_overflow():
    """Streaming path also recovers from context-overflow errors."""
    engine = MagicMock()
    wm = WorkingMemory()

    # First stream call raises overflow; second succeeds
    engine.stream.side_effect = [
        RuntimeError("context overflow"),
        iter(["recovered", " stream"]),
    ]

    br = BrainOrchestrator(model_engine=engine, working_memory=wm)

    for i in range(10):
        wm.add_message("user" if i % 2 == 0 else "assistant", "x" * 50)

    tokens = list(br.process_stream("hello"))
    assert "".join(tokens) in ("recovered stream", "recovered  stream")
    assert engine.stream.call_count == 2


def test_process_stream_non_overflow_errors_propagate():
    """Non-overflow errors in streaming path are re-raised."""
    engine = MagicMock()
    wm = WorkingMemory()
    engine.stream.side_effect = RuntimeError("gpu out of memory")

    br = BrainOrchestrator(model_engine=engine, working_memory=wm)

    with pytest.raises(RuntimeError, match="gpu out of memory"):
        list(br.process_stream("hello"))

    assert engine.stream.call_count == 1


def test_is_context_overflow_detection():
    """_is_context_overflow correctly identifies overflow error messages."""
    engine = MagicMock()
    wm = WorkingMemory()
    br = BrainOrchestrator(model_engine=engine, working_memory=wm)

    assert br._is_context_overflow(RuntimeError("context length exceeded"))
    assert br._is_context_overflow(RuntimeError("context overflow"))
    assert br._is_context_overflow(RuntimeError("input too long"))
    assert br._is_context_overflow(RuntimeError("Context Length Exceeded"))

    assert not br._is_context_overflow(RuntimeError("model crashed"))
    assert not br._is_context_overflow(RuntimeError("out of memory"))
    assert not br._is_context_overflow(ValueError("bad argument"))
