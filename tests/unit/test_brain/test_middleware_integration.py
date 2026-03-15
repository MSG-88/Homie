"""Integration test: middleware + backend + hooks working together."""
import pytest
from unittest.mock import MagicMock
from homie_core.middleware import HomieMiddleware, MiddlewareStack, HookRegistry, PipelineStage
from homie_core.backend import StateBackend
from homie_core.brain.orchestrator import BrainOrchestrator
from homie_core.memory.working import WorkingMemory


# ---------------------------------------------------------------------------
# Helpers / concrete middleware implementations
# ---------------------------------------------------------------------------

class PromptInjector(HomieMiddleware):
    """Prepend a tag to the incoming message."""

    name = "prompt_injector"
    order = 10

    def __init__(self, tag: str = "[INJECTED]") -> None:
        self._tag = tag
        self.before_calls: list[str] = []

    def before_turn(self, message: str, state: dict) -> str | None:
        tagged = f"{self._tag} {message}"
        self.before_calls.append(tagged)
        return tagged


class ResponseLogger(HomieMiddleware):
    """Record every response that passes through after_turn."""

    name = "response_logger"
    order = 20

    def __init__(self) -> None:
        self.logged: list[str] = []

    def after_turn(self, response: str, state: dict) -> str:
        self.logged.append(response)
        return response


class ToolBlocker(HomieMiddleware):
    """Block a specific tool by returning None from wrap_tool_call."""

    name = "tool_blocker"
    order = 5

    def __init__(self, blocked_tool: str) -> None:
        self._blocked = blocked_tool

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        if name == self._blocked:
            return None
        return args


class BlockingMiddleware(HomieMiddleware):
    """Returns None from before_turn to block the entire turn."""

    name = "blocker"
    order = 1

    def before_turn(self, message: str, state: dict) -> str | None:
        return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mock_engine():
    engine = MagicMock()
    engine.generate.return_value = "mocked response"
    engine.stream.return_value = iter(["mocked", " response"])
    return engine


@pytest.fixture
def working_memory():
    return WorkingMemory()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_full_middleware_lifecycle(mock_engine, working_memory):
    """PromptInjector and ResponseLogger fire correctly through BrainOrchestrator."""
    injector = PromptInjector(tag="[TEST]")
    logger = ResponseLogger()
    stack = MiddlewareStack([injector, logger])

    br = BrainOrchestrator(
        model_engine=mock_engine,
        working_memory=working_memory,
        middleware_stack=stack,
    )

    result = br.process("hello world")

    # before_turn fired — the injector should have seen the tagged message
    assert len(injector.before_calls) == 1
    assert injector.before_calls[0] == "[TEST] hello world"

    # after_turn fired — logger should have recorded the response
    assert len(logger.logged) == 1
    assert logger.logged[0] == result

    # The engine was called (turn was not blocked)
    mock_engine.generate.assert_called_once()


def test_tool_blocker_middleware():
    """ToolBlocker blocks run_command but allows read_file via MiddlewareStack."""
    blocker = ToolBlocker(blocked_tool="run_command")
    stack = MiddlewareStack([blocker])

    # run_command should be blocked (returns None)
    result_blocked = stack.run_wrap_tool_call("run_command", {"cmd": "rm -rf /"})
    assert result_blocked is None

    # read_file should pass through with its args intact
    args = {"path": "/etc/hosts"}
    result_allowed = stack.run_wrap_tool_call("read_file", args)
    assert result_allowed == args


def test_state_backend_works_in_memory():
    """Write → read → edit cycle on StateBackend."""
    backend = StateBackend()

    # Write
    backend.write("/notes/todo.txt", "buy milk\nbuy eggs\n")

    # Read
    fc = backend.read("/notes/todo.txt")
    assert "buy milk" in fc.content
    assert "buy eggs" in fc.content

    # Edit
    edit_result = backend.edit("/notes/todo.txt", old="buy milk", new="buy oat milk")
    assert edit_result.success is True

    # Verify the edit stuck
    fc2 = backend.read("/notes/todo.txt")
    assert "buy oat milk" in fc2.content
    assert "buy milk" not in fc2.content


def test_hooks_fire_without_middleware():
    """Register a hook on HookRegistry directly, emit, verify callback fires."""
    registry = HookRegistry()

    received: list[tuple] = []

    def my_hook(stage: PipelineStage, data) -> None:
        received.append((stage, data))

    registry.register(PipelineStage.PERCEIVED, my_hook)
    registry.emit(PipelineStage.PERCEIVED, "some input data")

    assert len(received) == 1
    stage, data = received[0]
    assert stage == PipelineStage.PERCEIVED
    assert data == "some input data"


def test_middleware_blocks_turn(mock_engine, working_memory):
    """Middleware returning None from before_turn causes process() to return '' and engine is not called."""
    stack = MiddlewareStack([BlockingMiddleware()])

    br = BrainOrchestrator(
        model_engine=mock_engine,
        working_memory=working_memory,
        middleware_stack=stack,
    )

    result = br.process("this should be blocked")

    assert result == ""
    mock_engine.generate.assert_not_called()
