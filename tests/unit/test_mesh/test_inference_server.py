"""Unit tests for InferenceServer."""
from unittest.mock import MagicMock

from homie_core.mesh.inference_server import InferenceServer
from homie_core.mesh.inference_queue import InferenceRequest, InferencePriority


def test_server_process_request():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "Hello!"
    server = InferenceServer(model_engine=engine, max_concurrent=2)
    assert server.process(prompt="hello", max_tokens=100, temperature=0.7) == "Hello!"
    engine.generate.assert_called_once_with(
        "hello", max_tokens=100, temperature=0.7, stop=None, timeout=120
    )


def test_server_process_no_model():
    engine = MagicMock()
    engine.is_loaded = False
    server = InferenceServer(model_engine=engine, max_concurrent=2)
    try:
        server.process(prompt="hello")
        assert False
    except RuntimeError as e:
        assert "no model" in str(e).lower()


def test_server_queue_and_process():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = "result"
    server = InferenceServer(model_engine=engine, max_concurrent=2)
    server.submit(
        InferenceRequest(
            request_id="r1",
            node_id="spoke-1",
            prompt="test",
            max_tokens=200,
            temperature=0.5,
            priority=InferencePriority.IMMEDIATE,
        )
    )
    assert server.queue_stats()["pending"] == 1
    result = server.process_next(timeout=1.0)
    assert result["request_id"] == "r1"
    assert result["content"] == "result"
    assert result["error"] is None


def test_server_process_next_empty_queue():
    engine = MagicMock()
    engine.is_loaded = True
    server = InferenceServer(model_engine=engine, max_concurrent=2)
    assert server.process_next(timeout=0.1) is None


def test_server_handles_generation_error():
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.side_effect = TimeoutError("Model too slow")
    server = InferenceServer(model_engine=engine, max_concurrent=2)
    server.submit(
        InferenceRequest(
            request_id="r1",
            node_id="spoke-1",
            prompt="test",
            max_tokens=100,
            temperature=0.7,
            priority=InferencePriority.IMMEDIATE,
        )
    )
    result = server.process_next(timeout=1.0)
    assert result["error"] is not None
    assert "too slow" in result["error"].lower()
