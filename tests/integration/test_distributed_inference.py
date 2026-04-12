"""End-to-end distributed inference integration tests.

Tests cover spoke→hub routing, hub streaming, multi-spoke queue processing,
and local-model short-circuit (mesh bypass).
"""
from __future__ import annotations

from unittest.mock import MagicMock

from homie_core.mesh.inference_client import MeshInferenceClient
from homie_core.mesh.inference_server import InferenceServer
from homie_core.mesh.inference_queue import InferenceRequest, InferencePriority
from homie_core.mesh.mesh_inference_router import MeshInferenceRouter


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _hub_engine(resp="Hub says hello"):
    e = MagicMock()
    e.is_loaded = True
    e.generate.return_value = resp
    e.stream.return_value = iter([resp])
    return e


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_spoke_to_hub_inference_flow():
    """Spoke with no local model should route generate() through the hub."""
    hub_engine = _hub_engine("The answer is 42")
    hub_server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    def hub_handler(prompt, max_tokens, temperature, stop):
        return hub_server.process(prompt=prompt, max_tokens=max_tokens, temperature=temperature, stop=stop)

    spoke_engine = MagicMock()
    spoke_engine.is_loaded = False

    client = MeshInferenceClient(node_id="spoke-1", hub_node_id="hub-1", request_handler=hub_handler)
    router = MeshInferenceRouter(model_engine=spoke_engine, mesh_client=client, priority=["local", "hub"])

    result = router.generate("What is the meaning?")

    assert result == "The answer is 42"
    spoke_engine.generate.assert_not_called()
    hub_engine.generate.assert_called_once()


def test_spoke_streams_from_hub():
    """Spoke with no local model should stream tokens from the hub."""
    hub_engine = _hub_engine("streaming tokens")
    server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    def handler(p, m, t, s):
        return server.process(prompt=p, max_tokens=m, temperature=t, stop=s)

    spoke_engine = MagicMock()
    spoke_engine.is_loaded = False

    client = MeshInferenceClient(node_id="s1", hub_node_id="h1", request_handler=handler)
    router = MeshInferenceRouter(model_engine=spoke_engine, mesh_client=client, priority=["local", "hub"])

    assert "".join(router.stream("test")) == "streaming tokens"


def test_hub_queue_processes_multiple_spokes():
    """Hub queue should process requests from multiple spokes and track completions."""
    hub_engine = _hub_engine()
    hub_engine.generate.side_effect = lambda p, **kw: f"reply to {p[:5]}"
    server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    for i in range(3):
        server.submit(
            InferenceRequest(
                request_id=f"r{i}",
                node_id=f"spoke-{i}",
                prompt=f"question {i}",
                max_tokens=100,
                temperature=0.7,
                priority=InferencePriority.IMMEDIATE,
            )
        )

    results = [server.process_next(timeout=1.0) for _ in range(3)]

    assert all(r["error"] is None for r in results)
    assert server.queue_stats()["completed"] == 3


def test_local_model_skips_mesh():
    """When local model is loaded it should be used and mesh client must not be called."""
    spoke_engine = MagicMock()
    spoke_engine.is_loaded = True
    spoke_engine.generate.return_value = "local answer"

    client = MagicMock()
    client.is_available = True

    router = MeshInferenceRouter(model_engine=spoke_engine, mesh_client=client, priority=["local", "hub"])

    assert router.generate("test") == "local answer"
    client.generate.assert_not_called()
