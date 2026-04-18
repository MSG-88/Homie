# Distributed Mesh — Phase 3: Distributed Inference

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Spokes can route inference requests to the Hub (or any mesh peer with a loaded model) and receive streamed tokens back. An inference queue on the Hub manages priority and concurrency. The existing `InferenceRouter` gains a "mesh" source in its fallback chain.

**Architecture:** `MeshInferenceClient` sends requests to Hub via the existing `MeshMessage` transport. `InferenceQueue` on Hub manages concurrent requests with priority levels (IMMEDIATE/BACKGROUND/BATCH). `MeshInferenceRouter` extends `InferenceRouter` by inserting "hub" and "lan_peer" sources between "local" and "qubrid". Model distribution metadata syncs via mesh events.

**Tech Stack:** Python 3.11+, asyncio, existing mesh transport (HMAC MeshMessage), existing ModelEngine

**Builds on:** Phase 1 (identity, capabilities, registry), Phase 2 (events, transport, mesh_manager)

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/inference_queue.py` | Priority queue for Hub inference requests |
| `src/homie_core/mesh/inference_client.py` | Client that sends inference to Hub and receives response |
| `src/homie_core/mesh/inference_server.py` | Server that accepts inference requests on Hub |
| `src/homie_core/mesh/mesh_inference_router.py` | Extended InferenceRouter with mesh sources |
| `tests/unit/test_mesh/test_inference_queue.py` | Queue tests |
| `tests/unit/test_mesh/test_inference_client.py` | Client tests |
| `tests/unit/test_mesh/test_mesh_inference_router.py` | Router tests |
| `tests/integration/test_distributed_inference.py` | End-to-end inference test |

---

### Task 1: Inference Queue — Priority-Based Request Management

**Files:**
- Create: `src/homie_core/mesh/inference_queue.py`
- Test: `tests/unit/test_mesh/test_inference_queue.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_inference_queue.py
import time
import threading
from homie_core.mesh.inference_queue import InferenceQueue, InferenceRequest, InferencePriority


def test_priority_levels():
    assert InferencePriority.IMMEDIATE < InferencePriority.BACKGROUND
    assert InferencePriority.BACKGROUND < InferencePriority.BATCH


def test_submit_and_get():
    """Submit a request and get it from the queue."""
    q = InferenceQueue(max_concurrent=2)
    req = InferenceRequest(
        request_id="r1", node_id="spoke-1", prompt="hello",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.IMMEDIATE,
    )
    q.submit(req)
    assert q.pending_count() == 1

    got = q.get(timeout=1.0)
    assert got is not None
    assert got.request_id == "r1"
    assert q.pending_count() == 0


def test_priority_ordering():
    """Higher priority requests come out first."""
    q = InferenceQueue(max_concurrent=5)
    q.submit(InferenceRequest(
        request_id="batch", node_id="n1", prompt="p",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.BATCH,
    ))
    q.submit(InferenceRequest(
        request_id="immediate", node_id="n1", prompt="p",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.IMMEDIATE,
    ))
    q.submit(InferenceRequest(
        request_id="background", node_id="n1", prompt="p",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.BACKGROUND,
    ))

    order = [q.get(timeout=1.0).request_id for _ in range(3)]
    assert order == ["immediate", "background", "batch"]


def test_max_concurrent_blocking():
    """Queue blocks when max_concurrent active requests reached."""
    q = InferenceQueue(max_concurrent=1)
    req1 = InferenceRequest(
        request_id="r1", node_id="n1", prompt="p",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.IMMEDIATE,
    )
    q.submit(req1)
    got = q.get(timeout=1.0)
    assert got is not None
    q.mark_active(got.request_id)

    assert q.active_count() == 1
    assert q.can_accept() is False

    q.mark_done(got.request_id)
    assert q.active_count() == 0
    assert q.can_accept() is True


def test_get_timeout_returns_none():
    """get() returns None on timeout when queue is empty."""
    q = InferenceQueue(max_concurrent=2)
    got = q.get(timeout=0.1)
    assert got is None


def test_queue_stats():
    """Stats show pending, active, and completed counts."""
    q = InferenceQueue(max_concurrent=5)
    for i in range(3):
        q.submit(InferenceRequest(
            request_id=f"r{i}", node_id="n1", prompt="p",
            max_tokens=100, temperature=0.7,
            priority=InferencePriority.IMMEDIATE,
        ))
    stats = q.stats()
    assert stats["pending"] == 3
    assert stats["active"] == 0
    assert stats["completed"] == 0

    req = q.get(timeout=1.0)
    q.mark_active(req.request_id)
    q.mark_done(req.request_id)
    stats = q.stats()
    assert stats["pending"] == 2
    assert stats["completed"] == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_queue.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/inference_queue.py
"""Priority queue for Hub inference requests."""
from __future__ import annotations

import queue
import threading
from dataclasses import dataclass, field
from typing import Optional


class InferencePriority:
    """Priority levels. Lower number = higher priority."""
    IMMEDIATE = 0    # User waiting (chat, voice)
    BACKGROUND = 1   # Proactive tasks, summarization
    BATCH = 2        # Synthetic data, bulk processing


@dataclass(order=True)
class InferenceRequest:
    """A queued inference request from a mesh node."""
    # Fields used for ordering (priority first, then submission order)
    priority: int = field(compare=True)
    _order: int = field(default=0, compare=True, repr=False)

    # Actual request data (not used for ordering)
    request_id: str = field(default="", compare=False)
    node_id: str = field(default="", compare=False)
    prompt: str = field(default="", compare=False)
    max_tokens: int = field(default=1024, compare=False)
    temperature: float = field(default=0.7, compare=False)
    stop: Optional[list[str]] = field(default=None, compare=False)


class InferenceQueue:
    """Thread-safe priority queue for inference requests on Hub."""

    def __init__(self, max_concurrent: int = 2):
        self._queue: queue.PriorityQueue = queue.PriorityQueue()
        self._max_concurrent = max_concurrent
        self._active: dict[str, InferenceRequest] = {}
        self._completed_count = 0
        self._order_counter = 0
        self._lock = threading.Lock()

    def submit(self, request: InferenceRequest) -> None:
        """Add a request to the queue."""
        with self._lock:
            request._order = self._order_counter
            self._order_counter += 1
        self._queue.put(request)

    def get(self, timeout: float = 5.0) -> Optional[InferenceRequest]:
        """Get the next highest-priority request. Returns None on timeout."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def mark_active(self, request_id: str) -> None:
        """Mark a request as currently being processed."""
        with self._lock:
            self._active[request_id] = True

    def mark_done(self, request_id: str) -> None:
        """Mark a request as completed."""
        with self._lock:
            self._active.pop(request_id, None)
            self._completed_count += 1

    def can_accept(self) -> bool:
        """Check if the queue can accept more concurrent requests."""
        with self._lock:
            return len(self._active) < self._max_concurrent

    def pending_count(self) -> int:
        return self._queue.qsize()

    def active_count(self) -> int:
        with self._lock:
            return len(self._active)

    def stats(self) -> dict:
        with self._lock:
            return {
                "pending": self._queue.qsize(),
                "active": len(self._active),
                "completed": self._completed_count,
                "max_concurrent": self._max_concurrent,
            }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_queue.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/inference_queue.py tests/unit/test_mesh/test_inference_queue.py
git commit -m "feat(mesh): add priority inference queue for Hub request management"
```

---

### Task 2: Inference Client — Spoke Sends Requests to Hub

**Files:**
- Create: `src/homie_core/mesh/inference_client.py`
- Test: `tests/unit/test_mesh/test_inference_client.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_inference_client.py
from unittest.mock import MagicMock
from homie_core.mesh.inference_client import MeshInferenceClient


def test_client_generate_calls_handler():
    """Client calls the request handler and returns the response content."""
    def fake_handler(prompt, max_tokens, temperature, stop):
        return f"response to: {prompt[:10]}"

    client = MeshInferenceClient(
        node_id="spoke-1",
        hub_node_id="hub-1",
        request_handler=fake_handler,
    )
    result = client.generate("hello world", max_tokens=100, temperature=0.7)
    assert result == "response to: hello worl"


def test_client_generate_with_stop():
    """Stop sequences are passed through."""
    captured = {}
    def fake_handler(prompt, max_tokens, temperature, stop):
        captured["stop"] = stop
        return "ok"

    client = MeshInferenceClient(
        node_id="spoke-1", hub_node_id="hub-1",
        request_handler=fake_handler,
    )
    client.generate("test", stop=["END"])
    assert captured["stop"] == ["END"]


def test_client_is_available():
    """Client reports availability based on handler presence."""
    client_with = MeshInferenceClient(
        node_id="s1", hub_node_id="h1",
        request_handler=lambda p, m, t, s: "ok",
    )
    assert client_with.is_available is True

    client_without = MeshInferenceClient(
        node_id="s1", hub_node_id="h1",
        request_handler=None,
    )
    assert client_without.is_available is False


def test_client_generate_raises_on_no_handler():
    """Client raises RuntimeError if no handler is set."""
    client = MeshInferenceClient(
        node_id="s1", hub_node_id="h1", request_handler=None,
    )
    try:
        client.generate("test")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "no hub" in str(e).lower() or "not available" in str(e).lower()


def test_client_stream_yields_tokens():
    """Client stream calls handler and yields the result as a single chunk."""
    def fake_handler(prompt, max_tokens, temperature, stop):
        return "streamed response"

    client = MeshInferenceClient(
        node_id="s1", hub_node_id="h1",
        request_handler=fake_handler,
    )
    tokens = list(client.stream("test", max_tokens=50))
    assert "".join(tokens) == "streamed response"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_client.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/inference_client.py
"""Mesh inference client — Spoke sends inference requests to Hub.

In the current implementation, this wraps a callable request_handler
that abstracts the transport (direct call for testing, WebSocket for
production). The MeshInferenceRouter plugs this into the fallback chain.
"""
from __future__ import annotations

from typing import Callable, Iterator, Optional


class MeshInferenceClient:
    """Client for sending inference requests to a mesh Hub."""

    def __init__(
        self,
        node_id: str,
        hub_node_id: str,
        request_handler: Optional[Callable] = None,
    ):
        self._node_id = node_id
        self._hub_node_id = hub_node_id
        self._handler = request_handler

    @property
    def is_available(self) -> bool:
        return self._handler is not None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Send inference request to Hub and return the full response."""
        if not self._handler:
            raise RuntimeError(
                "Mesh inference not available — no hub connected. "
                "Check mesh status with /mesh status."
            )
        return self._handler(prompt, max_tokens, temperature, stop)

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Send inference request and yield response tokens.

        Current implementation returns a single chunk (the full response).
        WebSocket-based streaming will yield token-by-token in production.
        """
        result = self.generate(prompt, max_tokens, temperature, stop)
        yield result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_client.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/inference_client.py tests/unit/test_mesh/test_inference_client.py
git commit -m "feat(mesh): add mesh inference client for Spoke-to-Hub request routing"
```

---

### Task 3: Inference Server — Hub Processes Requests

**Files:**
- Create: `src/homie_core/mesh/inference_server.py`
- Test: `tests/unit/test_mesh/test_inference_server.py` (inline in this task)

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_inference_server.py
from unittest.mock import MagicMock
from homie_core.mesh.inference_server import InferenceServer
from homie_core.mesh.inference_queue import InferenceRequest, InferencePriority


def test_server_process_request():
    """Server processes a request using the model engine."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.generate.return_value = "Hello! How can I help?"

    server = InferenceServer(model_engine=mock_engine, max_concurrent=2)
    result = server.process(
        prompt="hello", max_tokens=100,
        temperature=0.7, stop=None,
    )
    assert result == "Hello! How can I help?"
    mock_engine.generate.assert_called_once_with(
        "hello", max_tokens=100, temperature=0.7, stop=None, timeout=120,
    )


def test_server_process_no_model():
    """Server raises when no model is loaded."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = False

    server = InferenceServer(model_engine=mock_engine, max_concurrent=2)
    try:
        server.process(prompt="hello")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "no model" in str(e).lower()


def test_server_queue_and_process():
    """Server queues a request and processes it."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.generate.return_value = "result"

    server = InferenceServer(model_engine=mock_engine, max_concurrent=2)

    req = InferenceRequest(
        request_id="r1", node_id="spoke-1", prompt="test prompt",
        max_tokens=200, temperature=0.5,
        priority=InferencePriority.IMMEDIATE,
    )
    server.submit(req)
    assert server.queue_stats()["pending"] == 1

    # Process next queued request
    result = server.process_next(timeout=1.0)
    assert result is not None
    assert result["request_id"] == "r1"
    assert result["content"] == "result"
    assert result["error"] is None


def test_server_process_next_empty_queue():
    """process_next returns None when queue is empty."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = True

    server = InferenceServer(model_engine=mock_engine, max_concurrent=2)
    result = server.process_next(timeout=0.1)
    assert result is None


def test_server_handles_generation_error():
    """Server catches engine errors and returns error in result."""
    mock_engine = MagicMock()
    mock_engine.is_loaded = True
    mock_engine.generate.side_effect = TimeoutError("Model too slow")

    server = InferenceServer(model_engine=mock_engine, max_concurrent=2)
    req = InferenceRequest(
        request_id="r1", node_id="spoke-1", prompt="test",
        max_tokens=100, temperature=0.7,
        priority=InferencePriority.IMMEDIATE,
    )
    server.submit(req)
    result = server.process_next(timeout=1.0)
    assert result["error"] is not None
    assert "too slow" in result["error"].lower()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_server.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/inference_server.py
"""Inference server — Hub processes inference requests from Spokes.

Wraps the ModelEngine with a priority queue for managing concurrent
requests from multiple Spoke nodes.
"""
from __future__ import annotations

import logging
from typing import Optional

from homie_core.mesh.inference_queue import InferenceQueue, InferenceRequest

logger = logging.getLogger(__name__)


class InferenceServer:
    """Hub-side inference processor with priority queuing."""

    def __init__(self, model_engine, max_concurrent: int = 2):
        self._engine = model_engine
        self._queue = InferenceQueue(max_concurrent=max_concurrent)

    def process(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Direct inference — bypass the queue for immediate requests."""
        if not self._engine.is_loaded:
            raise RuntimeError(
                "No model loaded on Hub. Load a model with /model or homie init."
            )
        return self._engine.generate(
            prompt, max_tokens=max_tokens,
            temperature=temperature, stop=stop, timeout=timeout,
        )

    def submit(self, request: InferenceRequest) -> None:
        """Add a request to the priority queue."""
        self._queue.submit(request)

    def process_next(self, timeout: float = 5.0) -> Optional[dict]:
        """Get and process the next queued request.

        Returns dict with {request_id, node_id, content, error} or None
        if queue is empty.
        """
        req = self._queue.get(timeout=timeout)
        if req is None:
            return None

        self._queue.mark_active(req.request_id)
        try:
            content = self.process(
                prompt=req.prompt,
                max_tokens=req.max_tokens,
                temperature=req.temperature,
                stop=req.stop,
            )
            return {
                "request_id": req.request_id,
                "node_id": req.node_id,
                "content": content,
                "error": None,
            }
        except Exception as e:
            logger.error("Inference error for %s: %s", req.request_id, e)
            return {
                "request_id": req.request_id,
                "node_id": req.node_id,
                "content": "",
                "error": str(e),
            }
        finally:
            self._queue.mark_done(req.request_id)

    def queue_stats(self) -> dict:
        return self._queue.stats()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_inference_server.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/inference_server.py tests/unit/test_mesh/test_inference_server.py
git commit -m "feat(mesh): add Hub inference server with priority queue processing"
```

---

### Task 4: Mesh-Aware Inference Router

**Files:**
- Create: `src/homie_core/mesh/mesh_inference_router.py`
- Test: `tests/unit/test_mesh/test_mesh_inference_router.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_mesh_inference_router.py
from unittest.mock import MagicMock
from homie_core.mesh.mesh_inference_router import MeshInferenceRouter


def _make_mock_engine(loaded=True, response="local response"):
    engine = MagicMock()
    engine.is_loaded = loaded
    engine.generate.return_value = response
    engine.stream.return_value = iter([response])
    return engine


def _make_mock_mesh_client(available=True, response="hub response"):
    client = MagicMock()
    client.is_available = available
    client.generate.return_value = response
    client.stream.return_value = iter([response])
    return client


def test_local_model_used_first():
    """When local model is loaded, it's used instead of mesh."""
    engine = _make_mock_engine(loaded=True, response="local")
    client = _make_mock_mesh_client(available=True, response="hub")
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    result = router.generate("prompt")
    assert result == "local"
    engine.generate.assert_called_once()
    client.generate.assert_not_called()


def test_fallback_to_hub():
    """When local model not loaded, falls back to mesh Hub."""
    engine = _make_mock_engine(loaded=False)
    client = _make_mock_mesh_client(available=True, response="from hub")
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    result = router.generate("prompt")
    assert result == "from hub"
    client.generate.assert_called_once()


def test_fallback_to_qubrid():
    """When both local and hub unavailable, falls back to cloud."""
    engine = _make_mock_engine(loaded=False)
    client = _make_mock_mesh_client(available=False)
    qubrid = MagicMock()
    qubrid.is_available = True
    qubrid.generate.return_value = "cloud response"

    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client,
        qubrid_client=qubrid,
        priority=["local", "hub", "qubrid"],
    )
    result = router.generate("prompt")
    assert result == "cloud response"


def test_all_sources_unavailable_raises():
    """Raises RuntimeError when all sources are down."""
    engine = _make_mock_engine(loaded=False)
    client = _make_mock_mesh_client(available=False)
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    try:
        router.generate("prompt")
        assert False, "Should have raised"
    except RuntimeError as e:
        assert "unavailable" in str(e).lower()


def test_stream_uses_local():
    """Streaming uses local model when available."""
    engine = _make_mock_engine(loaded=True, response="token")
    client = _make_mock_mesh_client(available=True)
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    tokens = list(router.stream("prompt"))
    assert "".join(tokens) == "token"
    engine.stream.assert_called_once()


def test_stream_fallback_to_hub():
    """Streaming falls back to Hub when local unavailable."""
    engine = _make_mock_engine(loaded=False)
    client = _make_mock_mesh_client(available=True, response="hub token")
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    tokens = list(router.stream("prompt"))
    assert "".join(tokens) == "hub token"
    client.stream.assert_called_once()


def test_active_source_property():
    """active_source reports which source would be used."""
    engine = _make_mock_engine(loaded=True)
    client = _make_mock_mesh_client(available=True)
    router = MeshInferenceRouter(
        model_engine=engine, mesh_client=client, priority=["local", "hub"],
    )
    assert router.active_source == "Local"

    engine.is_loaded = False
    assert router.active_source == "Mesh Hub"

    client.is_available = False
    assert router.active_source == "None"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_mesh_inference_router.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/mesh_inference_router.py
"""Mesh-aware inference router — extends the fallback chain with mesh sources.

Priority chain: local -> hub -> lan_peer -> qubrid -> vertex
This replaces the existing InferenceRouter when mesh is enabled.
"""
from __future__ import annotations

import logging
from typing import Iterator, Optional

logger = logging.getLogger(__name__)


class MeshInferenceRouter:
    """Routes inference through local model, mesh hub, or cloud fallback."""

    def __init__(
        self,
        model_engine,
        mesh_client=None,
        qubrid_client=None,
        vertex_client=None,
        priority: Optional[list[str]] = None,
    ):
        self._engine = model_engine
        self._mesh_client = mesh_client
        self._qubrid = qubrid_client
        self._vertex = vertex_client
        self._priority = priority or ["local", "hub", "qubrid"]

    @property
    def active_source(self) -> str:
        """Report which source would be used for the next request."""
        for source in self._priority:
            if source == "local" and self._engine.is_loaded:
                return "Local"
            if source == "hub" and self._mesh_client and self._mesh_client.is_available:
                return "Mesh Hub"
            if source == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False):
                return "Cloud (Qubrid)"
            if source == "vertex" and self._vertex and getattr(self._vertex, "is_available", False):
                return "Cloud (Vertex AI)"
        return "None"

    def generate(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
        timeout: int = 120,
    ) -> str:
        """Generate a response using the highest-priority available source."""
        errors: list[str] = []

        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    return self._engine.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop, timeout=timeout,
                    )
                if source == "hub" and self._mesh_client and self._mesh_client.is_available:
                    return self._mesh_client.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                if source == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False):
                    return self._qubrid.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                if source == "vertex" and self._vertex and getattr(self._vertex, "is_available", False):
                    return self._vertex.generate(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
            except Exception as e:
                logger.warning("Inference source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )

    def stream(
        self,
        prompt: str,
        max_tokens: int = 1024,
        temperature: float = 0.7,
        stop: Optional[list[str]] = None,
    ) -> Iterator[str]:
        """Stream tokens from the highest-priority available source."""
        errors: list[str] = []

        for source in self._priority:
            try:
                if source == "local" and self._engine.is_loaded:
                    yield from self._engine.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "hub" and self._mesh_client and self._mesh_client.is_available:
                    yield from self._mesh_client.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "qubrid" and self._qubrid and getattr(self._qubrid, "is_available", False):
                    yield from self._qubrid.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
                if source == "vertex" and self._vertex and getattr(self._vertex, "is_available", False):
                    yield from self._vertex.stream(
                        prompt, max_tokens=max_tokens,
                        temperature=temperature, stop=stop,
                    )
                    return
            except Exception as e:
                logger.warning("Stream source '%s' failed: %s", source, e)
                errors.append(f"{source}: {e}")
                continue

        raise RuntimeError(
            "All inference sources unavailable. "
            f"Errors: {'; '.join(errors) if errors else 'no sources configured'}"
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_mesh_inference_router.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/mesh_inference_router.py tests/unit/test_mesh/test_mesh_inference_router.py
git commit -m "feat(mesh): add mesh-aware inference router with local/hub/cloud fallback chain"
```

---

### Task 5: Integration Test — End-to-End Distributed Inference

**Files:**
- Create: `tests/integration/test_distributed_inference.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_distributed_inference.py
"""End-to-end: Spoke routes inference to Hub, Hub processes and returns."""
from unittest.mock import MagicMock

from homie_core.mesh.inference_client import MeshInferenceClient
from homie_core.mesh.inference_server import InferenceServer
from homie_core.mesh.inference_queue import InferenceRequest, InferencePriority
from homie_core.mesh.mesh_inference_router import MeshInferenceRouter


def _make_hub_engine(response="Hub says hello"):
    engine = MagicMock()
    engine.is_loaded = True
    engine.generate.return_value = response
    engine.stream.return_value = iter([response])
    return engine


def test_spoke_to_hub_inference_flow():
    """Full flow: Spoke has no model -> routes to Hub -> gets response."""
    # Setup Hub with a model
    hub_engine = _make_hub_engine("The answer is 42")
    hub_server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    # Create a handler that bridges client to server
    def hub_handler(prompt, max_tokens, temperature, stop):
        return hub_server.process(
            prompt=prompt, max_tokens=max_tokens,
            temperature=temperature, stop=stop,
        )

    # Setup Spoke with no local model
    spoke_engine = MagicMock()
    spoke_engine.is_loaded = False

    mesh_client = MeshInferenceClient(
        node_id="spoke-1", hub_node_id="hub-1",
        request_handler=hub_handler,
    )

    # Spoke's router: local (unavailable) -> hub (available)
    router = MeshInferenceRouter(
        model_engine=spoke_engine,
        mesh_client=mesh_client,
        priority=["local", "hub"],
    )

    # Spoke generates — should route to Hub
    result = router.generate("What is the meaning of life?")
    assert result == "The answer is 42"
    spoke_engine.generate.assert_not_called()
    hub_engine.generate.assert_called_once()


def test_spoke_streams_from_hub():
    """Spoke streams tokens from Hub."""
    hub_engine = _make_hub_engine("streaming tokens")
    hub_server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    def hub_handler(prompt, max_tokens, temperature, stop):
        return hub_server.process(prompt=prompt, max_tokens=max_tokens,
                                  temperature=temperature, stop=stop)

    spoke_engine = MagicMock()
    spoke_engine.is_loaded = False

    mesh_client = MeshInferenceClient(
        node_id="spoke-1", hub_node_id="hub-1",
        request_handler=hub_handler,
    )

    router = MeshInferenceRouter(
        model_engine=spoke_engine, mesh_client=mesh_client,
        priority=["local", "hub"],
    )

    tokens = list(router.stream("test"))
    assert "".join(tokens) == "streaming tokens"


def test_hub_queue_processes_multiple_spokes():
    """Hub processes requests from multiple Spokes via queue."""
    hub_engine = _make_hub_engine()
    hub_engine.generate.side_effect = lambda p, **kw: f"reply to {p[:5]}"

    server = InferenceServer(model_engine=hub_engine, max_concurrent=2)

    # Submit requests from 3 different spokes
    for i in range(3):
        server.submit(InferenceRequest(
            request_id=f"r{i}", node_id=f"spoke-{i}",
            prompt=f"question {i}",
            max_tokens=100, temperature=0.7,
            priority=InferencePriority.IMMEDIATE,
        ))

    # Process all 3
    results = []
    for _ in range(3):
        r = server.process_next(timeout=1.0)
        assert r is not None
        results.append(r)

    assert len(results) == 3
    assert all(r["error"] is None for r in results)
    assert server.queue_stats()["completed"] == 3


def test_local_model_skips_mesh():
    """When Spoke has a local model, mesh is not used."""
    spoke_engine = MagicMock()
    spoke_engine.is_loaded = True
    spoke_engine.generate.return_value = "local answer"

    mesh_client = MagicMock()
    mesh_client.is_available = True

    router = MeshInferenceRouter(
        model_engine=spoke_engine, mesh_client=mesh_client,
        priority=["local", "hub"],
    )
    result = router.generate("test")
    assert result == "local answer"
    mesh_client.generate.assert_not_called()
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_distributed_inference.py -v`
Expected: All 4 tests PASS

- [ ] **Step 3: Run ALL tests for regression check**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/ tests/unit/test_platform/ tests/unit/test_network/test_discovery_mesh.py tests/integration/test_mesh_smoke.py tests/integration/test_mesh_sync.py tests/integration/test_distributed_inference.py -v`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_distributed_inference.py
git commit -m "feat(mesh): add end-to-end distributed inference integration tests"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | Inference Queue | `mesh/inference_queue.py` | 6 |
| 2 | Inference Client | `mesh/inference_client.py` | 5 |
| 3 | Inference Server | `mesh/inference_server.py` | 5 |
| 4 | Mesh Inference Router | `mesh/mesh_inference_router.py` | 7 |
| 5 | Integration Tests | `test_distributed_inference.py` | 4 |

**Total: 5 tasks, 27 tests, 4 new source files, 5 new test files**

After Phase 3, Homie nodes can:
- Route inference to Hub when no local model is available
- Queue requests with priority (IMMEDIATE > BACKGROUND > BATCH)
- Process concurrent requests up to a configurable limit
- Fall back through local → hub → cloud chain automatically
- Stream tokens from Hub to Spoke
- Handle errors gracefully (model timeout, no model loaded)
