# Distributed Mesh — Phase 4: Unified Context & Cross-Device Awareness

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Each node publishes its local context (active window, activity type, flow state, machine metrics) as mesh events. The Hub aggregates all node contexts into a `UnifiedUserModel`. The Brain's PERCEIVE stage gains cross-device awareness. Context handoff detects when the user switches devices and carries conversation state.

**Architecture:** `NodeContext` dataclass captures per-node state, published as `context.activity_changed` events via MeshManager. `UnifiedUserModel` on Hub aggregates all NodeContexts. `CrossDevicePerceiver` extends the Brain's PERCEIVE stage with mesh context. `ContextHandoff` detects device switches and offers continuity.

**Tech Stack:** Python 3.11+, existing mesh events/sync, existing SituationalAwareness from cognitive_arch.py

**Builds on:** Phase 1-3 (mesh identity, events, sync, inference routing)

---

## File Structure

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/node_context.py` | NodeContext dataclass + publish/collect from local sensors |
| `src/homie_core/mesh/unified_user_model.py` | Aggregates contexts from all nodes into unified view |
| `src/homie_core/mesh/context_handoff.py` | Detects device switches, offers conversation continuity |
| `src/homie_core/mesh/cross_device_perceiver.py` | Extends Brain PERCEIVE stage with mesh awareness |
| `tests/unit/test_mesh/test_node_context.py` | NodeContext tests |
| `tests/unit/test_mesh/test_unified_user_model.py` | Unified model tests |
| `tests/unit/test_mesh/test_context_handoff.py` | Handoff tests |
| `tests/unit/test_mesh/test_cross_device_perceiver.py` | Perceiver tests |
| `tests/integration/test_cross_device_flow.py` | End-to-end cross-device test |

---

### Task 1: NodeContext — Per-Node State Snapshot

**Files:**
- Create: `src/homie_core/mesh/node_context.py`
- Test: `tests/unit/test_mesh/test_node_context.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_node_context.py
from homie_core.mesh.node_context import NodeContext, collect_local_context


def test_node_context_creation():
    ctx = NodeContext(
        node_id="n1", node_name="desktop",
        active_window="VS Code — main.py",
        active_process="code.exe",
        activity_type="coding",
        activity_confidence=0.9,
        minutes_active=45.0,
        flow_score=0.8,
        cpu_usage=34.0,
        ram_usage_gb=18.0,
        gpu_usage=67.0,
    )
    assert ctx.node_id == "n1"
    assert ctx.activity_type == "coding"
    assert ctx.flow_score == 0.8


def test_node_context_to_dict():
    ctx = NodeContext(node_id="n1", node_name="laptop")
    d = ctx.to_dict()
    assert d["node_id"] == "n1"
    assert d["node_name"] == "laptop"
    assert "last_updated" in d


def test_node_context_from_dict():
    ctx = NodeContext(node_id="n1", node_name="box", activity_type="browsing")
    d = ctx.to_dict()
    restored = NodeContext.from_dict(d)
    assert restored.node_id == "n1"
    assert restored.activity_type == "browsing"


def test_node_context_to_event_payload():
    """Context converts to a payload suitable for mesh events."""
    ctx = NodeContext(
        node_id="n1", node_name="desktop",
        active_window="Chrome", activity_type="browsing",
    )
    payload = ctx.to_event_payload()
    assert payload["node_id"] == "n1"
    assert payload["activity_type"] == "browsing"


def test_node_context_summary():
    """Human-readable summary for prompt injection."""
    ctx = NodeContext(
        node_id="n1", node_name="desktop",
        active_window="VS Code — sync.py",
        activity_type="coding",
        minutes_active=30.0,
        flow_score=0.85,
    )
    summary = ctx.summary()
    assert "coding" in summary
    assert "desktop" in summary


def test_collect_local_context():
    """collect_local_context returns a NodeContext with real system data."""
    ctx = collect_local_context(node_id="test", node_name="test-box")
    assert ctx.node_id == "test"
    assert ctx.cpu_usage >= 0
    assert ctx.ram_usage_gb > 0


def test_context_is_idle():
    ctx = NodeContext(node_id="n1", node_name="n", idle_minutes=10.0)
    assert ctx.is_idle is True

    active = NodeContext(node_id="n1", node_name="n", idle_minutes=0.0, minutes_active=5.0)
    assert active.is_idle is False
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_node_context.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/node_context.py
"""Per-node context snapshot — what the user is doing on this machine."""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass, field
from typing import Optional

import psutil

from homie_core.utils import utc_now


@dataclass
class NodeContext:
    """Snapshot of a single node's current state."""

    node_id: str
    node_name: str

    # Activity
    active_window: str = ""
    active_process: str = ""
    activity_type: str = "idle"
    activity_confidence: float = 0.0

    # Session
    minutes_active: float = 0.0
    idle_minutes: float = 0.0
    flow_score: float = 0.5
    session_start: str = ""

    # Machine
    cpu_usage: float = 0.0
    gpu_usage: Optional[float] = None
    ram_usage_gb: float = 0.0
    battery_pct: Optional[float] = None

    # Metadata
    last_updated: str = field(default_factory=lambda: utc_now().isoformat())

    @property
    def is_idle(self) -> bool:
        return self.idle_minutes >= 5.0 and self.minutes_active == 0.0

    def summary(self) -> str:
        """Human-readable summary for prompt injection."""
        parts = [f"{self.node_name}:"]
        if self.is_idle:
            parts.append("idle")
        else:
            if self.activity_type != "idle":
                parts.append(self.activity_type)
            if self.active_window:
                parts.append(f"in {self.active_window}")
            if self.minutes_active > 1:
                parts.append(f"({int(self.minutes_active)}min)")
            if self.flow_score > 0.7:
                parts.append("[focused]")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "node_name": self.node_name,
            "active_window": self.active_window, "active_process": self.active_process,
            "activity_type": self.activity_type, "activity_confidence": self.activity_confidence,
            "minutes_active": self.minutes_active, "idle_minutes": self.idle_minutes,
            "flow_score": self.flow_score, "session_start": self.session_start,
            "cpu_usage": self.cpu_usage, "gpu_usage": self.gpu_usage,
            "ram_usage_gb": self.ram_usage_gb, "battery_pct": self.battery_pct,
            "last_updated": self.last_updated,
        }

    def to_event_payload(self) -> dict:
        """Convert to a payload for context.activity_changed mesh events."""
        return self.to_dict()

    @classmethod
    def from_dict(cls, d: dict) -> NodeContext:
        return cls(
            node_id=d["node_id"], node_name=d["node_name"],
            active_window=d.get("active_window", ""),
            active_process=d.get("active_process", ""),
            activity_type=d.get("activity_type", "idle"),
            activity_confidence=d.get("activity_confidence", 0.0),
            minutes_active=d.get("minutes_active", 0.0),
            idle_minutes=d.get("idle_minutes", 0.0),
            flow_score=d.get("flow_score", 0.5),
            session_start=d.get("session_start", ""),
            cpu_usage=d.get("cpu_usage", 0.0),
            gpu_usage=d.get("gpu_usage"),
            ram_usage_gb=d.get("ram_usage_gb", 0.0),
            battery_pct=d.get("battery_pct"),
            last_updated=d.get("last_updated", ""),
        )


def collect_local_context(node_id: str, node_name: str) -> NodeContext:
    """Collect current system metrics into a NodeContext."""
    mem = psutil.virtual_memory()
    disk_path = "C:\\" if sys.platform == "win32" else "/"
    return NodeContext(
        node_id=node_id,
        node_name=node_name,
        cpu_usage=psutil.cpu_percent(interval=0.1),
        ram_usage_gb=round(mem.used / (1024 ** 3), 1),
        battery_pct=_get_battery(),
    )


def _get_battery() -> Optional[float]:
    try:
        bat = psutil.sensors_battery()
        return bat.percent if bat else None
    except Exception:
        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_node_context.py -v`
Expected: All 7 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/node_context.py tests/unit/test_mesh/test_node_context.py
git commit -m "feat(mesh): add NodeContext for per-node state snapshots"
```

---

### Task 2: Unified User Model — Cross-Device Aggregation

**Files:**
- Create: `src/homie_core/mesh/unified_user_model.py`
- Test: `tests/unit/test_mesh/test_unified_user_model.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_unified_user_model.py
from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel


def test_empty_model():
    model = UnifiedUserModel()
    assert model.active_nodes == []
    assert model.primary_node is None


def test_update_single_node():
    model = UnifiedUserModel()
    ctx = NodeContext(node_id="desktop", node_name="desktop",
                      activity_type="coding", minutes_active=30.0, flow_score=0.8)
    model.update_node(ctx)
    assert model.active_nodes == ["desktop"]
    assert model.primary_node == "desktop"


def test_update_multiple_nodes():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=30.0))
    model.update_node(NodeContext(node_id="laptop", node_name="laptop",
                                  activity_type="browsing", minutes_active=5.0))
    assert len(model.active_nodes) == 2
    # Desktop has more activity, so it's primary
    assert model.primary_node == "desktop"


def test_idle_node_not_primary():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  idle_minutes=10.0, minutes_active=0.0))
    model.update_node(NodeContext(node_id="laptop", node_name="laptop",
                                  activity_type="coding", minutes_active=5.0))
    assert model.primary_node == "laptop"


def test_activity_summary():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", active_window="VS Code"))
    summary = model.activity_summary()
    assert "coding" in summary
    assert "desktop" in summary


def test_to_context_block():
    """Generates a prompt-injectable context block."""
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=30.0,
                                  flow_score=0.85))
    model.update_node(NodeContext(node_id="laptop", node_name="laptop",
                                  idle_minutes=10.0, minutes_active=0.0))
    block = model.to_context_block()
    assert "[CROSS-DEVICE CONTEXT]" in block
    assert "desktop" in block


def test_node_contexts_dict():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="n1", node_name="box1"))
    model.update_node(NodeContext(node_id="n2", node_name="box2"))
    assert len(model.node_contexts) == 2


def test_remove_stale_node():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="n1", node_name="box"))
    assert len(model.node_contexts) == 1
    model.remove_node("n1")
    assert len(model.node_contexts) == 0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_unified_user_model.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/unified_user_model.py
"""Unified user model — aggregates context from all mesh nodes.

Provides a complete picture of what the user is doing across all devices.
Used by the Brain's PERCEIVE stage for cross-device awareness.
"""
from __future__ import annotations

from typing import Optional

from homie_core.mesh.node_context import NodeContext


class UnifiedUserModel:
    """Cross-device user state aggregated from all mesh nodes."""

    def __init__(self):
        self._contexts: dict[str, NodeContext] = {}

    @property
    def node_contexts(self) -> dict[str, NodeContext]:
        return dict(self._contexts)

    @property
    def active_nodes(self) -> list[str]:
        """Node IDs that have reported context."""
        return list(self._contexts.keys())

    @property
    def primary_node(self) -> Optional[str]:
        """The node where the user is most active right now.

        Determined by: non-idle + highest minutes_active + highest flow_score.
        """
        if not self._contexts:
            return None

        candidates = [
            (nid, ctx) for nid, ctx in self._contexts.items()
            if not ctx.is_idle
        ]
        if not candidates:
            # All idle — pick the one most recently active
            return max(self._contexts, key=lambda n: self._contexts[n].last_updated)

        # Score: minutes_active weighted + flow bonus
        def score(item):
            _, ctx = item
            return ctx.minutes_active + (ctx.flow_score * 10)

        return max(candidates, key=score)[0]

    def update_node(self, context: NodeContext) -> None:
        """Update the context for a node."""
        self._contexts[context.node_id] = context

    def remove_node(self, node_id: str) -> None:
        """Remove a node's context (e.g., node went offline)."""
        self._contexts.pop(node_id, None)

    def activity_summary(self) -> str:
        """One-line summary of what the user is doing across all devices."""
        if not self._contexts:
            return "No devices connected"
        parts = []
        for ctx in self._contexts.values():
            parts.append(ctx.summary())
        return ", ".join(parts)

    def to_context_block(self) -> str:
        """Render as a [CROSS-DEVICE CONTEXT] block for prompt injection."""
        if not self._contexts:
            return ""

        lines = ["[CROSS-DEVICE CONTEXT]"]

        primary = self.primary_node
        for node_id, ctx in self._contexts.items():
            marker = " (primary)" if node_id == primary else ""
            lines.append(f"  {ctx.summary()}{marker}")

        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_unified_user_model.py -v`
Expected: All 8 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/unified_user_model.py tests/unit/test_mesh/test_unified_user_model.py
git commit -m "feat(mesh): add UnifiedUserModel for cross-device context aggregation"
```

---

### Task 3: Context Handoff — Device Switch Detection

**Files:**
- Create: `src/homie_core/mesh/context_handoff.py`
- Test: `tests/unit/test_mesh/test_context_handoff.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_context_handoff.py
from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.context_handoff import ContextHandoff


def test_no_handoff_on_first_activity():
    handoff = ContextHandoff()
    result = handoff.check(
        local_node_id="desktop",
        local_context=NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=5.0),
        peer_contexts={},
    )
    assert result is None


def test_detects_switch_from_peer():
    """User was active on laptop, now active on desktop."""
    handoff = ContextHandoff()

    # Simulate: laptop was active recently
    laptop_ctx = NodeContext(
        node_id="laptop", node_name="laptop",
        activity_type="coding", active_window="VS Code — sync.py",
        minutes_active=30.0, idle_minutes=0.0,
    )
    # First check: laptop is active, desktop idle
    handoff.check(
        local_node_id="desktop",
        local_context=NodeContext(node_id="desktop", node_name="desktop",
                                  idle_minutes=10.0, minutes_active=0.0),
        peer_contexts={"laptop": laptop_ctx},
    )

    # Now: laptop goes idle, desktop becomes active
    laptop_idle = NodeContext(
        node_id="laptop", node_name="laptop",
        idle_minutes=2.0, minutes_active=0.0,
    )
    result = handoff.check(
        local_node_id="desktop",
        local_context=NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=1.0),
        peer_contexts={"laptop": laptop_idle},
    )
    assert result is not None
    assert result["from_node"] == "laptop"
    assert result["to_node"] == "desktop"
    assert "coding" in result["previous_activity"]


def test_no_duplicate_handoff():
    """Same switch doesn't trigger twice."""
    handoff = ContextHandoff()

    laptop_active = NodeContext(node_id="laptop", node_name="laptop",
                                activity_type="coding", minutes_active=20.0)
    handoff.check("desktop",
                  NodeContext(node_id="desktop", node_name="desktop", idle_minutes=5.0, minutes_active=0.0),
                  {"laptop": laptop_active})

    laptop_idle = NodeContext(node_id="laptop", node_name="laptop", idle_minutes=2.0, minutes_active=0.0)
    desktop_active = NodeContext(node_id="desktop", node_name="desktop",
                                 activity_type="coding", minutes_active=1.0)

    r1 = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})
    assert r1 is not None

    r2 = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})
    assert r2 is None  # Already handled


def test_handoff_context_message():
    """Handoff result contains a human-readable message."""
    handoff = ContextHandoff()

    handoff.check("desktop",
                  NodeContext(node_id="desktop", node_name="desktop", idle_minutes=5.0, minutes_active=0.0),
                  {"laptop": NodeContext(node_id="laptop", node_name="laptop",
                                         activity_type="browsing",
                                         active_window="Chrome — Stack Overflow",
                                         minutes_active=15.0)})

    result = handoff.check("desktop",
                           NodeContext(node_id="desktop", node_name="desktop",
                                       activity_type="coding", minutes_active=1.0),
                           {"laptop": NodeContext(node_id="laptop", node_name="laptop",
                                                   idle_minutes=2.0, minutes_active=0.0)})
    assert result is not None
    assert "message" in result
    assert "laptop" in result["message"]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_context_handoff.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/context_handoff.py
"""Context handoff — detect device switches and offer continuity."""
from __future__ import annotations

from typing import Optional

from homie_core.mesh.node_context import NodeContext


class ContextHandoff:
    """Detects when user switches from one device to another.

    Tracks the last-known active peer. When a peer goes idle and the local
    node becomes active, it's a device switch.
    """

    def __init__(self):
        self._last_active_peer: Optional[str] = None
        self._last_peer_context: Optional[NodeContext] = None
        self._handled_switches: set[tuple[str, str]] = set()

    def check(
        self,
        local_node_id: str,
        local_context: NodeContext,
        peer_contexts: dict[str, NodeContext],
    ) -> Optional[dict]:
        """Check for a device switch.

        Returns a handoff dict if a switch is detected, or None.
        """
        # Find the most active peer
        active_peer = None
        active_peer_ctx = None
        for peer_id, ctx in peer_contexts.items():
            if not ctx.is_idle and ctx.minutes_active > 0:
                if active_peer_ctx is None or ctx.minutes_active > active_peer_ctx.minutes_active:
                    active_peer = peer_id
                    active_peer_ctx = ctx

        # Track the active peer for next check
        if active_peer and local_context.is_idle:
            self._last_active_peer = active_peer
            self._last_peer_context = active_peer_ctx
            return None

        # Detect switch: we were idle, peer was active, now we're active and peer is idle
        if (
            self._last_active_peer
            and self._last_peer_context
            and not local_context.is_idle
            and local_context.minutes_active > 0
        ):
            peer_id = self._last_active_peer
            peer_ctx = peer_contexts.get(peer_id)

            # Peer must now be idle (or at least less active)
            if peer_ctx and (peer_ctx.is_idle or peer_ctx.minutes_active == 0):
                switch_key = (peer_id, local_node_id)
                if switch_key not in self._handled_switches:
                    self._handled_switches.add(switch_key)
                    result = {
                        "from_node": peer_id,
                        "to_node": local_node_id,
                        "previous_activity": self._last_peer_context.activity_type,
                        "previous_window": self._last_peer_context.active_window,
                        "previous_minutes": self._last_peer_context.minutes_active,
                        "message": (
                            f"You were {self._last_peer_context.activity_type} on "
                            f"{self._last_peer_context.node_name}"
                            + (f" ({self._last_peer_context.active_window})"
                               if self._last_peer_context.active_window else "")
                            + ". Want to continue here?"
                        ),
                    }
                    self._last_active_peer = None
                    self._last_peer_context = None
                    return result

        return None
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_context_handoff.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/context_handoff.py tests/unit/test_mesh/test_context_handoff.py
git commit -m "feat(mesh): add context handoff for device switch detection and continuity"
```

---

### Task 4: Cross-Device Perceiver — Brain Integration

**Files:**
- Create: `src/homie_core/mesh/cross_device_perceiver.py`
- Test: `tests/unit/test_mesh/test_cross_device_perceiver.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_cross_device_perceiver.py
from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel
from homie_core.mesh.cross_device_perceiver import CrossDevicePerceiver


def test_perceiver_with_no_mesh():
    """When no mesh data, returns empty string."""
    perceiver = CrossDevicePerceiver(unified_model=None)
    assert perceiver.get_context_block() == ""


def test_perceiver_with_single_node():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=30.0))
    perceiver = CrossDevicePerceiver(unified_model=model)
    block = perceiver.get_context_block()
    assert "CROSS-DEVICE" in block
    assert "desktop" in block


def test_perceiver_with_multiple_nodes():
    model = UnifiedUserModel()
    model.update_node(NodeContext(node_id="desktop", node_name="desktop",
                                  activity_type="coding", minutes_active=30.0, flow_score=0.9))
    model.update_node(NodeContext(node_id="laptop", node_name="laptop",
                                  idle_minutes=10.0, minutes_active=0.0))
    perceiver = CrossDevicePerceiver(unified_model=model)
    block = perceiver.get_context_block()
    assert "primary" in block
    assert "desktop" in block
    assert "laptop" in block


def test_perceiver_handoff_message():
    """Perceiver includes handoff message when device switch detected."""
    model = UnifiedUserModel()
    perceiver = CrossDevicePerceiver(unified_model=model)

    handoff = {
        "from_node": "laptop",
        "to_node": "desktop",
        "message": "You were coding on laptop. Want to continue here?",
    }
    perceiver.set_pending_handoff(handoff)
    block = perceiver.get_context_block()
    assert "coding on laptop" in block


def test_perceiver_clears_handoff_after_use():
    model = UnifiedUserModel()
    perceiver = CrossDevicePerceiver(unified_model=model)
    perceiver.set_pending_handoff({"message": "test handoff"})
    perceiver.get_context_block()  # Should include handoff
    block2 = perceiver.get_context_block()  # Should NOT include it again
    assert "test handoff" not in block2
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_cross_device_perceiver.py -v`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/cross_device_perceiver.py
"""Cross-device perceiver — extends Brain's PERCEIVE with mesh awareness.

Generates a [CROSS-DEVICE CONTEXT] block for prompt injection,
including unified user model state and pending device handoffs.
"""
from __future__ import annotations

from typing import Optional

from homie_core.mesh.unified_user_model import UnifiedUserModel


class CrossDevicePerceiver:
    """Provides cross-device context for the Brain's cognitive pipeline."""

    def __init__(self, unified_model: Optional[UnifiedUserModel] = None):
        self._model = unified_model
        self._pending_handoff: Optional[dict] = None

    def set_pending_handoff(self, handoff: dict) -> None:
        """Set a pending device handoff message to include in next context."""
        self._pending_handoff = handoff

    def get_context_block(self) -> str:
        """Generate the cross-device context block for prompt injection.

        Returns empty string if no mesh data is available.
        """
        parts = []

        # Unified model context
        if self._model:
            model_block = self._model.to_context_block()
            if model_block:
                parts.append(model_block)

        # Pending handoff
        if self._pending_handoff:
            msg = self._pending_handoff.get("message", "")
            if msg:
                parts.append(f"\n[DEVICE SWITCH]\n{msg}")
            self._pending_handoff = None  # Clear after use

        return "\n".join(parts)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/test_cross_device_perceiver.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/cross_device_perceiver.py tests/unit/test_mesh/test_cross_device_perceiver.py
git commit -m "feat(mesh): add CrossDevicePerceiver for Brain PERCEIVE integration"
```

---

### Task 5: Integration Test — Full Cross-Device Flow

**Files:**
- Create: `tests/integration/test_cross_device_flow.py`

- [ ] **Step 1: Write the integration test**

```python
# tests/integration/test_cross_device_flow.py
"""End-to-end: context flows from nodes through unified model to Brain context."""
import time

from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.node_context import NodeContext
from homie_core.mesh.unified_user_model import UnifiedUserModel
from homie_core.mesh.context_handoff import ContextHandoff
from homie_core.mesh.cross_device_perceiver import CrossDevicePerceiver


def test_full_cross_device_context_flow(tmp_path):
    """Desktop and laptop contexts aggregate into unified model and perceiver."""
    # Setup two nodes
    desktop_id = NodeIdentity.generate()
    laptop_id = NodeIdentity.generate()
    desktop_mgr = MeshManager(identity=desktop_id, data_dir=tmp_path / "desktop")
    laptop_mgr = MeshManager(identity=laptop_id, data_dir=tmp_path / "laptop")

    # Create contexts
    desktop_ctx = NodeContext(
        node_id=desktop_id.node_id, node_name="desktop",
        activity_type="coding", active_window="VS Code — mesh.py",
        minutes_active=45.0, flow_score=0.85,
    )
    laptop_ctx = NodeContext(
        node_id=laptop_id.node_id, node_name="laptop",
        activity_type="browsing", active_window="Chrome — docs",
        minutes_active=10.0,
    )

    # Emit contexts as mesh events
    desktop_mgr.emit("context", "activity_changed", desktop_ctx.to_event_payload())
    laptop_mgr.emit("context", "activity_changed", laptop_ctx.to_event_payload())

    # Hub aggregates into unified model
    model = UnifiedUserModel()
    model.update_node(desktop_ctx)
    model.update_node(laptop_ctx)

    assert len(model.active_nodes) == 2
    assert model.primary_node == desktop_id.node_id  # More active

    # Perceiver generates context block
    perceiver = CrossDevicePerceiver(unified_model=model)
    block = perceiver.get_context_block()
    assert "CROSS-DEVICE" in block
    assert "desktop" in block
    assert "laptop" in block
    assert "primary" in block


def test_device_switch_handoff_flow(tmp_path):
    """User switches from laptop to desktop, handoff is detected."""
    handoff = ContextHandoff()
    model = UnifiedUserModel()
    perceiver = CrossDevicePerceiver(unified_model=model)

    # Phase 1: Laptop active, desktop idle
    laptop_active = NodeContext(
        node_id="laptop", node_name="laptop",
        activity_type="coding", active_window="VS Code — api.py",
        minutes_active=30.0,
    )
    desktop_idle = NodeContext(
        node_id="desktop", node_name="desktop",
        idle_minutes=10.0, minutes_active=0.0,
    )
    model.update_node(laptop_active)
    model.update_node(desktop_idle)
    result = handoff.check("desktop", desktop_idle, {"laptop": laptop_active})
    assert result is None  # No switch yet

    # Phase 2: User moves to desktop
    laptop_idle = NodeContext(
        node_id="laptop", node_name="laptop",
        idle_minutes=2.0, minutes_active=0.0,
    )
    desktop_active = NodeContext(
        node_id="desktop", node_name="desktop",
        activity_type="coding", minutes_active=1.0,
    )
    model.update_node(laptop_idle)
    model.update_node(desktop_active)
    result = handoff.check("desktop", desktop_active, {"laptop": laptop_idle})

    assert result is not None
    assert result["from_node"] == "laptop"
    assert "coding" in result["previous_activity"]

    # Perceiver includes handoff
    perceiver.set_pending_handoff(result)
    block = perceiver.get_context_block()
    assert "DEVICE SWITCH" in block
    assert "laptop" in block


def test_context_events_sync_between_nodes(tmp_path):
    """Context events from one node sync to the other."""
    from homie_core.mesh.sync_protocol import SyncRequest

    node_a = NodeIdentity.generate()
    node_b = NodeIdentity.generate()
    mgr_a = MeshManager(identity=node_a, data_dir=tmp_path / "a")
    mgr_b = MeshManager(identity=node_b, data_dir=tmp_path / "b")

    # Node A emits a context event
    ctx = NodeContext(node_id=node_a.node_id, node_name="node-a",
                      activity_type="coding", minutes_active=20.0)
    mgr_a.emit("context", "activity_changed", ctx.to_event_payload())

    # Node B syncs from A
    req = SyncRequest(node_id=node_b.node_id, last_event_id=None, vector_clock={})
    resp = mgr_a.handle_sync_request(req)
    applied = mgr_b.apply_sync_response(resp)
    assert applied == 1

    # Reconstruct NodeContext from synced event
    events = mgr_b.events_since(None)
    assert len(events) == 1
    assert events[0].category == "context"
    restored_ctx = NodeContext.from_dict(events[0].payload)
    assert restored_ctx.activity_type == "coding"
    assert restored_ctx.node_name == "node-a"
```

- [ ] **Step 2: Run integration tests**

Run: `.venv/Scripts/python.exe -m pytest tests/integration/test_cross_device_flow.py -v`
Expected: All 3 tests PASS

- [ ] **Step 3: Run ALL mesh tests**

Run: `.venv/Scripts/python.exe -m pytest tests/unit/test_mesh/ tests/unit/test_platform/ tests/unit/test_network/test_discovery_mesh.py tests/integration/ -v --ignore=tests/integration/test_finetune_integration.py --ignore=tests/integration/test_daemon_startup.py`
Expected: All tests PASS

- [ ] **Step 4: Commit**

```bash
git add tests/integration/test_cross_device_flow.py
git commit -m "feat(mesh): add cross-device context flow integration tests"
```

---

## Summary

| Task | Component | Files | Tests |
|------|-----------|-------|-------|
| 1 | NodeContext | `mesh/node_context.py` | 7 |
| 2 | Unified User Model | `mesh/unified_user_model.py` | 8 |
| 3 | Context Handoff | `mesh/context_handoff.py` | 4 |
| 4 | Cross-Device Perceiver | `mesh/cross_device_perceiver.py` | 5 |
| 5 | Integration Tests | `test_cross_device_flow.py` | 3 |

**Total: 5 tasks, 27 tests, 4 new source files, 5 new test files**

After Phase 4, Homie nodes can:
- Capture per-node context (activity, flow state, machine metrics)
- Publish context as mesh events that sync to Hub
- Aggregate all node contexts into a UnifiedUserModel
- Determine which device is "primary" (most active)
- Detect device switches and offer conversation continuity
- Inject cross-device awareness into the Brain's PERCEIVE stage
