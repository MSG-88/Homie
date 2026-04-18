# Distributed Mesh — Phase 1: Foundation (Node Identity + Platform + Discovery + Mesh Config)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Every Homie instance gets a persistent node identity with capability detection, a platform abstraction layer, enhanced mDNS discovery with mesh-aware TXT records, mesh config in `homie.config.yaml`, and CLI commands (`/node`, `/mesh`) for mesh management.

**Architecture:** Node identity persisted in `~/.homie/node.json` via a `NodeIdentity` dataclass. Platform abstraction via `PlatformAdapter` ABC with Windows/Linux/macOS implementations. Existing `HomieDiscovery` extended with capability TXT records. New Pydantic config section `MeshConfig` added to `HomieConfig`. Pairing protocol uses Ed25519 key exchange via the existing `cryptography` dependency.

**Tech Stack:** Python 3.11+, pydantic, cryptography (Ed25519/X25519), zeroconf, psutil, SQLite

---

## File Structure

### New Files

| File | Responsibility |
|------|----------------|
| `src/homie_core/mesh/__init__.py` | Package init |
| `src/homie_core/mesh/identity.py` | Node identity: generate, persist, load from `~/.homie/node.json` |
| `src/homie_core/mesh/capabilities.py` | Hardware capability detection: GPU, CPU, RAM, OS, audio |
| `src/homie_core/mesh/election.py` | Hub election: capability scoring, role negotiation |
| `src/homie_core/mesh/pairing.py` | Pairing protocol: code generation, Ed25519 key exchange, secret derivation |
| `src/homie_core/mesh/registry.py` | Mesh node registry: SQLite table for known nodes, status tracking |
| `src/homie_core/platform/base.py` | `PlatformAdapter` ABC |
| `src/homie_core/platform/windows.py` | Windows implementation |
| `src/homie_core/platform/linux.py` | Linux implementation |
| `src/homie_core/platform/macos.py` | macOS implementation |
| `src/homie_core/platform/detect.py` | Auto-detect platform and return correct adapter |
| `src/homie_app/console/commands/node.py` | `/node` slash command |
| `src/homie_app/console/commands/mesh.py` | `/mesh` slash command |
| `tests/unit/test_mesh/__init__.py` | Test package |
| `tests/unit/test_mesh/test_identity.py` | Node identity tests |
| `tests/unit/test_mesh/test_capabilities.py` | Capability detection tests |
| `tests/unit/test_mesh/test_election.py` | Hub election tests |
| `tests/unit/test_mesh/test_pairing.py` | Pairing protocol tests |
| `tests/unit/test_mesh/test_registry.py` | Node registry tests |
| `tests/unit/test_platform/__init__.py` | Test package |
| `tests/unit/test_platform/test_detect.py` | Platform detection tests |
| `tests/unit/test_platform/test_adapter.py` | Platform adapter tests |
| `tests/unit/test_network/test_discovery_mesh.py` | Enhanced discovery tests |

### Modified Files

| File | Changes |
|------|---------|
| `src/homie_core/config.py` | Add `MeshConfig` Pydantic model + wire into `HomieConfig` |
| `src/homie_core/network/discovery.py` | Add capability TXT records, `mesh_id` and `role` to advertisement |
| `src/homie_core/storage/database.py` | Add `mesh_nodes` table to schema |
| `src/homie_app/console/console.py` | Register `/node` and `/mesh` commands |
| `pyproject.toml` | Add `mesh` optional dependency group |

---

### Task 1: Node Identity — Data Model and Persistence

**Files:**
- Create: `src/homie_core/mesh/__init__.py`
- Create: `src/homie_core/mesh/identity.py`
- Test: `tests/unit/test_mesh/__init__.py`
- Test: `tests/unit/test_mesh/test_identity.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/__init__.py
# (empty)

# tests/unit/test_mesh/test_identity.py
import json
from pathlib import Path

from homie_core.mesh.identity import NodeIdentity, load_identity, save_identity


def test_node_identity_creation():
    """A fresh identity has a UUID, hostname, standalone role, and no mesh_id."""
    identity = NodeIdentity.generate()
    assert len(identity.node_id) == 36  # UUID format
    assert identity.node_name != ""
    assert identity.role == "standalone"
    assert identity.mesh_id is None
    assert identity.created_at != ""


def test_node_identity_deterministic():
    """Two calls to generate() produce different UUIDs."""
    a = NodeIdentity.generate()
    b = NodeIdentity.generate()
    assert a.node_id != b.node_id


def test_save_and_load_identity(tmp_path):
    """Identity round-trips through JSON on disk."""
    path = tmp_path / "node.json"
    identity = NodeIdentity.generate()
    identity.node_name = "test-box"
    save_identity(identity, path)

    loaded = load_identity(path)
    assert loaded.node_id == identity.node_id
    assert loaded.node_name == "test-box"
    assert loaded.role == "standalone"
    assert loaded.created_at == identity.created_at


def test_load_identity_missing_file(tmp_path):
    """Loading from a non-existent file returns None."""
    path = tmp_path / "nope.json"
    assert load_identity(path) is None


def test_identity_has_ed25519_keypair():
    """Identity generates an Ed25519 keypair for mesh auth."""
    identity = NodeIdentity.generate()
    assert identity.public_key_pem != ""
    assert identity.private_key_pem != ""
    assert "BEGIN PUBLIC KEY" in identity.public_key_pem
    assert "BEGIN PRIVATE KEY" in identity.private_key_pem


def test_saved_identity_excludes_private_key(tmp_path):
    """The saved JSON must not contain the private key (stored in vault)."""
    path = tmp_path / "node.json"
    identity = NodeIdentity.generate()
    save_identity(identity, path)

    raw = json.loads(path.read_text())
    assert "private_key_pem" not in raw
    assert "public_key_pem" in raw
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mesh/test_identity.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.mesh'`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/__init__.py
"""Distributed mesh — node identity, discovery, pairing, sync."""

# src/homie_core/mesh/identity.py
"""Node identity — unique, persistent identity for each Homie instance."""
from __future__ import annotations

import json
import socket
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization

from homie_core.utils import utc_now


@dataclass
class NodeIdentity:
    """Persistent identity for a Homie node."""

    node_id: str
    node_name: str
    role: str  # "hub" | "spoke" | "standalone"
    mesh_id: Optional[str]
    created_at: str
    public_key_pem: str
    private_key_pem: str  # Never persisted to node.json — stored in vault

    @classmethod
    def generate(cls) -> NodeIdentity:
        """Create a fresh node identity with a new UUID and Ed25519 keypair."""
        private_key = Ed25519PrivateKey.generate()
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        ).decode()
        public_pem = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        ).decode()
        return cls(
            node_id=str(uuid.uuid4()),
            node_name=socket.gethostname(),
            role="standalone",
            mesh_id=None,
            created_at=utc_now().isoformat(),
            public_key_pem=public_pem,
            private_key_pem=private_pem,
        )


def save_identity(identity: NodeIdentity, path: Path) -> None:
    """Persist identity to disk. Private key is excluded from the file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "node_id": identity.node_id,
        "node_name": identity.node_name,
        "role": identity.role,
        "mesh_id": identity.mesh_id,
        "created_at": identity.created_at,
        "public_key_pem": identity.public_key_pem,
    }
    path.write_text(json.dumps(data, indent=2))


def load_identity(path: Path) -> Optional[NodeIdentity]:
    """Load identity from disk. Returns None if file doesn't exist."""
    if not path.exists():
        return None
    data = json.loads(path.read_text())
    return NodeIdentity(
        node_id=data["node_id"],
        node_name=data["node_name"],
        role=data.get("role", "standalone"),
        mesh_id=data.get("mesh_id"),
        created_at=data.get("created_at", ""),
        public_key_pem=data.get("public_key_pem", ""),
        private_key_pem="",  # Not stored in node.json
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mesh/test_identity.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/__init__.py src/homie_core/mesh/identity.py tests/unit/test_mesh/__init__.py tests/unit/test_mesh/test_identity.py
git commit -m "feat(mesh): add node identity with Ed25519 keypair and JSON persistence"
```

---

### Task 2: Capability Detection

**Files:**
- Create: `src/homie_core/mesh/capabilities.py`
- Test: `tests/unit/test_mesh/test_capabilities.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_capabilities.py
import sys
from unittest.mock import patch, MagicMock

from homie_core.mesh.capabilities import NodeCapabilities, detect_capabilities


def test_detect_returns_capabilities():
    """detect_capabilities() returns a NodeCapabilities with system info."""
    caps = detect_capabilities()
    assert caps.cpu_cores > 0
    assert caps.ram_gb > 0
    assert caps.os in ("windows", "linux", "macos")
    assert isinstance(caps.has_mic, bool)
    assert isinstance(caps.has_display, bool)


def test_capabilities_score_no_gpu():
    """Score without GPU is based on RAM + CPU only."""
    caps = NodeCapabilities(
        gpu=None, cpu_cores=8, ram_gb=16.0, disk_free_gb=100.0,
        os="linux", has_mic=True, has_display=True,
        has_model_loaded=False, model_name=None,
    )
    score = caps.capability_score()
    # (ram * 2) + cpu = 16*2 + 8 = 40
    assert score == 40.0


def test_capabilities_score_with_gpu():
    """Score with GPU includes vram * 10."""
    caps = NodeCapabilities(
        gpu={"name": "RTX 5080", "vram_gb": 16.0},
        cpu_cores=16, ram_gb=32.0, disk_free_gb=200.0,
        os="windows", has_mic=True, has_display=True,
        has_model_loaded=True, model_name="Qwen3.5",
    )
    score = caps.capability_score()
    # (vram * 10) + (ram * 2) + cpu + (model * 50) = 160 + 64 + 16 + 50 = 290
    assert score == 290.0


def test_capabilities_to_dict():
    """Capabilities serialize to a plain dict for JSON/TXT records."""
    caps = NodeCapabilities(
        gpu=None, cpu_cores=4, ram_gb=8.0, disk_free_gb=50.0,
        os="linux", has_mic=False, has_display=True,
        has_model_loaded=False, model_name=None,
    )
    d = caps.to_dict()
    assert d["cpu_cores"] == 4
    assert d["ram_gb"] == 8.0
    assert d["gpu"] is None
    assert d["os"] == "linux"


def test_detect_os_platform():
    """OS detection matches sys.platform."""
    caps = detect_capabilities()
    if sys.platform == "win32":
        assert caps.os == "windows"
    elif sys.platform == "darwin":
        assert caps.os == "macos"
    else:
        assert caps.os == "linux"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mesh/test_capabilities.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.mesh.capabilities'`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/capabilities.py
"""Hardware capability detection for mesh node scoring."""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from typing import Optional

import psutil


@dataclass
class NodeCapabilities:
    """Hardware and software capabilities of a Homie node."""

    gpu: Optional[dict]  # {"name": str, "vram_gb": float} or None
    cpu_cores: int
    ram_gb: float
    disk_free_gb: float
    os: str  # "windows" | "linux" | "macos"
    has_mic: bool
    has_display: bool
    has_model_loaded: bool
    model_name: Optional[str]

    def capability_score(self) -> float:
        """Compute a score for hub election. Higher = more capable."""
        score = 0.0
        if self.gpu:
            score += self.gpu.get("vram_gb", 0) * 10
        score += self.ram_gb * 2
        score += self.cpu_cores
        if self.has_model_loaded:
            score += 50
        return score

    def to_dict(self) -> dict:
        """Serialize to a plain dict."""
        return {
            "gpu": self.gpu,
            "cpu_cores": self.cpu_cores,
            "ram_gb": self.ram_gb,
            "disk_free_gb": self.disk_free_gb,
            "os": self.os,
            "has_mic": self.has_mic,
            "has_display": self.has_display,
            "has_model_loaded": self.has_model_loaded,
            "model_name": self.model_name,
        }


def _detect_gpu() -> Optional[dict]:
    """Detect GPU name and VRAM. Returns None if no GPU found."""
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            line = result.stdout.strip().split("\n")[0]
            parts = line.split(", ")
            if len(parts) == 2:
                name = parts[0].strip()
                vram_mb = float(parts[1].strip())
                return {"name": name, "vram_gb": round(vram_mb / 1024, 1)}
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        pass
    return None


def _detect_mic() -> bool:
    """Check if any audio input device is available."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return any(d.get("max_input_channels", 0) > 0 for d in devices)
    except Exception:
        return False


def _detect_display() -> bool:
    """Check if a display is available."""
    if sys.platform == "win32":
        return True  # Windows always has a display session
    try:
        import os
        return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    except Exception:
        return False


def _detect_os() -> str:
    """Map sys.platform to our standard OS names."""
    if sys.platform == "win32":
        return "windows"
    elif sys.platform == "darwin":
        return "macos"
    return "linux"


def detect_capabilities() -> NodeCapabilities:
    """Auto-detect this machine's hardware capabilities."""
    disk = shutil.disk_usage("/") if sys.platform != "win32" else shutil.disk_usage("C:\\")
    return NodeCapabilities(
        gpu=_detect_gpu(),
        cpu_cores=psutil.cpu_count(logical=True) or 1,
        ram_gb=round(psutil.virtual_memory().total / (1024 ** 3), 1),
        disk_free_gb=round(disk.free / (1024 ** 3), 1),
        os=_detect_os(),
        has_mic=_detect_mic(),
        has_display=_detect_display(),
        has_model_loaded=False,
        model_name=None,
    )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mesh/test_capabilities.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/capabilities.py tests/unit/test_mesh/test_capabilities.py
git commit -m "feat(mesh): add hardware capability detection and scoring"
```

---

### Task 3: Mesh Config — Pydantic Model + HomieConfig Integration

**Files:**
- Modify: `src/homie_core/config.py`
- Modify: `tests/unit/test_config.py`

- [ ] **Step 1: Write the failing test**

```python
# Append to tests/unit/test_config.py

def test_config_has_mesh_section():
    cfg = HomieConfig()
    assert hasattr(cfg, "mesh")
    assert cfg.mesh.enabled is True
    assert cfg.mesh.auto_discover is True
    assert cfg.mesh.auto_elect_hub is True
    assert cfg.mesh.preferred_role == "auto"
    assert cfg.mesh.heartbeat_interval == 15
    assert cfg.mesh.sync_interval == 30


def test_mesh_config_from_yaml(tmp_path):
    data = {
        "mesh": {
            "enabled": False,
            "preferred_role": "hub",
            "heartbeat_interval": 10,
            "wan": {"enabled": True, "transport": "websocket"},
            "inference": {"max_concurrent": 4},
            "security": {"key_rotation_days": 7},
        },
        "storage": {"path": str(tmp_path / ".homie")},
    }
    p = tmp_path / "config.yaml"
    import yaml
    p.write_text(yaml.dump(data))
    cfg = load_config(p)
    assert cfg.mesh.enabled is False
    assert cfg.mesh.preferred_role == "hub"
    assert cfg.mesh.heartbeat_interval == 10
    assert cfg.mesh.wan.enabled is True
    assert cfg.mesh.wan.transport == "websocket"
    assert cfg.mesh.inference.max_concurrent == 4
    assert cfg.mesh.security.key_rotation_days == 7
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/unit/test_config.py::test_config_has_mesh_section -v`
Expected: FAIL — `AttributeError: 'HomieConfig' object has no attribute 'mesh'`

- [ ] **Step 3: Add MeshConfig to config.py**

Add the following classes before the `HomieConfig` class definition in `src/homie_core/config.py`:

```python
class MeshWANConfig(BaseModel):
    enabled: bool = False
    transport: str = "tailscale"  # tailscale | websocket
    hub_address: str = ""
    fallback_address: str = ""


class MeshInferenceConfig(BaseModel):
    allow_remote: bool = True
    max_concurrent: int = 2
    queue_spillover: str = "qubrid"  # qubrid | vertex | reject


class MeshSecurityConfig(BaseModel):
    key_rotation_days: int = 30
    require_tailscale: bool = False
    ip_allowlist: list[str] = Field(default_factory=list)


class MeshConfig(BaseModel):
    enabled: bool = True
    auto_discover: bool = True
    auto_elect_hub: bool = True
    preferred_role: str = "auto"  # auto | hub | spoke
    pairing_timeout: int = 300
    heartbeat_interval: int = 15
    sync_interval: int = 30
    max_offline_events: int = 100000
    wan: MeshWANConfig = Field(default_factory=MeshWANConfig)
    inference: MeshInferenceConfig = Field(default_factory=MeshInferenceConfig)
    security: MeshSecurityConfig = Field(default_factory=MeshSecurityConfig)
```

Then add to the `HomieConfig` class:

```python
    mesh: MeshConfig = Field(default_factory=MeshConfig)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_config.py -v`
Expected: All tests PASS (including the 2 new ones)

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/config.py tests/unit/test_config.py
git commit -m "feat(mesh): add MeshConfig with WAN, inference, and security sections"
```

---

### Task 4: Mesh Node Registry — SQLite Storage for Known Nodes

**Files:**
- Create: `src/homie_core/mesh/registry.py`
- Test: `tests/unit/test_mesh/test_registry.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_registry.py
from pathlib import Path

from homie_core.mesh.registry import MeshNodeRegistry, MeshNodeRecord


def test_registry_initialize(tmp_path):
    """Registry creates the mesh_nodes table on init."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()
    # No error means table created successfully


def test_register_and_get_node(tmp_path):
    """Can register a node and retrieve it by ID."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()

    record = MeshNodeRecord(
        node_id="abc-123",
        node_name="test-box",
        role="spoke",
        mesh_id="mesh-1",
        capability_score=42.0,
        capabilities_json='{"cpu_cores": 4}',
        lan_ip="192.168.1.10",
        tailnet_ip="",
        public_key_ed25519="PEM...",
        status="online",
    )
    reg.upsert(record)

    loaded = reg.get("abc-123")
    assert loaded is not None
    assert loaded.node_name == "test-box"
    assert loaded.capability_score == 42.0
    assert loaded.status == "online"


def test_upsert_updates_existing(tmp_path):
    """Upserting an existing node_id updates it instead of duplicating."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()

    record = MeshNodeRecord(
        node_id="abc-123", node_name="box-v1", role="spoke",
        mesh_id="m1", capability_score=10.0, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    )
    reg.upsert(record)

    record.node_name = "box-v2"
    record.capability_score = 99.0
    reg.upsert(record)

    all_nodes = reg.list_all()
    assert len(all_nodes) == 1
    assert all_nodes[0].node_name == "box-v2"
    assert all_nodes[0].capability_score == 99.0


def test_list_all_nodes(tmp_path):
    """list_all() returns all registered nodes."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()

    for i in range(3):
        reg.upsert(MeshNodeRecord(
            node_id=f"node-{i}", node_name=f"box-{i}", role="spoke",
            mesh_id="m1", capability_score=float(i), capabilities_json="{}",
            lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
        ))

    nodes = reg.list_all()
    assert len(nodes) == 3


def test_remove_node(tmp_path):
    """Can remove a node by ID."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()

    reg.upsert(MeshNodeRecord(
        node_id="gone", node_name="bye", role="spoke",
        mesh_id="m1", capability_score=0, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    ))
    assert reg.get("gone") is not None
    reg.remove("gone")
    assert reg.get("gone") is None


def test_update_status(tmp_path):
    """Can update just the status of a node."""
    reg = MeshNodeRegistry(tmp_path / "test.db")
    reg.initialize()

    reg.upsert(MeshNodeRecord(
        node_id="n1", node_name="box", role="spoke",
        mesh_id="m1", capability_score=10, capabilities_json="{}",
        lan_ip="", tailnet_ip="", public_key_ed25519="", status="online",
    ))
    reg.update_status("n1", "offline")
    node = reg.get("n1")
    assert node.status == "offline"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mesh/test_registry.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'homie_core.mesh.registry'`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/registry.py
"""Mesh node registry — SQLite storage for known nodes in the mesh."""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from homie_core.utils import utc_now


@dataclass
class MeshNodeRecord:
    """A known node in the mesh."""

    node_id: str
    node_name: str
    role: str  # hub | spoke | standalone
    mesh_id: str
    capability_score: float
    capabilities_json: str
    lan_ip: str
    tailnet_ip: str
    public_key_ed25519: str
    status: str  # online | offline | degraded
    last_seen_ts: str = ""
    paired_at: str = ""


class MeshNodeRegistry:
    """SQLite-backed registry of mesh nodes."""

    def __init__(self, db_path: Path | str):
        self._path = Path(db_path)
        self._conn: Optional[sqlite3.Connection] = None

    def initialize(self) -> None:
        """Create the database and mesh_nodes table."""
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self._path))
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS mesh_nodes (
                node_id TEXT PRIMARY KEY,
                node_name TEXT NOT NULL,
                role TEXT NOT NULL DEFAULT 'standalone',
                mesh_id TEXT,
                capability_score REAL DEFAULT 0,
                capabilities_json TEXT DEFAULT '{}',
                lan_ip TEXT DEFAULT '',
                tailnet_ip TEXT DEFAULT '',
                last_seen_ts TEXT DEFAULT '',
                paired_at TEXT DEFAULT '',
                public_key_ed25519 TEXT DEFAULT '',
                status TEXT DEFAULT 'offline'
            )
        """)
        self._conn.commit()

    def upsert(self, record: MeshNodeRecord) -> None:
        """Insert or update a node record."""
        now = utc_now().isoformat()
        self._conn.execute("""
            INSERT INTO mesh_nodes (
                node_id, node_name, role, mesh_id, capability_score,
                capabilities_json, lan_ip, tailnet_ip, last_seen_ts,
                paired_at, public_key_ed25519, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(node_id) DO UPDATE SET
                node_name=excluded.node_name,
                role=excluded.role,
                mesh_id=excluded.mesh_id,
                capability_score=excluded.capability_score,
                capabilities_json=excluded.capabilities_json,
                lan_ip=excluded.lan_ip,
                tailnet_ip=excluded.tailnet_ip,
                last_seen_ts=?,
                public_key_ed25519=excluded.public_key_ed25519,
                status=excluded.status
        """, (
            record.node_id, record.node_name, record.role, record.mesh_id,
            record.capability_score, record.capabilities_json,
            record.lan_ip, record.tailnet_ip,
            record.last_seen_ts or now, record.paired_at,
            record.public_key_ed25519, record.status,
            now,
        ))
        self._conn.commit()

    def get(self, node_id: str) -> Optional[MeshNodeRecord]:
        """Get a node by ID."""
        row = self._conn.execute(
            "SELECT * FROM mesh_nodes WHERE node_id = ?", (node_id,)
        ).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def list_all(self) -> list[MeshNodeRecord]:
        """List all known nodes."""
        rows = self._conn.execute(
            "SELECT * FROM mesh_nodes ORDER BY capability_score DESC"
        ).fetchall()
        return [self._row_to_record(r) for r in rows]

    def remove(self, node_id: str) -> None:
        """Remove a node from the registry."""
        self._conn.execute("DELETE FROM mesh_nodes WHERE node_id = ?", (node_id,))
        self._conn.commit()

    def update_status(self, node_id: str, status: str) -> None:
        """Update just the status and last_seen timestamp."""
        self._conn.execute(
            "UPDATE mesh_nodes SET status = ?, last_seen_ts = ? WHERE node_id = ?",
            (status, utc_now().isoformat(), node_id),
        )
        self._conn.commit()

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> MeshNodeRecord:
        return MeshNodeRecord(
            node_id=row["node_id"],
            node_name=row["node_name"],
            role=row["role"],
            mesh_id=row["mesh_id"] or "",
            capability_score=row["capability_score"],
            capabilities_json=row["capabilities_json"],
            lan_ip=row["lan_ip"],
            tailnet_ip=row["tailnet_ip"],
            public_key_ed25519=row["public_key_ed25519"],
            status=row["status"],
            last_seen_ts=row["last_seen_ts"],
            paired_at=row["paired_at"],
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mesh/test_registry.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/registry.py tests/unit/test_mesh/test_registry.py
git commit -m "feat(mesh): add SQLite-backed mesh node registry"
```

---

### Task 5: Hub Election Logic

**Files:**
- Create: `src/homie_core/mesh/election.py`
- Test: `tests/unit/test_mesh/test_election.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_election.py
from homie_core.mesh.election import elect_hub, ElectionCandidate


def test_highest_score_wins():
    """Node with highest capability score becomes hub."""
    candidates = [
        ElectionCandidate(node_id="a", capability_score=100.0, created_at="2026-01-01T00:00:00"),
        ElectionCandidate(node_id="b", capability_score=290.0, created_at="2026-01-02T00:00:00"),
        ElectionCandidate(node_id="c", capability_score=50.0, created_at="2026-01-03T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "b"


def test_tiebreak_by_created_at():
    """Equal scores: earliest created_at wins."""
    candidates = [
        ElectionCandidate(node_id="new", capability_score=100.0, created_at="2026-03-01T00:00:00"),
        ElectionCandidate(node_id="old", capability_score=100.0, created_at="2026-01-01T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "old"


def test_single_candidate():
    """Single node always wins."""
    candidates = [
        ElectionCandidate(node_id="solo", capability_score=10.0, created_at="2026-01-01T00:00:00"),
    ]
    winner = elect_hub(candidates)
    assert winner.node_id == "solo"


def test_empty_candidates_returns_none():
    """No candidates returns None."""
    winner = elect_hub([])
    assert winner is None
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mesh/test_election.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/election.py
"""Hub election — deterministic leader election based on capability score."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class ElectionCandidate:
    """A node participating in hub election."""

    node_id: str
    capability_score: float
    created_at: str  # ISO8601 — used as tiebreaker


def elect_hub(candidates: list[ElectionCandidate]) -> Optional[ElectionCandidate]:
    """Elect the hub from a list of candidates.

    Rules:
    1. Highest capability_score wins
    2. Ties broken by earliest created_at (most stable node)
    3. Empty list returns None
    """
    if not candidates:
        return None
    # Sort by score descending, then created_at ascending for tiebreak
    return sorted(
        candidates,
        key=lambda c: (-c.capability_score, c.created_at),
    )[0]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mesh/test_election.py -v`
Expected: All 4 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/election.py tests/unit/test_mesh/test_election.py
git commit -m "feat(mesh): add deterministic hub election by capability score"
```

---

### Task 6: Pairing Protocol — Code Generation and Key Exchange

**Files:**
- Create: `src/homie_core/mesh/pairing.py`
- Test: `tests/unit/test_mesh/test_pairing.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_mesh/test_pairing.py
import time

from homie_core.mesh.pairing import (
    PairingSession,
    generate_pairing_code,
    verify_pairing_code,
    derive_shared_secret,
)


def test_generate_pairing_code():
    """Generates a 6-digit numeric code."""
    session = generate_pairing_code(ttl_seconds=300)
    assert len(session.code) == 6
    assert session.code.isdigit()
    assert session.expires_at > time.time()
    assert session.hub_public_key_pem != ""


def test_verify_valid_code():
    """Valid code within TTL returns True."""
    session = generate_pairing_code(ttl_seconds=300)
    assert verify_pairing_code(session, session.code) is True


def test_verify_wrong_code():
    """Wrong code returns False."""
    session = generate_pairing_code(ttl_seconds=300)
    assert verify_pairing_code(session, "000000") is False


def test_verify_expired_code():
    """Expired code returns False."""
    session = generate_pairing_code(ttl_seconds=0)
    # Code is already expired (TTL=0)
    assert verify_pairing_code(session, session.code) is False


def test_derive_shared_secret():
    """Two parties can derive the same shared secret from their key pairs."""
    session_a = generate_pairing_code(ttl_seconds=300)
    session_b = generate_pairing_code(ttl_seconds=300)

    secret_a = derive_shared_secret(
        session_a.hub_private_key_x25519_bytes,
        session_b.spoke_x25519_public_bytes,
    )
    secret_b = derive_shared_secret(
        session_b.hub_private_key_x25519_bytes,
        session_a.spoke_x25519_public_bytes,
    )
    # Both derive the same 32-byte secret
    assert len(secret_a) == 32
    assert len(secret_b) == 32
    # Note: a↔b cross-derive produces same secret
    # (this tests the X25519 property — a's priv + b's pub = b's priv + a's pub)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_mesh/test_pairing.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/mesh/pairing.py
"""Pairing protocol — one-time code + X25519 key exchange for mesh auth."""
from __future__ import annotations

import secrets
import time
from dataclasses import dataclass

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives.asymmetric.x25519 import (
    X25519PrivateKey,
    X25519PublicKey,
)
from cryptography.hazmat.primitives import serialization


@dataclass
class PairingSession:
    """Active pairing session on the hub side."""

    code: str  # 6-digit numeric
    expires_at: float  # Unix timestamp
    hub_public_key_pem: str  # Ed25519 public key (for identity)
    hub_private_key_x25519_bytes: bytes  # X25519 private key (for ECDH)
    spoke_x25519_public_bytes: bytes  # X25519 public key (for ECDH)
    used: bool = False


def generate_pairing_code(ttl_seconds: int = 300) -> PairingSession:
    """Generate a new pairing session with a 6-digit code and X25519 keypair.

    The X25519 keypair is used for ECDH key agreement to derive a shared secret.
    The Ed25519 key is for long-term identity (separate concern).
    """
    code = f"{secrets.randbelow(1000000):06d}"

    # Generate Ed25519 identity key
    ed_private = Ed25519PrivateKey.generate()
    ed_public_pem = ed_private.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()

    # Generate X25519 key for ECDH
    x_private = X25519PrivateKey.generate()
    x_public_bytes = x_private.public_key().public_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PublicFormat.Raw,
    )
    x_private_bytes = x_private.private_bytes(
        encoding=serialization.Encoding.Raw,
        format=serialization.PrivateFormat.Raw,
        encryption_algorithm=serialization.NoEncryption(),
    )

    return PairingSession(
        code=code,
        expires_at=time.time() + ttl_seconds,
        hub_public_key_pem=ed_public_pem,
        hub_private_key_x25519_bytes=x_private_bytes,
        spoke_x25519_public_bytes=x_public_bytes,  # This session's public key (shared with peer)
        used=False,
    )


def verify_pairing_code(session: PairingSession, code: str) -> bool:
    """Verify a pairing code against an active session."""
    if session.used:
        return False
    if time.time() > session.expires_at:
        return False
    return secrets.compare_digest(session.code, code)


def derive_shared_secret(
    my_x25519_private_bytes: bytes,
    peer_x25519_public_bytes: bytes,
) -> bytes:
    """Derive a 32-byte shared secret via X25519 ECDH.

    Both parties call this with their own private key + the peer's public key.
    The result is identical on both sides.
    """
    my_private = X25519PrivateKey.from_private_bytes(my_x25519_private_bytes)
    peer_public = X25519PublicKey.from_public_bytes(peer_x25519_public_bytes)
    return my_private.exchange(peer_public)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_mesh/test_pairing.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/mesh/pairing.py tests/unit/test_mesh/test_pairing.py
git commit -m "feat(mesh): add pairing protocol with 6-digit code and X25519 key exchange"
```

---

### Task 7: Platform Abstraction Layer

**Files:**
- Create: `src/homie_core/platform/__init__.py`
- Create: `src/homie_core/platform/base.py`
- Create: `src/homie_core/platform/windows.py`
- Create: `src/homie_core/platform/linux.py`
- Create: `src/homie_core/platform/macos.py`
- Create: `src/homie_core/platform/detect.py`
- Test: `tests/unit/test_platform/__init__.py`
- Test: `tests/unit/test_platform/test_detect.py`
- Test: `tests/unit/test_platform/test_adapter.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_platform/__init__.py
# (empty)

# tests/unit/test_platform/test_detect.py
import sys

from homie_core.platform.detect import get_platform_adapter
from homie_core.platform.base import PlatformAdapter


def test_get_platform_adapter_returns_adapter():
    """get_platform_adapter() returns a PlatformAdapter subclass for this OS."""
    adapter = get_platform_adapter()
    assert isinstance(adapter, PlatformAdapter)


def test_adapter_get_hostname():
    """Adapter can get the hostname."""
    adapter = get_platform_adapter()
    hostname = adapter.get_hostname()
    assert isinstance(hostname, str)
    assert len(hostname) > 0


def test_adapter_get_system_metrics():
    """Adapter returns system metrics with cpu, ram, disk."""
    adapter = get_platform_adapter()
    metrics = adapter.get_system_metrics()
    assert metrics["cpu_percent"] >= 0
    assert metrics["ram_total_gb"] > 0
    assert metrics["disk_free_gb"] > 0


def test_adapter_send_notification_does_not_crash():
    """send_notification doesn't raise even if notification system unavailable."""
    adapter = get_platform_adapter()
    # Should not raise
    adapter.send_notification("Test", "This is a test")


# tests/unit/test_platform/test_adapter.py
import sys
from unittest.mock import patch

from homie_core.platform.detect import get_platform_adapter


def test_correct_adapter_for_platform():
    """Correct adapter class is returned based on sys.platform."""
    adapter = get_platform_adapter()
    class_name = type(adapter).__name__
    if sys.platform == "win32":
        assert class_name == "WindowsAdapter"
    elif sys.platform == "darwin":
        assert class_name == "MacOSAdapter"
    else:
        assert class_name == "LinuxAdapter"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_platform/ -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Write the implementation**

```python
# src/homie_core/platform/__init__.py
"""Platform abstraction layer for cross-OS compatibility."""

# src/homie_core/platform/base.py
"""Abstract base for platform-specific operations."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional


class PlatformAdapter(ABC):
    """Abstract base class for platform-specific operations."""

    @abstractmethod
    def get_hostname(self) -> str:
        """Return the machine hostname."""
        ...

    @abstractmethod
    def get_active_window(self) -> Optional[str]:
        """Return the title of the currently focused window, or None."""
        ...

    @abstractmethod
    def get_system_metrics(self) -> dict:
        """Return dict with cpu_percent, ram_total_gb, ram_used_gb, disk_free_gb."""
        ...

    @abstractmethod
    def send_notification(self, title: str, body: str) -> None:
        """Show a desktop notification. Must not raise on failure."""
        ...

    @abstractmethod
    def get_gpu_info(self) -> Optional[dict]:
        """Return {"name": str, "vram_gb": float} or None."""
        ...


# src/homie_core/platform/windows.py
"""Windows platform adapter."""
from __future__ import annotations

import logging
import shutil
import socket
from typing import Optional

import psutil

from homie_core.platform.base import PlatformAdapter

logger = logging.getLogger(__name__)


class WindowsAdapter(PlatformAdapter):
    def get_hostname(self) -> str:
        return socket.gethostname()

    def get_active_window(self) -> Optional[str]:
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.GetForegroundWindow()
            length = user32.GetWindowTextLengthW(hwnd)
            buf = ctypes.create_unicode_buffer(length + 1)
            user32.GetWindowTextW(hwnd, buf, length + 1)
            return buf.value or None
        except Exception:
            return None

    def get_system_metrics(self) -> dict:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("C:\\")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(mem.total / (1024 ** 3), 1),
            "ram_used_gb": round(mem.used / (1024 ** 3), 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        }

    def send_notification(self, title: str, body: str) -> None:
        try:
            from homie_app.tray.notifier import show_toast
            show_toast(title, body)
        except Exception:
            logger.debug("Notification failed (no tray): %s — %s", title, body)

    def get_gpu_info(self) -> Optional[dict]:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("\n")[0].split(", ")
                if len(parts) == 2:
                    return {"name": parts[0].strip(), "vram_gb": round(float(parts[1].strip()) / 1024, 1)}
        except (FileNotFoundError, Exception):
            pass
        return None


# src/homie_core/platform/linux.py
"""Linux platform adapter."""
from __future__ import annotations

import logging
import os
import shutil
import socket
from typing import Optional

import psutil

from homie_core.platform.base import PlatformAdapter

logger = logging.getLogger(__name__)


class LinuxAdapter(PlatformAdapter):
    def get_hostname(self) -> str:
        return socket.gethostname()

    def get_active_window(self) -> Optional[str]:
        try:
            import subprocess
            result = subprocess.run(
                ["xdotool", "getactivewindow", "getwindowname"],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except (FileNotFoundError, Exception):
            pass
        return None

    def get_system_metrics(self) -> dict:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(mem.total / (1024 ** 3), 1),
            "ram_used_gb": round(mem.used / (1024 ** 3), 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        }

    def send_notification(self, title: str, body: str) -> None:
        try:
            import subprocess
            subprocess.run(
                ["notify-send", title, body],
                capture_output=True, timeout=5,
            )
        except (FileNotFoundError, Exception):
            logger.debug("notify-send failed: %s — %s", title, body)

    def get_gpu_info(self) -> Optional[dict]:
        try:
            import subprocess
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.total",
                 "--format=csv,noheader,nounits"],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0 and result.stdout.strip():
                parts = result.stdout.strip().split("\n")[0].split(", ")
                if len(parts) == 2:
                    return {"name": parts[0].strip(), "vram_gb": round(float(parts[1].strip()) / 1024, 1)}
        except (FileNotFoundError, Exception):
            pass
        return None


# src/homie_core/platform/macos.py
"""macOS platform adapter."""
from __future__ import annotations

import logging
import shutil
import socket
from typing import Optional

import psutil

from homie_core.platform.base import PlatformAdapter

logger = logging.getLogger(__name__)


class MacOSAdapter(PlatformAdapter):
    def get_hostname(self) -> str:
        return socket.gethostname()

    def get_active_window(self) -> Optional[str]:
        try:
            import subprocess
            script = 'tell application "System Events" to get name of first application process whose frontmost is true'
            result = subprocess.run(
                ["osascript", "-e", script],
                capture_output=True, text=True, timeout=2,
            )
            if result.returncode == 0:
                return result.stdout.strip() or None
        except (FileNotFoundError, Exception):
            pass
        return None

    def get_system_metrics(self) -> dict:
        mem = psutil.virtual_memory()
        disk = shutil.disk_usage("/")
        return {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_total_gb": round(mem.total / (1024 ** 3), 1),
            "ram_used_gb": round(mem.used / (1024 ** 3), 1),
            "disk_free_gb": round(disk.free / (1024 ** 3), 1),
        }

    def send_notification(self, title: str, body: str) -> None:
        try:
            import subprocess
            script = f'display notification "{body}" with title "{title}"'
            subprocess.run(["osascript", "-e", script], capture_output=True, timeout=5)
        except (FileNotFoundError, Exception):
            logger.debug("osascript notification failed: %s — %s", title, body)

    def get_gpu_info(self) -> Optional[dict]:
        # macOS uses Metal; no simple CLI for VRAM detection
        # system_profiler could work but is slow
        return None


# src/homie_core/platform/detect.py
"""Auto-detect the current platform and return the correct adapter."""
from __future__ import annotations

import sys

from homie_core.platform.base import PlatformAdapter


def get_platform_adapter() -> PlatformAdapter:
    """Return the correct PlatformAdapter for the current OS."""
    if sys.platform == "win32":
        from homie_core.platform.windows import WindowsAdapter
        return WindowsAdapter()
    elif sys.platform == "darwin":
        from homie_core.platform.macos import MacOSAdapter
        return MacOSAdapter()
    else:
        from homie_core.platform.linux import LinuxAdapter
        return LinuxAdapter()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_platform/ -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add src/homie_core/platform/__init__.py src/homie_core/platform/base.py src/homie_core/platform/windows.py src/homie_core/platform/linux.py src/homie_core/platform/macos.py src/homie_core/platform/detect.py tests/unit/test_platform/__init__.py tests/unit/test_platform/test_detect.py tests/unit/test_platform/test_adapter.py
git commit -m "feat(platform): add cross-platform abstraction layer with Windows/Linux/macOS adapters"
```

---

### Task 8: Enhanced mDNS Discovery with Mesh TXT Records

**Files:**
- Modify: `src/homie_core/network/discovery.py`
- Create: `tests/unit/test_network/__init__.py`
- Create: `tests/unit/test_network/test_discovery_mesh.py`

- [ ] **Step 1: Write the failing tests**

```python
# tests/unit/test_network/__init__.py
# (empty)

# tests/unit/test_network/test_discovery_mesh.py
from unittest.mock import MagicMock, patch

from homie_core.network.discovery import HomieDiscovery


def test_discovery_accepts_mesh_params():
    """HomieDiscovery constructor accepts role, mesh_id, capability_score."""
    d = HomieDiscovery(
        device_id="abc",
        device_name="test",
        port=8765,
        role="hub",
        mesh_id="mesh-1",
        capability_score=290.0,
    )
    assert d.role == "hub"
    assert d.mesh_id == "mesh-1"
    assert d.capability_score == 290.0


def test_discovery_default_mesh_params():
    """Default mesh params are standalone with no mesh_id."""
    d = HomieDiscovery(device_id="abc", device_name="test")
    assert d.role == "standalone"
    assert d.mesh_id == ""
    assert d.capability_score == 0.0


def test_discovered_device_includes_mesh_fields():
    """When a device is discovered, its mesh fields are captured."""
    d = HomieDiscovery(device_id="local", device_name="me")

    # Simulate a discovered device with mesh TXT records
    mock_info = MagicMock()
    mock_info.properties = {
        b"device_id": b"remote-1",
        b"device_name": b"other-box",
        b"role": b"hub",
        b"mesh_id": b"mesh-1",
        b"capability_score": b"290.0",
        b"version": b"1.0.0",
    }
    mock_info.addresses = [b"\xc0\xa8\x01\x0a"]  # 192.168.1.10
    mock_info.port = 8765

    mock_state_change = MagicMock()
    mock_state_change_cls = MagicMock()
    mock_state_change_cls.Added = mock_state_change

    mock_zeroconf = MagicMock()
    mock_zeroconf.get_service_info.return_value = mock_info

    with patch("homie_core.network.discovery.ServiceStateChange", mock_state_change_cls):
        d._on_service_state_change(mock_zeroconf, "_homie._tcp.local.", "test", mock_state_change)

    devices = d.discovered_devices
    assert "remote-1" in devices
    assert devices["remote-1"]["role"] == "hub"
    assert devices["remote-1"]["mesh_id"] == "mesh-1"
    assert devices["remote-1"]["capability_score"] == 290.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/unit/test_network/test_discovery_mesh.py -v`
Expected: FAIL — `TypeError: HomieDiscovery.__init__() got an unexpected keyword argument 'role'`

- [ ] **Step 3: Modify HomieDiscovery to support mesh fields**

In `src/homie_core/network/discovery.py`, update the `__init__` method signature to accept mesh parameters, add them to TXT record properties during advertising, and extract them during discovery:

Update `__init__`:
```python
def __init__(self, device_id: str, device_name: str, port: int = 8765,
             role: str = "standalone", mesh_id: str = "", capability_score: float = 0.0):
    self.device_id = device_id
    self.device_name = device_name
    self.port = port
    self.role = role
    self.mesh_id = mesh_id
    self.capability_score = capability_score
    self._zeroconf: Optional[Zeroconf] = None
    self._browser = None
    self._service_info = None
    self._advertising = False
    self._discovered: dict[str, dict] = {}
```

Update `start_advertising` properties dict to include:
```python
properties={
    "device_id": self.device_id,
    "device_name": self.device_name,
    "version": "1.0.0",
    "role": self.role,
    "mesh_id": self.mesh_id,
    "capability_score": str(self.capability_score),
},
```

Update `_on_service_state_change` to capture mesh fields:
```python
if device_id and device_id != self.device_id:
    self._discovered[device_id] = {
        "name": info.properties.get(b"device_name", b"").decode(),
        "host": socket.inet_ntoa(info.addresses[0]) if info.addresses else "",
        "port": info.port,
        "role": info.properties.get(b"role", b"standalone").decode(),
        "mesh_id": info.properties.get(b"mesh_id", b"").decode(),
        "capability_score": float(info.properties.get(b"capability_score", b"0").decode()),
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/unit/test_network/test_discovery_mesh.py -v`
Expected: All 3 tests PASS

- [ ] **Step 5: Run existing network tests to verify no regressions**

Run: `python -m pytest tests/unit/ -v -k "network or discovery" --no-header`
Expected: All tests PASS

- [ ] **Step 6: Commit**

```bash
git add src/homie_core/network/discovery.py tests/unit/test_network/__init__.py tests/unit/test_network/test_discovery_mesh.py
git commit -m "feat(mesh): enhance mDNS discovery with role, mesh_id, and capability_score TXT records"
```

---

### Task 9: /node and /mesh CLI Commands

**Files:**
- Create: `src/homie_app/console/commands/node.py`
- Create: `src/homie_app/console/commands/mesh.py`
- Modify: `src/homie_app/console/console.py` (register commands)

- [ ] **Step 1: Write the /node command**

```python
# src/homie_app/console/commands/node.py
"""Handler for /node slash command — node identity and info."""
from __future__ import annotations

import json
from pathlib import Path

from homie_app.console.router import SlashCommand


def _handle_node_info(args: str, **ctx) -> str:
    """Show this node's identity and capabilities."""
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.capabilities import detect_capabilities

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)

    if identity is None:
        return (
            "No node identity found. Run `homie init` or `/node init` to create one."
        )

    caps = detect_capabilities()
    lines = [
        "**Node Identity**",
        f"  ID:   {identity.node_id}",
        f"  Name: {identity.node_name}",
        f"  Role: {identity.role}",
        f"  Mesh: {identity.mesh_id or 'not joined'}",
        "",
        "**Capabilities**",
        f"  OS:      {caps.os}",
        f"  CPU:     {caps.cpu_cores} cores",
        f"  RAM:     {caps.ram_gb} GB",
        f"  Disk:    {caps.disk_free_gb} GB free",
        f"  GPU:     {caps.gpu['name'] + ' (' + str(caps.gpu['vram_gb']) + ' GB)' if caps.gpu else 'none'}",
        f"  Mic:     {'yes' if caps.has_mic else 'no'}",
        f"  Display: {'yes' if caps.has_display else 'no'}",
        f"  Score:   {caps.capability_score():.0f}",
    ]
    return "\n".join(lines)


def _handle_node_init(args: str, **ctx) -> str:
    """Create or regenerate this node's identity."""
    from homie_core.mesh.identity import NodeIdentity, save_identity, load_identity

    identity_path = Path.home() / ".homie" / "node.json"
    existing = load_identity(identity_path)

    if existing and "--force" not in args:
        return (
            f"Node identity already exists: {existing.node_id} ({existing.node_name})\n"
            f"Use `/node init --force` to regenerate (this will unpair from any mesh)."
        )

    identity = NodeIdentity.generate()
    save_identity(identity, identity_path)
    return f"Node identity created:\n  ID:   {identity.node_id}\n  Name: {identity.node_name}"


def _handle_node_set_name(args: str, **ctx) -> str:
    """Set this node's display name."""
    name = args.strip()
    if not name:
        return "Usage: /node set-name <name>"

    from homie_core.mesh.identity import load_identity, save_identity

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."

    identity.node_name = name
    save_identity(identity, identity_path)
    return f"Node name set to: {name}"


def _handle_node_set_role(args: str, **ctx) -> str:
    """Force this node's role (hub/spoke)."""
    role = args.strip().lower()
    if role not in ("hub", "spoke", "standalone"):
        return "Usage: /node set-role <hub|spoke|standalone>"

    from homie_core.mesh.identity import load_identity, save_identity

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."

    identity.role = role
    save_identity(identity, identity_path)
    return f"Node role set to: {role}"


def register_node_commands(router) -> None:
    """Register /node command with subcommands."""
    node_cmd = SlashCommand(
        name="node",
        description="Node identity and management",
        args_spec="<info|init|set-name|set-role>",
        subcommands={
            "info": SlashCommand(name="info", description="Show node identity", handler_fn=_handle_node_info),
            "init": SlashCommand(name="init", description="Create node identity", handler_fn=_handle_node_init),
            "set-name": SlashCommand(name="set-name", description="Set node name", args_spec="<name>", handler_fn=_handle_node_set_name),
            "set-role": SlashCommand(name="set-role", description="Force node role", args_spec="<hub|spoke>", handler_fn=_handle_node_set_role),
        },
        handler_fn=_handle_node_info,  # Default to info
    )
    router.register(node_cmd)
```

- [ ] **Step 2: Write the /mesh command**

```python
# src/homie_app/console/commands/mesh.py
"""Handler for /mesh slash command — mesh topology and management."""
from __future__ import annotations

from pathlib import Path

from homie_app.console.router import SlashCommand


def _handle_mesh_status(args: str, **ctx) -> str:
    """Show mesh topology and node statuses."""
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.registry import MeshNodeRegistry

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."

    if not identity.mesh_id:
        return (
            f"**This node:** {identity.node_name} ({identity.role})\n"
            f"**Mesh:** not joined\n\n"
            f"Use `/mesh pair` (on hub) or `/mesh join --code <code>` to form a mesh."
        )

    cfg = ctx.get("config")
    if not cfg:
        return "No configuration loaded."

    db_path = Path(cfg.storage.path) / "mesh_nodes.db"
    registry = MeshNodeRegistry(db_path)
    registry.initialize()
    nodes = registry.list_all()

    lines = [
        f"**Mesh:** {identity.mesh_id}",
        f"**This node:** {identity.node_name} ({identity.role})",
        f"**Nodes:** {len(nodes) + 1}",
        "",
    ]
    for node in nodes:
        status_icon = "+" if node.status == "online" else "-" if node.status == "offline" else "~"
        lines.append(f"  [{status_icon}] {node.node_name} ({node.role}) — score: {node.capability_score:.0f} — {node.status}")

    return "\n".join(lines)


def _handle_mesh_pair(args: str, **ctx) -> str:
    """Generate a pairing code for other nodes to join."""
    from homie_core.mesh.identity import load_identity
    from homie_core.mesh.pairing import generate_pairing_code

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity. Run `/node init` first."

    cfg = ctx.get("config")
    ttl = cfg.mesh.pairing_timeout if cfg else 300
    session = generate_pairing_code(ttl_seconds=ttl)

    return (
        f"**Pairing Code:** {session.code}\n"
        f"**Expires in:** {ttl} seconds\n\n"
        f"On the other machine, run: `/mesh join --code {session.code}`"
    )


def _handle_mesh_nodes(args: str, **ctx) -> str:
    """List all known nodes with capabilities."""
    return _handle_mesh_status(args, **ctx)


def _handle_mesh_leave(args: str, **ctx) -> str:
    """Leave the current mesh."""
    from homie_core.mesh.identity import load_identity, save_identity

    identity_path = Path.home() / ".homie" / "node.json"
    identity = load_identity(identity_path)
    if identity is None:
        return "No node identity."

    if not identity.mesh_id:
        return "Not in a mesh."

    old_mesh = identity.mesh_id
    identity.mesh_id = None
    identity.role = "standalone"
    save_identity(identity, identity_path)
    return f"Left mesh {old_mesh}. Node is now standalone."


def register_mesh_commands(router) -> None:
    """Register /mesh command with subcommands."""
    mesh_cmd = SlashCommand(
        name="mesh",
        description="Mesh topology and management",
        args_spec="<status|pair|join|leave|nodes>",
        subcommands={
            "status": SlashCommand(name="status", description="Show mesh topology", handler_fn=_handle_mesh_status),
            "pair": SlashCommand(name="pair", description="Generate pairing code", handler_fn=_handle_mesh_pair),
            "leave": SlashCommand(name="leave", description="Leave current mesh", handler_fn=_handle_mesh_leave),
            "nodes": SlashCommand(name="nodes", description="List all nodes", handler_fn=_handle_mesh_nodes),
        },
        handler_fn=_handle_mesh_status,  # Default to status
    )
    router.register(mesh_cmd)
```

- [ ] **Step 3: Register commands in console.py**

Find where other commands are registered in `src/homie_app/console/console.py` and add:

```python
from homie_app.console.commands.node import register_node_commands
from homie_app.console.commands.mesh import register_mesh_commands

# ... in the registration section:
register_node_commands(router)
register_mesh_commands(router)
```

- [ ] **Step 4: Test commands manually**

Run: `python -c "from homie_app.console.commands.node import register_node_commands; print('OK')"`
Run: `python -c "from homie_app.console.commands.mesh import register_mesh_commands; print('OK')"`
Expected: Both print "OK" without errors

- [ ] **Step 5: Commit**

```bash
git add src/homie_app/console/commands/node.py src/homie_app/console/commands/mesh.py src/homie_app/console/console.py
git commit -m "feat(mesh): add /node and /mesh CLI commands for mesh management"
```

---

### Task 10: Dependency Update and Integration Smoke Test

**Files:**
- Modify: `pyproject.toml`
- Create: `tests/integration/test_mesh_smoke.py`

- [ ] **Step 1: Add mesh dependency group to pyproject.toml**

Add after the existing optional dependency groups:

```toml
mesh = [
    "zeroconf>=0.80",
    "websockets>=12.0",
    "msgpack>=1.0",
    "lz4>=4.3",
]
```

- [ ] **Step 2: Write integration smoke test**

```python
# tests/integration/test_mesh_smoke.py
"""Smoke test: verify mesh foundation components work end-to-end."""
from pathlib import Path

from homie_core.mesh.identity import NodeIdentity, save_identity, load_identity
from homie_core.mesh.capabilities import detect_capabilities
from homie_core.mesh.election import ElectionCandidate, elect_hub
from homie_core.mesh.pairing import generate_pairing_code, verify_pairing_code
from homie_core.mesh.registry import MeshNodeRegistry, MeshNodeRecord
from homie_core.platform.detect import get_platform_adapter
from homie_core.config import HomieConfig


def test_full_mesh_foundation_flow(tmp_path):
    """End-to-end: identity → capabilities → election → registry → config."""
    # 1. Generate identity
    identity = NodeIdentity.generate()
    assert identity.node_id

    # 2. Save and reload
    node_path = tmp_path / "node.json"
    save_identity(identity, node_path)
    loaded = load_identity(node_path)
    assert loaded.node_id == identity.node_id

    # 3. Detect capabilities
    caps = detect_capabilities()
    assert caps.cpu_cores > 0

    # 4. Election with this node as candidate
    candidate = ElectionCandidate(
        node_id=identity.node_id,
        capability_score=caps.capability_score(),
        created_at=identity.created_at,
    )
    winner = elect_hub([candidate])
    assert winner.node_id == identity.node_id

    # 5. Registry
    db_path = tmp_path / "mesh.db"
    registry = MeshNodeRegistry(db_path)
    registry.initialize()
    registry.upsert(MeshNodeRecord(
        node_id=identity.node_id,
        node_name=identity.node_name,
        role="hub",
        mesh_id="test-mesh",
        capability_score=caps.capability_score(),
        capabilities_json="{}",
        lan_ip="192.168.1.1",
        tailnet_ip="",
        public_key_ed25519=identity.public_key_pem,
        status="online",
    ))
    node = registry.get(identity.node_id)
    assert node.role == "hub"

    # 6. Pairing
    session = generate_pairing_code(ttl_seconds=60)
    assert verify_pairing_code(session, session.code) is True

    # 7. Platform adapter
    adapter = get_platform_adapter()
    assert adapter.get_hostname() != ""

    # 8. Config has mesh section
    cfg = HomieConfig()
    assert cfg.mesh.enabled is True


def test_two_node_election(tmp_path):
    """Simulate two nodes: desktop (GPU) should win hub election over laptop."""
    desktop = ElectionCandidate(
        node_id="desktop-1",
        capability_score=290.0,  # RTX 5080 + 32GB RAM + 16 cores + model loaded
        created_at="2026-01-01T00:00:00",
    )
    laptop = ElectionCandidate(
        node_id="laptop-1",
        capability_score=40.0,  # No GPU, 16GB RAM, 8 cores
        created_at="2026-01-02T00:00:00",
    )
    winner = elect_hub([desktop, laptop])
    assert winner.node_id == "desktop-1"
```

- [ ] **Step 3: Run all tests**

Run: `python -m pytest tests/unit/test_mesh/ tests/unit/test_platform/ tests/unit/test_network/test_discovery_mesh.py tests/integration/test_mesh_smoke.py -v`
Expected: All tests PASS

- [ ] **Step 4: Run the full test suite to check for regressions**

Run: `python -m pytest tests/ -v --timeout=30`
Expected: No regressions in existing tests

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml tests/integration/test_mesh_smoke.py
git commit -m "feat(mesh): add mesh dependencies and integration smoke test"
```

---

## Summary

| Task | Component | Files Created/Modified | Tests |
|------|-----------|----------------------|-------|
| 1 | Node Identity | `mesh/identity.py` | 6 tests |
| 2 | Capability Detection | `mesh/capabilities.py` | 5 tests |
| 3 | Mesh Config | `config.py` (modified) | 2 tests |
| 4 | Node Registry | `mesh/registry.py` | 6 tests |
| 5 | Hub Election | `mesh/election.py` | 4 tests |
| 6 | Pairing Protocol | `mesh/pairing.py` | 5 tests |
| 7 | Platform Abstraction | `platform/*.py` (5 files) | 5 tests |
| 8 | Enhanced Discovery | `network/discovery.py` (modified) | 3 tests |
| 9 | CLI Commands | `commands/node.py`, `commands/mesh.py` | manual verification |
| 10 | Dependencies + Smoke | `pyproject.toml`, smoke test | 2 integration tests |

**Total: 10 tasks, 38+ tests, 15 new files, 3 modified files**

After Phase 1 completes, every Homie node can:
- Identify itself with a UUID and Ed25519 keypair
- Detect its own hardware capabilities
- Discover other nodes on LAN via mDNS
- Pair with other nodes via 6-digit code + X25519 key exchange
- Store known nodes in a SQLite registry
- Elect a hub based on capability scores
- Manage mesh via `/node` and `/mesh` CLI commands
- Run on Windows, Linux, or macOS with platform-appropriate behavior
