"""Node identity — unique, persistent identity for each Homie instance."""
from __future__ import annotations
import json, socket, uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from cryptography.hazmat.primitives import serialization
from homie_core.utils import utc_now

@dataclass
class NodeIdentity:
    node_id: str
    node_name: str
    role: str  # "hub" | "spoke" | "standalone"
    mesh_id: Optional[str]
    created_at: str
    public_key_pem: str
    private_key_pem: str  # Never persisted to node.json

    @classmethod
    def generate(cls) -> NodeIdentity:
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
        private_key_pem="",
    )
