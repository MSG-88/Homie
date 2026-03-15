# Phase 1: Core Architecture Refactor — Design Spec

**Date:** 2026-03-15
**Status:** Approved
**Branch:** `feat/homie-ai-v2`
**Scope:** Middleware architecture, Backend protocol, Dynamic model switching
**Inspiration:** [langchain-ai/deepagents](https://github.com/langchain-ai/deepagents) — patterns adopted natively, zero LangChain dependency

---

## 1. Overview

This spec introduces three foundational abstractions into Homie's cognitive architecture, inspired by DeepAgents but implemented natively in pure Python with zero new dependencies:

1. **Middleware Architecture** — stackable layers that add tools, modify prompts, and intercept model/tool calls
2. **Backend Protocol** — unified interface for storage and execution across local, remote, and encrypted environments
3. **Dynamic Model Switching** — complexity-aware model selection that feeds into the existing inference router

These form the foundation for Phases 2–5 (context intelligence, multi-agent, safety, extensibility), where each future feature becomes "just add a new middleware."

---

## 2. Motivation

### Current Limitations

| Area | Problem |
|------|---------|
| Extension | Tools, prompts, and behavior are hardwired in `cognitive_arch.py` and `builtin_tools.py` — no clean way to add/modify without editing core code |
| File/Shell I/O | `builtin_tools.py` uses raw `pathlib` and `subprocess` — no path containment, no abstraction for remote/sandboxed/encrypted environments |
| Model Selection | Fixed at startup via `config.llm.model_path` — can't use small models for trivial queries or switch mid-conversation |
| Testability | Can't mock file/shell operations without monkey-patching — no interface boundary |

### What DeepAgents Does Well

DeepAgents composes all behavior through a middleware stack (`AgentMiddleware`) and routes all I/O through a `BackendProtocol`. This means:
- Adding a feature = adding a middleware class
- Changing the execution environment = swapping a backend
- Everything is testable via `StateBackend` (in-memory)

We adopt these patterns while preserving Homie's unique 6-stage cognitive pipeline and adding Homie-specific backends (LAN, encrypted vault).

---

## 3. Middleware Architecture

### 3.1 Design: Hybrid Two-Layer System

**Outer Middleware Stack** — wraps `BrainOrchestrator.process()`. Each middleware is a class with well-defined intercept points, executed in onion order (lowest `order` first for `before_*`, reverse for `after_*`).

**Inner Pipeline Hooks** — lightweight callbacks emitted at boundaries between the 6 cognitive stages. Not full middleware — just signals that can observe and modify stage outputs.

### 3.2 Outer Middleware Base Class

```python
# src/homie_core/middleware/base.py

from typing import Protocol, Any

class HomieMiddleware:
    """Base class for outer middleware. Override only what you need."""

    name: str = "unnamed"
    order: int = 100  # lower = runs first for before_*, last for after_*

    def modify_tools(self, tools: list[dict]) -> list[dict]:
        """Add, remove, or wrap tools before they're presented to the model."""
        return tools

    def modify_prompt(self, system_prompt: str) -> str:
        """Inject context into the system prompt."""
        return system_prompt

    def before_turn(self, message: str, state: dict) -> str | None:
        """Pre-process user message. Return None to block the turn."""
        return message

    def after_turn(self, response: str, state: dict) -> str:
        """Post-process assistant response."""
        return response

    def wrap_tool_call(self, name: str, args: dict) -> dict | None:
        """Intercept before tool executes. Return None to block execution."""
        return args

    def wrap_tool_result(self, name: str, result: str) -> str:
        """Intercept after tool returns. Transform the result."""
        return result
```

### 3.3 Middleware Stack

```python
# src/homie_core/middleware/stack.py

class MiddlewareStack:
    """Ordered execution engine for middleware layers."""

    def __init__(self, middleware: list[HomieMiddleware] | None = None):
        self._middleware = sorted(middleware or [], key=lambda m: m.order)

    def add(self, mw: HomieMiddleware) -> None:
        """Add middleware and re-sort."""
        self._middleware.append(mw)
        self._middleware.sort(key=lambda m: m.order)

    def apply_tools(self, tools: list[dict]) -> list[dict]:
        """Run modify_tools through the stack in order."""
        for mw in self._middleware:
            tools = mw.modify_tools(tools)
        return tools

    def apply_prompt(self, prompt: str) -> str:
        """Run modify_prompt through the stack in order."""
        for mw in self._middleware:
            prompt = mw.modify_prompt(prompt)
        return prompt

    def run_before_turn(self, message: str, state: dict) -> str | None:
        """Run before_turn in order. Any middleware returning None blocks."""
        for mw in self._middleware:
            result = mw.before_turn(message, state)
            if result is None:
                return None
            message = result
        return message

    def run_after_turn(self, response: str, state: dict) -> str:
        """Run after_turn in reverse order (onion unwinding)."""
        for mw in reversed(self._middleware):
            response = mw.after_turn(response, state)
        return response

    def run_wrap_tool_call(self, name: str, args: dict) -> dict | None:
        """Run wrap_tool_call in order. Any returning None blocks execution."""
        for mw in self._middleware:
            result = mw.wrap_tool_call(name, args)
            if result is None:
                return None
            args = result
        return args

    def run_wrap_tool_result(self, name: str, result: str) -> str:
        """Run wrap_tool_result in reverse order."""
        for mw in reversed(self._middleware):
            result = mw.wrap_tool_result(name, result)
        return result
```

### 3.4 Inner Pipeline Hooks

```python
# src/homie_core/middleware/hooks.py

from typing import Callable, Any
from enum import Enum

class PipelineStage(str, Enum):
    PERCEIVED = "on_perceived"      # after PERCEIVE, receives SituationalAwareness
    CLASSIFIED = "on_classified"    # after CLASSIFY, receives complexity level
    RETRIEVED = "on_retrieved"      # after RETRIEVE, receives combined RetrievalBundle
    PROMPT_BUILT = "on_prompt_built"  # after REASON, receives assembled prompt
    REFLECTED = "on_reflected"      # after REFLECT, receives confidence score
    # Note: ADAPT is intentionally excluded — it is the final generation stage
    # where inference runs. By that point, all hooks have had their chance to
    # modify context. Intercepting generation itself is handled by the outer
    # middleware's after_turn() hook.

HookCallback = Callable[[PipelineStage, Any], Any]

class HookRegistry:
    """Registry for inner pipeline hooks. Callbacks can observe and modify stage outputs."""

    def __init__(self):
        self._hooks: dict[PipelineStage, list[HookCallback]] = {
            stage: [] for stage in PipelineStage
        }

    def register(self, stage: PipelineStage, callback: HookCallback) -> None:
        self._hooks[stage].append(callback)

    def emit(self, stage: PipelineStage, data: Any) -> Any:
        """Emit a stage signal. Each hook can modify and return data."""
        for callback in self._hooks[stage]:
            result = callback(stage, data)
            if result is not None:
                data = result
        return data
```

### 3.5 Integration with CognitiveArchitecture

`cognitive_arch.py` is modified minimally — each stage boundary gets a `self._hooks.emit()` call.

The actual method signatures in the existing code differ from a textbook pipeline — `_perceive()` reads from `self._wm` internally (no args), and retrieval is split across three methods (`_retrieve_relevant_facts`, `_retrieve_relevant_episodes`, `_retrieve_documents`). The hooks adapt to the real signatures:

```python
# In CognitiveArchitecture._prepare_prompt() — the actual pipeline method:

# PERCEIVE: _perceive() reads from self._wm, returns SituationalAwareness
awareness = self._perceive()  # no args — reads self._wm internally
awareness = self._hooks.emit(PipelineStage.PERCEIVED, awareness)

# CLASSIFY: uses query text + awareness
complexity = self._classify_complexity(query, awareness)
complexity = self._hooks.emit(PipelineStage.CLASSIFIED, complexity)

# RETRIEVE: three separate retrievals combined into a bundle
@dataclass
class RetrievalBundle:
    facts: list       # from _retrieve_relevant_facts (filtered by confidence >= 0.4)
    episodes: list    # from _retrieve_relevant_episodes (vector search)
    documents: list   # from _retrieve_documents (RAG pipeline)

facts = self._retrieve_relevant_facts(query, complexity)
episodes = self._retrieve_relevant_episodes(query, complexity)
documents = self._retrieve_documents(query, complexity)
bundle = RetrievalBundle(facts=facts, episodes=episodes, documents=documents)
bundle = self._hooks.emit(PipelineStage.RETRIEVED, bundle)

# REASON: builds the prompt from all context
prompt = self._build_prompt(query, awareness, complexity, bundle)
prompt = self._hooks.emit(PipelineStage.PROMPT_BUILT, prompt)

# REFLECT: scores confidence
confidence = self._reflect(prompt, query)
confidence = self._hooks.emit(PipelineStage.REFLECTED, confidence)
```

If no hooks are registered, `emit()` returns data unchanged — zero overhead. The `RetrievalBundle` dataclass is defined in `middleware/hooks.py` alongside `PipelineStage`.

### 3.6 Integration with BrainOrchestrator

The existing `BrainOrchestrator.process()` signature is `process(self, user_input: str) -> str` with no `state` parameter. We add `state` as an optional kwarg for backward compatibility — all existing callers (`cli.py`, `daemon.py`, `overlay.py`) continue to work unchanged.

```python
# In BrainOrchestrator:

class BrainOrchestrator:
    def __init__(self, config, middleware_stack: MiddlewareStack | None = None, ...):
        self._hooks = HookRegistry()  # inner pipeline hooks
        self._middleware = middleware_stack or MiddlewareStack()
        self._cognitive_arch = CognitiveArchitecture(config, hooks=self._hooks)
        # Middleware can register inner hooks via self._hooks during construction

    def process(self, user_input: str, *, state: dict | None = None) -> str:
        """Process a user message. state is optional — omit for backward compat."""
        state = state or {}

        # Outer: before
        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return ""  # blocked by middleware

        # Outer: modify tools and prompt
        tools = self._middleware.apply_tools(self._base_tools)
        system_prompt = self._middleware.apply_prompt(self._base_prompt)

        # Core: run cognitive pipeline + agentic loop
        response = self._cognitive_arch.process(
            message, tools=tools, system_prompt=system_prompt, state=state
        )

        # Outer: after
        response = self._middleware.run_after_turn(response, state)
        return response
```

---

## 4. Backend Protocol

### 4.1 Protocol Interfaces

```python
# src/homie_core/backend/protocol.py

from typing import Protocol, runtime_checkable
from dataclasses import dataclass

@dataclass
class FileInfo:
    path: str
    name: str
    is_dir: bool
    size: int | None = None
    modified: float | None = None

@dataclass
class FileContent:
    content: str
    total_lines: int
    truncated: bool = False

@dataclass
class EditResult:
    success: bool
    occurrences: int = 0
    error: str | None = None

@dataclass
class GrepMatch:
    path: str
    line_number: int
    line: str

@dataclass
class ExecutionResult:
    stdout: str
    stderr: str
    exit_code: int
    timed_out: bool = False

@runtime_checkable
class BackendProtocol(Protocol):
    """Anything that can store and retrieve files."""

    def ls(self, path: str = "/") -> list[FileInfo]: ...
    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent: ...
    def write(self, path: str, content: str) -> None: ...
    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult: ...
    def glob(self, pattern: str) -> list[FileInfo]: ...
    def grep(self, pattern: str, path: str = "/", include: str | None = None) -> list[GrepMatch]: ...

@runtime_checkable
class ExecutableBackend(BackendProtocol, Protocol):
    """Backend that can also run commands."""

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult: ...
```

### 4.2 LocalFilesystemBackend

Default backend for desktop. Replaces hardwired `pathlib`/`subprocess` calls in `builtin_tools.py`.

```python
# src/homie_core/backend/local_filesystem.py

class LocalFilesystemBackend:
    """Local filesystem with path containment and symlink protection."""

    def __init__(self, root_dir: str | Path):
        self._root = Path(root_dir).resolve()

    def _resolve(self, path: str) -> Path:
        """Resolve path within root_dir. Raises ValueError on escape."""
        resolved = (self._root / path.lstrip("/")).resolve()
        if not resolved.is_relative_to(self._root):
            raise ValueError(f"Path escapes root: {path}")
        return resolved

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        resolved = self._resolve(path)
        lines = resolved.read_text(encoding="utf-8").splitlines()
        total = len(lines)
        selected = lines[offset:offset + limit]
        return FileContent(
            content="\n".join(selected),
            total_lines=total,
            truncated=(offset + limit < total),
        )

    def write(self, path: str, content: str) -> None:
        resolved = self._resolve(path)
        resolved.parent.mkdir(parents=True, exist_ok=True)
        if sys.platform != "win32":
            # Unix: use O_NOFOLLOW to prevent symlink writes
            fd = os.open(str(resolved), os.O_WRONLY | os.O_CREAT | os.O_TRUNC | os.O_NOFOLLOW, 0o644)
            try:
                os.write(fd, content.encode("utf-8"))
            finally:
                os.close(fd)
        else:
            # Windows: no symlink risk for non-admin users; check explicitly
            if resolved.is_symlink():
                raise ValueError(f"Refusing to write through symlink: {path}")
            resolved.write_text(content, encoding="utf-8")

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        resolved = self._resolve(path)
        text = resolved.read_text(encoding="utf-8")
        count = text.count(old)
        if count == 0:
            return EditResult(success=False, error=f"String not found in {path}")
        if not replace_all and count > 1:
            return EditResult(success=False, error=f"String found {count} times — not unique. Use replace_all=True or provide more context.")
        new_text = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        self.write(path, new_text)
        return EditResult(success=True, occurrences=count if replace_all else 1)

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        try:
            proc = subprocess.run(
                command, shell=True, capture_output=True, text=True,
                timeout=timeout, cwd=str(self._root),
            )
            return ExecutionResult(
                stdout=proc.stdout, stderr=proc.stderr,
                exit_code=proc.returncode,
            )
        except subprocess.TimeoutExpired:
            return ExecutionResult(stdout="", stderr="Command timed out", exit_code=124, timed_out=True)

    def ls(self, path: str = "/") -> list[FileInfo]:
        resolved = self._resolve(path)
        entries = []
        for child in sorted(resolved.iterdir()):
            stat = child.stat()
            entries.append(FileInfo(
                path=str(child.relative_to(self._root)),
                name=child.name,
                is_dir=child.is_dir(),
                size=stat.st_size if child.is_file() else None,
                modified=stat.st_mtime,
            ))
        return entries

    def glob(self, pattern: str) -> list[FileInfo]:
        matches = []
        for match in self._root.glob(pattern):
            if not match.is_relative_to(self._root):
                continue  # skip escapes
            stat = match.stat()
            matches.append(FileInfo(
                path=str(match.relative_to(self._root)),
                name=match.name,
                is_dir=match.is_dir(),
                size=stat.st_size if match.is_file() else None,
                modified=stat.st_mtime,
            ))
        return matches

    def grep(self, pattern: str, path: str = "/", include: str | None = None) -> list[GrepMatch]:
        """Literal text search across files under path."""
        import re
        resolved = self._resolve(path)
        glob_pattern = include or "**/*"
        matches = []
        for file_path in resolved.glob(glob_pattern):
            if not file_path.is_file() or not file_path.is_relative_to(self._root):
                continue
            try:
                for i, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), 1):
                    if re.search(pattern, line):
                        matches.append(GrepMatch(
                            path=str(file_path.relative_to(self._root)),
                            line_number=i,
                            line=line,
                        ))
            except (UnicodeDecodeError, PermissionError):
                continue  # skip binary/unreadable files
        return matches
```

### 4.3 StateBackend

In-memory backend for tests and sandboxed operations.

```python
# src/homie_core/backend/state.py

class StateBackend:
    """Ephemeral in-memory file storage. Perfect for tests and sub-agent isolation."""

    def __init__(self):
        self._files: dict[str, str] = {}

    def read(self, path: str, offset: int = 0, limit: int = 100) -> FileContent:
        if path not in self._files:
            raise FileNotFoundError(path)
        lines = self._files[path].splitlines()
        selected = lines[offset:offset + limit]
        return FileContent(content="\n".join(selected), total_lines=len(lines), truncated=(offset + limit < len(lines)))

    def write(self, path: str, content: str) -> None:
        self._files[path] = content

    def edit(self, path: str, old: str, new: str, replace_all: bool = False) -> EditResult:
        if path not in self._files:
            return EditResult(success=False, error=f"File not found: {path}")
        text = self._files[path]
        count = text.count(old)
        if count == 0:
            return EditResult(success=False, error=f"String not found in {path}")
        if not replace_all and count > 1:
            return EditResult(success=False, error=f"String found {count} times — not unique.")
        self._files[path] = text.replace(old, new) if replace_all else text.replace(old, new, 1)
        return EditResult(success=True, occurrences=count if replace_all else 1)

    def ls(self, path: str = "/") -> list[FileInfo]:
        prefix = path.rstrip("/") + "/" if path != "/" else "/"
        seen_dirs: set[str] = set()
        entries = []
        for fpath in sorted(self._files):
            if not fpath.startswith(prefix) and path != "/":
                continue
            rel = fpath[len(prefix):] if path != "/" else fpath.lstrip("/")
            parts = rel.split("/")
            if len(parts) == 1:
                entries.append(FileInfo(path=fpath, name=parts[0], is_dir=False, size=len(self._files[fpath])))
            elif parts[0] not in seen_dirs:
                seen_dirs.add(parts[0])
                entries.append(FileInfo(path=prefix + parts[0], name=parts[0], is_dir=True))
        return entries

    def glob(self, pattern: str) -> list[FileInfo]:
        from fnmatch import fnmatch
        return [FileInfo(path=p, name=p.split("/")[-1], is_dir=False, size=len(c))
                for p, c in sorted(self._files.items()) if fnmatch(p, pattern)]

    def grep(self, pattern: str, path: str = "/", include: str | None = None) -> list[GrepMatch]:
        import re
        matches = []
        for fpath, content in sorted(self._files.items()):
            if path != "/" and not fpath.startswith(path):
                continue
            for i, line in enumerate(content.splitlines(), 1):
                if re.search(pattern, line):
                    matches.append(GrepMatch(path=fpath, line_number=i, line=line))
        return matches
```

### 4.4 EncryptedVaultBackend

Decorator that wraps any backend with AES-256-GCM encryption.

```python
# src/homie_core/backend/encrypted.py

class EncryptedVaultBackend:
    """Transparent encrypt-on-write, decrypt-on-read wrapper."""

    def __init__(self, inner: BackendProtocol, key_provider: Callable[[], bytes]):
        self._inner = inner
        self._get_key = key_provider  # e.g., lambda: keyring.get_password("homie", "vault")

    def read(self, path: str, **kwargs) -> FileContent:
        result = self._inner.read(path, **kwargs)
        result.content = self._decrypt(result.content)
        return result

    def write(self, path: str, content: str) -> None:
        self._inner.write(path, self._encrypt(content))

    def _encrypt(self, plaintext: str) -> str:
        """AES-256-GCM encryption, returns base64-encoded ciphertext.
        Uses Homie's existing security/vault.py which already depends on
        the `cryptography` library (already in pyproject.toml).
        Format: base64(nonce || ciphertext || tag) — 12-byte nonce, 16-byte tag.
        Each write generates a fresh random nonce via os.urandom(12)."""
        from homie_core.security.vault import aes_gcm_encrypt
        key = self._get_key()
        return aes_gcm_encrypt(plaintext.encode("utf-8"), key)

    def _decrypt(self, ciphertext: str) -> str:
        """AES-256-GCM decryption from base64-encoded ciphertext.
        Reuses Homie's existing security/vault.py decrypt function."""
        from homie_core.security.vault import aes_gcm_decrypt
        key = self._get_key()
        return aes_gcm_decrypt(ciphertext, key).decode("utf-8")
```

### 4.5 LANBackend

Proxies file/exec operations to paired LAN devices via WebSocket.

```python
# src/homie_core/backend/lan.py

class LANBackend:
    """Proxies operations to a paired LAN node via WebSocket."""

    def __init__(self, peer_address: str, auth_key: bytes):
        self._peer = peer_address
        self._auth = auth_key  # HMAC key from Homie's LAN pairing

    def read(self, path: str, **kwargs) -> FileContent:
        response = self._request("read", {"path": path, **kwargs})
        return FileContent(**response)

    def execute(self, command: str, timeout: int = 30) -> ExecutionResult:
        response = self._request("execute", {"command": command, "timeout": timeout})
        return ExecutionResult(**response)

    def _request(self, method: str, params: dict) -> dict:
        """HMAC-authenticated WebSocket RPC to peer.
        Uses Homie's existing network/lan_discovery.py WebSocket protocol.
        Signs requests with HMAC-SHA256 over (method + json(params) + timestamp).
        Raises ConnectionError if peer is unreachable."""
        import json, hmac, hashlib, time, websockets
        timestamp = str(int(time.time()))
        payload = json.dumps({"method": method, "params": params, "ts": timestamp})
        sig = hmac.new(self._auth, payload.encode(), hashlib.sha256).hexdigest()
        # Send via existing WebSocket connection pool
        response = self._ws_pool.send(self._peer, payload, headers={"X-Sig": sig})
        return json.loads(response)
```

### 4.6 CompositeBackend

Routes by path prefix.

```python
# src/homie_core/backend/composite.py

class CompositeBackend:
    """Routes operations to different backends by path prefix."""

    def __init__(self, default: BackendProtocol, routes: dict[str, BackendProtocol] | None = None):
        self._default = default
        # Sort routes longest-first for correct matching
        self._routes = sorted((routes or {}).items(), key=lambda r: len(r[0]), reverse=True)

    def _route(self, path: str) -> tuple[BackendProtocol, str]:
        """Find backend for path. Returns (backend, relative_path)."""
        for prefix, backend in self._routes:
            if path.startswith(prefix):
                return backend, path[len(prefix):] or "/"
        return self._default, path

    def read(self, path: str, **kwargs) -> FileContent:
        backend, rel_path = self._route(path)
        return backend.read(rel_path, **kwargs)

    # All other methods delegate via self._route()...
```

### 4.7 Refactoring builtin_tools.py

Tools receive a backend instance instead of using raw I/O:

```python
# Before:
def _tool_read_file(path: str) -> str:
    return Path(path).read_text()

# After:
def _tool_read_file(path: str, backend: BackendProtocol) -> str:
    result = backend.read(path)
    return result.content
```

**Backend injection mechanism:** The `ToolRegistry` gains a `context` dict that is set by the orchestrator before each turn. Tool functions that need a backend declare a `backend` parameter (detected via `inspect.signature`). When `ToolRegistry.execute()` calls a tool, it checks whether the tool function accepts a `backend` parameter — if so, it injects `self._context["backend"]` automatically. Tools without a `backend` parameter (e.g., `current_time`, `recall`) are called unchanged.

```python
# In ToolRegistry.execute():
sig = inspect.signature(tool.execute)
if "backend" in sig.parameters:
    call.arguments["backend"] = self._context.get("backend")
result = tool.execute(**call.arguments)
```

Only file/exec tools (`search_files`, `read_file`, `run_command`) are refactored to accept a `backend` parameter. Memory tools (`remember`, `recall`), git tools, clipboard tools, web search, and plugin tools remain unchanged — they don't do file I/O through the backend.

---

## 5. Dynamic Model Switching

### 5.1 Model Tiers and Profiles

```python
# src/homie_core/config.py (additions)

class ModelTier(str, Enum):
    SMALL = "small"    # 1-3B params — greetings, simple lookups
    MEDIUM = "medium"  # 7-14B params — general conversation
    LARGE = "large"    # 30B+ params — complex reasoning, code generation

class ModelProfile(BaseModel):
    name: str                  # "qwen2.5-7b-instruct"
    tier: ModelTier
    context_length: int        # 32768
    supports_tools: bool       # True
    location: str              # "local" | "lan" | "qubrid"
    priority: int = 0          # higher = preferred within tier
```

### 5.2 Model Registry

```python
# src/homie_core/inference/model_registry.py

class ModelRegistry:
    """Discovers and tracks available models across all inference sources."""

    def __init__(self, config: HomieConfig):
        self._profiles: dict[str, ModelProfile] = {}
        self._config = config

    def refresh(self) -> None:
        """Re-scan local engine, LAN peers, and Qubrid for available models."""
        self._profiles.clear()
        self._scan_local()
        self._scan_lan()
        self._scan_qubrid()

    def available(self, tier: ModelTier | None = None) -> list[ModelProfile]:
        """List available models, optionally filtered by tier."""
        profiles = list(self._profiles.values())
        if tier:
            profiles = [p for p in profiles if p.tier == tier]
        return sorted(profiles, key=lambda p: p.priority, reverse=True)

    def best_for(self, tier: ModelTier) -> ModelProfile | None:
        """Return highest-priority available model for a tier, or None."""
        candidates = self.available(tier)
        return candidates[0] if candidates else None

    def _scan_local(self) -> None:
        """Check local model engine for loaded/available models.
        Reads config.llm.model_path for the currently configured model.
        Also scans config.storage.path/models/ for downloaded GGUF files.
        Estimates tier from file size: <4GB=SMALL, <16GB=MEDIUM, else LARGE."""
        model_dir = Path(self._config.storage.path) / "models"
        if model_dir.exists():
            for gguf in model_dir.glob("*.gguf"):
                size_gb = gguf.stat().st_size / (1024**3)
                tier = ModelTier.SMALL if size_gb < 4 else ModelTier.MEDIUM if size_gb < 16 else ModelTier.LARGE
                self._profiles[gguf.stem] = ModelProfile(
                    name=gguf.stem, tier=tier, context_length=4096,
                    supports_tools=True, location="local",
                )

    def _scan_lan(self) -> None:
        """Query LAN peers for their available models.
        Uses Homie's existing network/lan_discovery.py mDNS service.
        Sends a 'list_models' RPC to each discovered peer."""
        # Delegates to network module — returns list of ModelProfile dicts
        pass  # implemented when LANBackend is wired (Step 7)

    def _scan_qubrid(self) -> None:
        """Query Qubrid API for available cloud models.
        Uses Homie's existing inference/qubrid.py client.
        Qubrid models are always LARGE tier with high context lengths."""
        if not self._config.inference.qubrid.api_key:
            return
        # Query Qubrid's OpenAI-compatible /models endpoint
        from homie_core.inference.qubrid import QubridClient
        client = QubridClient(self._config.inference.qubrid)
        for model in client.list_models():
            self._profiles[f"qubrid:{model.id}"] = ModelProfile(
                name=model.id, tier=ModelTier.LARGE,
                context_length=model.context_length or 32768,
                supports_tools=True, location="qubrid", priority=-1,  # prefer local
            )
```

### 5.3 Complexity-to-Tier Mapping

Leverages the existing cognitive classifier:

| Complexity (from CLASSIFY stage) | Model Tier | Rationale |
|---|---|---|
| Trivial | SMALL | "hi", "thanks" — fast response preferred |
| Simple | SMALL | Factual lookup — doesn't need large model |
| Moderate | MEDIUM | Context + reasoning — balanced |
| Complex | LARGE | Multi-step synthesis — needs capability |
| Deep | LARGE | Full planning — maximum capability |

### 5.4 ModelResolverMiddleware

```python
# src/homie_core/brain/model_resolver.py

class ModelResolverMiddleware(HomieMiddleware):
    """Selects the best model for each turn based on complexity and availability."""

    name = "model_resolver"
    order = 50  # runs after prompt modification, before tool middleware

    TIER_MAP = {
        "trivial": ModelTier.SMALL,
        "simple": ModelTier.SMALL,
        "moderate": ModelTier.MEDIUM,
        "complex": ModelTier.LARGE,
        "deep": ModelTier.LARGE,
    }

    def __init__(self, registry: ModelRegistry, hooks: HookRegistry):
        self._registry = registry
        self._complexity: str | None = None
        # Listen to inner hook for complexity classification
        hooks.register(PipelineStage.CLASSIFIED, self._on_classified)

    def _on_classified(self, stage: PipelineStage, complexity: str) -> str:
        """Capture complexity and update state with the correct model.

        This is the PRIMARY resolution point. The cognitive pipeline runs
        CLASSIFY before inference, so by the time the model is actually
        needed (during ADAPT/generation), state["active_model"] is correct.

        Resolution timing:
        1. before_turn: sets initial model from explicit override or MEDIUM default
        2. on_classified hook: updates state["active_model"] based on actual complexity
        3. ADAPT stage: orchestrator reads state["active_model"] for inference
        """
        tier = self.TIER_MAP.get(complexity, ModelTier.MEDIUM)
        profile = self._registry.best_for(tier)
        if profile and self._state is not None:
            self._state["active_model"] = profile.name
            self._state["active_model_location"] = profile.location
        self._complexity = complexity
        return complexity  # pass through unchanged

    def before_turn(self, message: str, state: dict) -> str:
        """Set initial model (may be overridden by on_classified hook)."""
        self._state = state  # hold reference so on_classified can update it

        # 1. Check explicit override (takes precedence over everything)
        if "model_override" in state:
            state["active_model"] = state.pop("model_override")
            return message

        # 2. Set MEDIUM as default — will be refined when CLASSIFY runs
        profile = self._registry.best_for(ModelTier.MEDIUM)
        if profile:
            state["active_model"] = profile.name
            state["active_model_location"] = profile.location

        return message
```

### 5.5 Integration with InferenceRouter

The InferenceRouter remains unchanged in its role (deciding *where* to run). The `active_model` and `active_model_location` set by ModelResolverMiddleware are consumed by the orchestrator when invoking the router:

```python
# In BrainOrchestrator, when calling inference:
model = state.get("active_model", self._config.llm.model_path)
location = state.get("active_model_location", "local")
response = self._inference_router.generate(prompt, model=model, preferred_location=location)
```

The router's priority chain (local → LAN → Qubrid) still applies, but now with a hint about which model and location to prefer.

---

## 6. Migration Strategy

Incremental, non-breaking rollout in 7 steps:

| Step | Change | Risk | Behavior Change |
|------|--------|------|-----------------|
| 1 | Add `middleware/` and `backend/` modules with base classes + `LocalFilesystemBackend` | None | None — new code only |
| 2 | Refactor file/exec tools in `builtin_tools.py` (`search_files`, `read_file`, `run_command`) to accept a backend, default to `LocalFilesystemBackend(root_dir=config.storage.path)`. Memory, git, clipboard, notes, web, and plugin tools remain unchanged. | Low | Identical behavior, now path-contained |
| 3 | Add inner hook emission points to `cognitive_arch.py` | None | No-op if no hooks registered |
| 4 | Wire `MiddlewareStack` into `BrainOrchestrator` — empty stack = identical to current | Low | Zero change with empty stack |
| 5 | Add `ModelRegistry` + `ModelResolverMiddleware` | Low | New capability, opt-in via config |
| 6 | Add `StateBackend`, `EncryptedVaultBackend`, `CompositeBackend` | None | New code, used on demand |
| 7 | Add `LANBackend` (depends on network module) | Low | New capability for LAN-paired devices |

At every step, the test suite should pass with zero regressions.

---

## 7. File Inventory

### New Files

| File | Purpose |
|------|---------|
| `src/homie_core/middleware/__init__.py` | Public API exports |
| `src/homie_core/middleware/base.py` | `HomieMiddleware` base class |
| `src/homie_core/middleware/stack.py` | `MiddlewareStack` execution engine |
| `src/homie_core/middleware/hooks.py` | `HookRegistry`, `PipelineStage` enum |
| `src/homie_core/backend/__init__.py` | Public API exports |
| `src/homie_core/backend/protocol.py` | `BackendProtocol`, `ExecutableBackend`, data classes |
| `src/homie_core/backend/local_filesystem.py` | `LocalFilesystemBackend` |
| `src/homie_core/backend/state.py` | `StateBackend` |
| `src/homie_core/backend/encrypted.py` | `EncryptedVaultBackend` |
| `src/homie_core/backend/lan.py` | `LANBackend` |
| `src/homie_core/backend/composite.py` | `CompositeBackend` |
| `src/homie_core/brain/model_resolver.py` | `ModelResolverMiddleware` |
| `src/homie_core/inference/model_registry.py` | `ModelRegistry` |

### Modified Files

| File | Change |
|------|--------|
| `src/homie_core/brain/cognitive_arch.py` | Add `self._hooks.emit()` calls at 5 stage boundaries |
| `src/homie_core/brain/orchestrator.py` | Accept `MiddlewareStack`, run outer lifecycle around `process()` |
| `src/homie_core/brain/builtin_tools.py` | Replace raw `pathlib`/`subprocess` with `backend` parameter |
| `src/homie_core/brain/tool_registry.py` | Add `wrap_tool_call`/`wrap_tool_result` middleware hooks |
| `src/homie_core/config.py` | Add `ModelTier`, `ModelProfile`, `BackendConfig` |
| `src/homie_core/inference/router.py` | Accept `model` and `preferred_location` hints |

---

## 8. Design Principles

1. **Zero new dependencies** — pure Python + existing deps (`cryptography` for vault), no LangChain, no LangGraph
2. **Structural typing** — `Protocol` classes for interfaces, not inheritance hierarchies
3. **Empty stack = current behavior** — middleware is additive, never required
4. **Inner hooks are signals** — observe and modify stage outputs, not full middleware lifecycle
5. **Backends are composable** — `EncryptedVaultBackend(LANBackend(...))` stacking works
6. **Model resolution is separate from routing** — *what* model (ModelResolver) vs *where* to run (InferenceRouter)
7. **Incremental migration** — each step is independently deployable and testable

---

## 9. Future Phase Unlocks

This foundation enables all subsequent phases as middleware:

| Phase | Enabled By |
|-------|-----------|
| Phase 2: Context Intelligence | `SummarizationMiddleware`, `LargeResultEvictionMiddleware`, `ArgTruncationMiddleware` — all outer middleware |
| Phase 3: Multi-Agent | `SubAgentMiddleware` spawns child orchestrators with their own `MiddlewareStack` + `StateBackend` |
| Phase 4: Safety | `HITLMiddleware` gates `wrap_tool_call()`, `ShellAllowlistMiddleware` filters `execute()` |
| Phase 5: Extensibility | `SkillsMiddleware` uses `modify_prompt()`, `MemoryMiddleware` uses `before_turn()`, `HooksMiddleware` uses `after_turn()` |

---

## 10. Testing Strategy

| Component | Test Approach |
|-----------|--------------|
| `MiddlewareStack` | Unit tests with mock middleware verifying order, onion unwinding, blocking |
| `HookRegistry` | Unit tests verifying emission, modification, no-op when empty |
| `BackendProtocol` implementations | Each backend tested against the same interface contract (parametrized) |
| `LocalFilesystemBackend` | Path containment tests (escape attempts), symlink protection, timeout handling |
| `StateBackend` | In-memory CRUD, used as test double for other components |
| `CompositeBackend` | Routing correctness, longest-prefix matching, aggregation for `ls`/`grep` |
| `ModelResolverMiddleware` | Complexity-to-tier mapping, override precedence, fallback behavior |
| `BrainOrchestrator` integration | Empty stack produces identical output to current, middleware modifications apply correctly |

All existing tests must pass unchanged after each migration step.
