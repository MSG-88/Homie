# SP1: Make Homie Actually Work — Design Spec

**Date:** 2026-04-05
**Goal:** Fix all critical broken paths so Homie works reliably as a local AI assistant on Windows/Linux/Mac.

---

## 1. Scope

Fix the 5 critical blockers that prevent Homie from being a working product:

1. **LAN Backend** — Currently raises NotImplementedError. Implement real peer-to-peer model serving via mDNS discovery.
2. **Finetune Merge** — QLoRA → merged model → GGUF pipeline is incomplete. Complete the merge and quantization flow.
3. **Knowledge Graph** — Entity extraction and relationship inference exist but aren't wired to the brain's reasoning pipeline. Connect them.
4. **Error Handling** — Many silent try/except blocks. Add structured logging and graceful degradation.
5. **Test Coverage** — Integration tests exist but gaps remain. Target 80%+ on core modules.

## 2. LAN Backend

**File:** `src/homie_core/backend/lan.py`

**Current state:** Raises `NotImplementedError`

**Design:**
- Use `zeroconf` (mDNS/DNS-SD) for automatic discovery of other Homie instances on the LAN
- Each Homie instance advertises an OpenAI-compatible API endpoint on a random port
- The LAN backend connects to discovered peers and routes inference requests
- Fallback chain: local GGUF → LAN peer → cloud (Qubrid)
- Health checks every 30s to detect peer availability
- No authentication needed for LAN (trusted network assumption, matching existing config)

**Interface:**
```python
class LanBackend:
    async def discover_peers(self) -> list[PeerInfo]
    async def generate(self, prompt, **kwargs) -> str
    async def health_check(self) -> bool
```

## 3. Finetune Merge Pipeline

**File:** `src/homie_core/finetune/training/merge.py`

**Current state:** Raises `NotImplementedError("Requires model files")`

**Design:**
- Accept a base model path + LoRA adapter path
- Merge using `peft` library's `merge_and_unload()`
- Save merged model to temp directory
- Quantize to GGUF using `llama.cpp`'s `convert.py` if available, or `ctransformers`
- Update the model registry with the new merged model
- Cleanup temp files on success

**Interface:**
```python
class MergeEngine:
    def merge_lora(self, base_model: str, adapter_path: str) -> str  # returns merged path
    def quantize_gguf(self, model_path: str, quant_type: str = "Q4_K_M") -> str  # returns GGUF path
    def merge_and_quantize(self, base_model: str, adapter_path: str) -> str  # full pipeline
```

## 4. Knowledge Graph Integration

**Current state:** Entity extraction framework exists in the brain but isn't wired to reasoning.

**Design:**
- Wire entity extraction into the brain's PERCEIVE stage
- During RETRIEVE stage, query the knowledge graph for related entities
- During REASON stage, include graph context in the synthesis prompt
- Store new entities/relationships after each conversation turn
- Use existing SQLite for graph storage (simple adjacency table, no external graph DB)

**Changes:**
- `src/homie_core/brain/cogarc.py` — Add graph context to perceive/retrieve/reason stages
- `src/homie_core/intelligence/knowledge/` — Ensure entity extractor and relationship builder are called

## 5. Error Handling

**Current state:** Many `except Exception: pass` or `except: log.debug()` patterns.

**Design:**
- Replace silent failures with structured logging (level-appropriate)
- Critical paths (inference, memory, voice) get retry logic with exponential backoff
- Non-critical paths (plugins, telemetry) get graceful degradation with user notification
- Add a health dashboard command (`/health`) that reports component status
- All subsystems report status to the self-healing watchdog

**Scope:** Focus on the top 20 most impactful error paths, not every single try/except.

## 6. Test Coverage

**Target:** 80%+ on core modules (brain, memory, inference, RAG)

**Design:**
- Add unit tests for each brain stage (perceive, classify, retrieve, reason, reflect, adapt)
- Add integration tests for inference router (mock backends)
- Add memory system tests (write/read/consolidate/forget cycles)
- Add RAG pipeline tests (ingest → search → retrieve)
- Add plugin loading tests
- Use pytest with fixtures for common setup

## 7. Implementation Approach

All changes are PRs to MuthuGsubramanian/Homie from this dev machine. Claude Desktop on the GPU laptop will test and apply them.

**PR Strategy:**
- PR 1: LAN Backend implementation
- PR 2: Finetune merge pipeline
- PR 3: Knowledge graph wiring
- PR 4: Error handling improvements
- PR 5: Test coverage expansion

Each PR is independent and can be reviewed/merged separately.
