# Phase 2: Context Intelligence — Design Spec

**Date:** 2026-03-16
**Status:** Approved
**Branch:** `feat/homie-ai-v2`
**Scope:** Token-aware summarization, large result eviction, long line splitting, argument truncation, context overflow recovery
**Depends on:** Phase 1 (middleware architecture, backend protocol)

---

## 1. Overview

Five middleware implementations that manage context window usage intelligently. All build on Phase 1's `HomieMiddleware` base class and use the `BackendProtocol` for persisting evicted content.

| Middleware | Order | Hook Point | Purpose |
|-----------|-------|------------|---------|
| `ContextOverflowRecoveryMiddleware` | 1 | `after_turn` | Safety net — emergency compaction on overflow errors |
| `SummarizationMiddleware` | 5 | `before_turn` + tool | Token-aware auto-compaction at 85% context |
| `ArgTruncationMiddleware` | 8 | `before_turn` | Trim old write/edit args to save tokens |
| `LongLineSplitMiddleware` | 85 | `wrap_tool_result` | Split lines > 5000 chars with continuation markers |
| `LargeResultEvictionMiddleware` | 90 | `wrap_tool_result` | Evict results > 80k chars to backend with preview |

---

## 2. Token Estimation

All thresholds use character-based approximation: `tokens ≈ chars / 4`, matching the existing `_CHARS_PER_TOKEN = 4` constant in `orchestrator.py`. This is sufficient for threshold decisions — we're deciding "is context getting too big?" not computing exact token counts.

Helper function:

```python
def estimate_tokens(text: str) -> int:
    return len(text) // 4

def estimate_conversation_tokens(conversation: list[dict]) -> int:
    return sum(estimate_tokens(msg.get("content", "")) for msg in conversation)
```

Defined in `src/homie_core/middleware/token_utils.py` and shared across all middleware.

---

## 3. Configuration

```python
# src/homie_core/config.py — addition

class ContextConfig(BaseModel):
    summarize_trigger_pct: float = 0.85  # trigger auto-summarization at this % of context_length
    summarize_keep_pct: float = 0.10     # keep this % of context after compaction
    manual_compact_pct: float = 0.50     # agent can self-compact at this % of trigger
    large_result_threshold: int = 80000  # chars — tool results above this get evicted
    long_line_threshold: int = 5000      # chars — lines above this get split
    arg_truncation_threshold: int = 2000 # chars — old write/edit args above this get trimmed
```

Added to `HomieConfig` as `context: ContextConfig = Field(default_factory=ContextConfig)`.

The trigger threshold in tokens is: `config.llm.context_length * config.context.summarize_trigger_pct`. Since we use char-based estimation, the char threshold is: `trigger_tokens * 4`.

---

## 4. SummarizationMiddleware

### 4.1 Responsibility

Compresses conversation history when estimated token usage approaches the model's context window. Replaces the existing `ContextCompressor` call in `cognitive_arch.py:_prepare_prompt()`.

### 4.2 Interface

```python
class SummarizationMiddleware(HomieMiddleware):
    name = "summarization"
    order = 5

    def __init__(self, config: HomieConfig, backend: BackendProtocol,
                 working_memory: WorkingMemory):
        self._config = config
        self._backend = backend
        self._wm = working_memory

    @property
    def _trigger_chars(self) -> int:
        return int(self._config.llm.context_length * self._config.context.summarize_trigger_pct * 4)

    def _make_compressor(self, conversation_length: int) -> ContextCompressor:
        """Create a compressor with keep_count computed from current conversation size."""
        keep_count = max(3, int(conversation_length * self._config.context.summarize_keep_pct))
        return ContextCompressor(
            threshold_chars=0,  # we handle threshold externally
            protect_first_n=2,
            protect_last_n=keep_count,
            summary_target_chars=800,
        )
```

### 4.3 before_turn

```python
def before_turn(self, message: str, state: dict) -> str:
    conversation = self._wm.get_conversation()
    total_chars = sum(len(m.get("content", "")) for m in conversation)

    if total_chars > self._trigger_chars:
        # Offload full history to backend before compressing
        self._offload_history(conversation, state)
        # Compress using ContextCompressor with keep_count based on current size
        compressor = self._make_compressor(len(conversation))
        compressed = compressor.compress(conversation)
        self._wm._conversation = compressed
    return message
```

### 4.4 Tool: compact_conversation

Exposed via `modify_tools` — adds a `compact_conversation` tool that the agent can call manually when > 50% of trigger threshold:

```python
def modify_tools(self, tools: list[dict]) -> list[dict]:
    return tools + [compact_conversation_tool]
```

The tool checks if current usage exceeds `manual_compact_pct * trigger_chars`. If so, runs compression. If not, returns "Not enough context to compact yet."

### 4.5 History Offloading

When summarization triggers, the full conversation before compression is written to the backend:

```python
def _offload_history(self, conversation: list[dict], state: dict) -> None:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    content = self._format_conversation(conversation)
    self._backend.write(f"/conversation_history/{timestamp}.md", content)
```

### 4.6 Replacing the Old Compressor

The existing auto-compress block in `cognitive_arch.py:_prepare_prompt()` is removed (the block that checks `self._compressor.needs_compression` and replaces `self._wm._conversation`):

```python
# REMOVE this block:
# conversation = self._wm.get_conversation()
# if self._compressor.needs_compression(conversation):
#     compressed = self._compressor.compress(conversation)
#     self._wm._conversation = compressed
```

The `ContextCompressor` class itself is NOT deleted — `SummarizationMiddleware` reuses it internally. Only the auto-trigger in `_prepare_prompt` is removed since the middleware now handles timing.

---

## 5. LargeResultEvictionMiddleware

### 5.1 Responsibility

Prevents oversized tool outputs from consuming the context window. Writes full results to backend, returns a head+tail preview to the agent.

### 5.2 Interface

```python
class LargeResultEvictionMiddleware(HomieMiddleware):
    name = "large_result_eviction"
    order = 90

    # Tools that already self-truncate — skip eviction
    EXCLUDED_TOOLS = {"ls", "glob", "grep", "read_file", "edit_file", "write_file",
                      "search_files"}

    def __init__(self, config: HomieConfig, backend: BackendProtocol):
        self._threshold = config.context.large_result_threshold
        self._backend = backend
```

### 5.3 wrap_tool_result

```python
def wrap_tool_result(self, name: str, result: str) -> str:
    if name in self.EXCLUDED_TOOLS:
        return result
    if len(result) <= self._threshold:
        return result

    # Evict to backend
    safe_name = re.sub(r'[^\w\-.]', '_', name)
    path = f"/large_tool_results/{safe_name}_{id(result)}.md"
    self._backend.write(path, result)

    # Return preview: first 5 + last 5 lines
    lines = result.splitlines()
    head = "\n".join(lines[:5])
    tail = "\n".join(lines[-5:])
    return (
        f"{head}\n\n"
        f"... ({len(lines)} total lines, {len(result)} chars) ...\n"
        f"Full output saved to: {path}\n\n"
        f"{tail}"
    )
```

---

## 6. LongLineSplitMiddleware

### 6.1 Responsibility

Splits extremely long lines (e.g., minified JSON/CSS) to prevent single lines from dominating the context window.

### 6.2 Interface

```python
class LongLineSplitMiddleware(HomieMiddleware):
    name = "long_line_split"
    order = 85

    def __init__(self, config: HomieConfig):
        self._threshold = config.context.long_line_threshold
```

### 6.3 wrap_tool_result

```python
def wrap_tool_result(self, name: str, result: str) -> str:
    lines = result.splitlines()
    output_lines = []
    for i, line in enumerate(lines, 1):
        if len(line) <= self._threshold:
            output_lines.append(line)
        else:
            # Split into chunks with continuation markers
            chunks = [line[j:j + self._threshold]
                      for j in range(0, len(line), self._threshold)]
            for k, chunk in enumerate(chunks, 1):
                output_lines.append(f"[line {i}.{k}] {chunk}")
    return "\n".join(output_lines)
```

---

## 7. ArgTruncationMiddleware

### 7.1 Responsibility

Pre-summarization optimization: truncates large `content` arguments from old `write_file` and `edit_file` tool calls in conversation history. Runs before the summarizer to reduce token count before compression even starts.

### 7.2 Interface

```python
class ArgTruncationMiddleware(HomieMiddleware):
    name = "arg_truncation"
    order = 8

    TRUNCATABLE_TOOLS = {"write_file", "edit_file"}

    def __init__(self, config: HomieConfig, working_memory: WorkingMemory):
        self._threshold = config.context.arg_truncation_threshold
        self._wm = working_memory
```

### 7.3 before_turn

Scans conversation history. For messages older than the keep window (protect last 3), checks content length and truncates large assistant messages that contain tool call patterns. Works on the content string directly — if an assistant message contains a tool call for `write_file` or `edit_file` and the entire message content exceeds the threshold, the content is truncated:

```python
def before_turn(self, message: str, state: dict) -> str:
    conversation = self._wm.get_conversation()
    if len(conversation) < 6:
        return message  # too short to truncate

    # Only truncate messages outside the keep window (protect last 3)
    for msg in conversation[:-3]:
        content = msg.get("content", "")
        if not content or msg.get("role") != "assistant":
            continue
        if len(content) <= self._threshold:
            continue
        # Check if this message contains a truncatable tool call
        if self._has_truncatable_tool(content):
            msg["content"] = content[:20] + "...(argument truncated)"
    return message

def _has_truncatable_tool(self, content: str) -> bool:
    """Check if content contains a write_file or edit_file tool call."""
    # Works with Homie's tool call formats: <tool>write_file(...)</tool>,
    # {"tool": "write_file", ...}, Action: write_file(...)
    lower = content.lower()
    return any(tool in lower for tool in self.TRUNCATABLE_TOOLS)
```

This approach is format-agnostic — it doesn't parse the tool call structure, just detects the presence of truncatable tool names in large assistant messages. The actual content is replaced wholesale since the value of old write/edit arguments is purely historical.

---

## 8. ContextOverflowRecoveryMiddleware

### 8.1 Responsibility

Safety net for when token estimation underestimates and the model call hits a context overflow. Instead of crashing, triggers emergency summarization and retries.

### 8.2 Interface

```python
class ContextOverflowRecoveryMiddleware(HomieMiddleware):
    name = "context_overflow_recovery"
    order = 1  # runs earliest on before_turn, latest on after_turn

    def __init__(self, working_memory: WorkingMemory):
        self._wm = working_memory
        self._compressor = ContextCompressor(
            threshold_chars=0,  # always compress when called
            protect_first_n=1,
            protect_last_n=3,
            summary_target_chars=400,
        )
```

### 8.3 Integration

This middleware works through the state dict. When a context overflow error occurs during inference, the orchestrator catches it and sets `state["context_overflow"] = True`. On the next pass through the middleware stack, `ContextOverflowRecoveryMiddleware.before_turn()` detects this flag and triggers emergency compaction:

```python
def before_turn(self, message: str, state: dict) -> str:
    if state.pop("context_overflow", False):
        conversation = self._wm.get_conversation()
        compressed = self._compressor.compress(conversation)
        self._wm._conversation = compressed
    return message
```

The orchestrator modifications in both `process()` and `process_stream()`:

```python
# In process():
try:
    response = self._cognitive.process(message)
except Exception as e:
    if self._is_context_overflow(e):
        state["context_overflow"] = True
        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return ""
        response = self._cognitive.process(message)
    else:
        raise

# In process_stream():
try:
    for token in self._cognitive.process_stream(message):
        tokens.append(token)
        yield token
except Exception as e:
    if self._is_context_overflow(e):
        state["context_overflow"] = True
        message = self._middleware.run_before_turn(user_input, state)
        if message is None:
            return
        # Fall back to blocking mode after overflow recovery
        response = self._cognitive.process(message)
        yield response
    else:
        raise

# Helper method:
def _is_context_overflow(self, error: Exception) -> bool:
    msg = str(error).lower()
    return "context" in msg and ("overflow" in msg or "length" in msg or "too long" in msg)
```

---

## 9. File Inventory

### New Files

| File | Purpose |
|------|---------|
| `src/homie_core/middleware/token_utils.py` | `estimate_tokens()`, `estimate_conversation_tokens()` |
| `src/homie_core/middleware/summarization.py` | `SummarizationMiddleware` |
| `src/homie_core/middleware/large_result_eviction.py` | `LargeResultEvictionMiddleware` |
| `src/homie_core/middleware/long_line_split.py` | `LongLineSplitMiddleware` |
| `src/homie_core/middleware/arg_truncation.py` | `ArgTruncationMiddleware` |
| `src/homie_core/middleware/context_overflow.py` | `ContextOverflowRecoveryMiddleware` |
| Tests for each (6 test files) |

### Modified Files

| File | Change |
|------|--------|
| `src/homie_core/config.py` | Add `ContextConfig` class and `context` field to `HomieConfig` |
| `src/homie_core/brain/cognitive_arch.py` | Remove auto-compress block in `_prepare_prompt()` (lines ~659-663) |
| `src/homie_core/brain/orchestrator.py` | Add context overflow catch + retry in `process()` |
| `src/homie_core/middleware/__init__.py` | Export new middleware classes |

---

## 10. Migration Strategy

| Step | Change | Risk |
|------|--------|------|
| 1 | Add `token_utils.py` + `ContextConfig` | None — new code only |
| 2 | Add `LongLineSplitMiddleware` | None — new middleware |
| 3 | Add `LargeResultEvictionMiddleware` | None — new middleware |
| 4 | Add `ArgTruncationMiddleware` | Low — modifies conversation history in-place |
| 5 | Add `SummarizationMiddleware` + remove old auto-compress | Medium — replaces existing compressor trigger |
| 6 | Add `ContextOverflowRecoveryMiddleware` + orchestrator catch | Low — safety net only |

---

## 11. Design Principles

1. **All middleware, no core changes** — every feature is a `HomieMiddleware` subclass (except the orchestrator overflow catch)
2. **Reuse existing compressor** — `ContextCompressor` logic is proven; we wrap it with token-aware thresholds
3. **Character approximation is sufficient** — `chars / 4` for threshold decisions, no new dependencies
4. **Evicted content is preserved** — offloaded to backend, not deleted
5. **Onion ordering matters** — `before_turn` execution order: overflow recovery (1) → summarization (5) → arg truncation (8). Check overflow first, then summarize, then truncate remaining args.
6. **Protected pairs** — tool-call/result pairs are never split during summarization (inherited from existing compressor)

---

## 12. Testing Strategy

| Component | Test Approach |
|-----------|--------------|
| `token_utils` | Unit tests: known strings → expected token counts |
| `SummarizationMiddleware` | Conversation exceeding threshold → compressed; below → untouched; history offloaded to backend; compact_conversation tool |
| `LargeResultEvictionMiddleware` | Result > threshold → evicted with preview; < threshold → passthrough; excluded tools → passthrough |
| `LongLineSplitMiddleware` | Long line → split with markers; normal lines → untouched |
| `ArgTruncationMiddleware` | Old messages with large tool args → truncated; recent messages → protected; non-tool messages → untouched |
| `ContextOverflowRecoveryMiddleware` | state["context_overflow"]=True → compresses; no flag → passthrough |
| Integration | Full middleware stack with all 5 → conversation stays within limits through multiple turns |
