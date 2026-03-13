# Voice Integration Design — Speech-to-Speech for Homie

**Date:** 2026-03-13
**Status:** Approved
**Approach:** Cherry-pick from HuggingFace speech-to-speech repo, keep Homie's architecture

## Overview

Add full voice interaction to Homie — the user can speak to Homie and hear spoken responses, with live transcription in the overlay. Voice feeds through the existing brain/cognitive pipeline with zero changes to the intelligence layer.

The design adopts the best components and patterns from [huggingface/speech-to-speech](https://github.com/huggingface/speech-to-speech) (queue-based threading, Silero VAD, multi-engine TTS) while preserving Homie's architecture. No separate LLM — all voice queries go through `BrainOrchestrator.process_stream()`.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        VoiceManager                                 │
│  (orchestrates modes, owns component lifecycle, config-driven)      │
│                                                                     │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐      │
│  │ AudioIn  │──▶│ SileroVAD│──▶│   STT    │──▶│  Brain   │      │
│  │ (sounddev│    │ (neural) │    │ (faster- │    │ (existing│      │
│  │  ice)    │    │          │    │  whisper) │    │ cognitive│      │
│  └──────────┘    └──────────┘    └──────────┘    │ pipeline)│      │
│                                                   └────┬─────┘      │
│                                                        │            │
│  ┌──────────┐    ┌─────────────────────────────┐       │            │
│  │ AudioOut │◀──│ TTS Engine (switchable)      │◀─────┘            │
│  │ (sounddev│    │  ├─ PiperTTS   (fast mode)  │                    │
│  │  ice)    │    │  ├─ KokoroTTS  (quality)    │                    │
│  │          │    │  └─ MeloTTS    (multilang)  │                    │
│  └──────────┘    └─────────────────────────────┘                    │
│                                                                     │
│  Threading: Queue-based producer-consumer per component             │
│  Barge-in: should_listen Event flag stops TTS + flushes queues      │
└─────────────────────────────────────────────────────────────────────┘
```

**Key principles:**

- **No separate LLM** — voice queries go through `BrainOrchestrator.process_stream()`, same as text.
- **Queue-based threading** — each component (VAD, STT, TTS, AudioOut) runs in its own thread, connected by `queue.Queue`. Adopted from HF repo's `BaseHandler` pattern.
- **Barge-in** — a `should_listen` threading.Event. When VAD detects speech during TTS playback, it sets the flag, which immediately stops audio output and flushes the TTS queue.
- **Mode-agnostic core** — pipeline components don't know about modes. `VoiceManager` handles mode logic by controlling when the pipeline starts/stops listening.

## Voice Modes & State Machine

Three modes unified under a single state machine:

```
                    ┌─────────────────────────────┐
                    │          IDLE                │
                    │  (pipeline warm, not listening)│
                    └──────┬──────────┬───────────┘
                           │          │
                   wake word detected  hotkey (ctrl+8)
                           │          │
                    ┌──────▼──────────▼───────────┐
                    │        LISTENING              │
                    │  (VAD active, waiting for     │
                    │   speech segments)             │
                    └──────────┬───────────────────┘
                               │
                        speech detected
                               │
                    ┌──────────▼───────────────────┐
                    │        RECORDING              │
                    │  (accumulating audio,          │
                    │   VAD tracking silence)        │
                    └──────────┬───────────────────┘
                               │
                     silence threshold reached
                               │
                    ┌──────────▼───────────────────┐
                    │       PROCESSING              │
                    │  (STT → Brain → response)     │
                    └──────────┬───────────────────┘
                               │
                    ┌──────────▼───────────────────┐
                    │        SPEAKING               │
                    │  (TTS playback, barge-in       │
                    │   monitoring active)           │
                    └──────┬──────────┬───────────┘
                           │          │
                    barge-in!    playback done
                           │          │
                    ┌──────▼──┐  ┌────▼────────────┐
                    │LISTENING│  │ mode check       │
                    └─────────┘  └────┬────────────┘
                                      │
                          ┌───────────┼───────────┐
                          │           │           │
                    conversational  push-to-talk  wake word
                          │           │           │
                    ┌─────▼───┐ ┌─────▼───┐ ┌────▼────┐
                    │LISTENING│ │  IDLE   │ │  IDLE   │
                    └─────────┘ └─────────┘ └─────────┘
```

### Mode behaviors

| Mode | Activation | After response | Exit |
|------|-----------|----------------|------|
| **Wake word** | "Hey Homie" detected | → IDLE (single turn) | Automatic |
| **Push-to-talk** | Hotkey held/toggled | → IDLE | Release key / toggle |
| **Conversational** | `homie voice` CLI or "let's talk" | → LISTENING (stays in loop) | "Goodbye" / silence timeout → confirmation |

### Conversational exit flow

1. Silence timeout (configurable, default 2min) or exit phrase ("goodbye", "stop", "that's all")
2. Homie asks: *"Would you like to end our conversation?"*
3. User confirms → IDLE. User says no → LISTENING resumes.
4. Timer only runs during LISTENING state — pauses during PROCESSING/SPEAKING.
5. **Max exit prompts: 3.** If silence triggers 3 consecutive exit confirmations with no user response, Homie exits automatically with "Ending the conversation since you seem to be away. Talk to you later!"

### Hotkey behavior (Ctrl+8)

**Breaking change:** The codebase currently uses `alt+8` as the default hotkey. This spec changes it to `ctrl+8`. This requires adding a `"ctrl+8": "<ctrl>+8"` mapping to `_HOTKEY_MAP` in `hotkey.py` and updating the default in `VoiceConfig` and `daemon.py`.

| Current state | Hotkey press | Result |
|---|---|---|
| IDLE, voice enabled | Ctrl+8 | Open overlay in voice mode, start LISTENING |
| IDLE, voice disabled | Ctrl+8 | Open overlay in text mode (unchanged) |
| LISTENING/RECORDING | Ctrl+8 | Cancel current recording, return to IDLE |
| SPEAKING | Ctrl+8 | Stop TTS playback, return to LISTENING |
| Conversational active | Ctrl+8 | Toggle mute/unmute |

## Component Design

### VAD: Silero VAD (replacing energy-based)

- Neural network-based via `torch.hub.load("snakers4/silero-vad")`
- Speech probability score (0.0–1.0), configurable threshold (default 0.5)
- Hysteresis: trigger at 0.5, release at 0.35 to prevent flickering
- `min_silence_duration_ms`: 300ms (conversational), 600ms (wake word) — configurable per mode
- Tiny model (~2MB), runs on CPU — no GPU contention with LLM
- **Degradation chain:** Silero → webrtcvad (existing `VAD` class) → energy-based (`VoiceActivityDetector`). All three tiers preserved.

### STT: faster-whisper (existing, upgraded config)

- Language auto-detection with code passthrough for TTS routing
- Model hot-switching per mode:
  - `tiny.en` for push-to-talk (speed)
  - `medium` for conversational (accuracy)
- Multilingual: `medium` and `large-v3` support Tamil, Telugu, Malayalam, French, Spanish
- English-only modes use `tiny.en` or `small.en` for speed
- **Bug fix:** Existing `voice_pipeline.py` passes `model_name=` to `SpeechToText.__init__()`, but the constructor expects `model_size=`. This will be corrected during implementation.

### TTS: Three switchable engines

All engines implement a common `BaseTTS` abstract base class:

```python
class BaseTTS(ABC):
    """Base class for all TTS engines."""

    @abstractmethod
    def load(self, device: str = "cpu") -> None:
        """Load model onto device. Called once at startup."""

    @abstractmethod
    def synthesize(self, text: str) -> bytes:
        """Synthesize text to raw PCM audio (16-bit, 16kHz, mono)."""

    @abstractmethod
    def synthesize_stream(self, text: str) -> Iterator[bytes]:
        """Streaming synthesis — yields audio chunks as they're generated."""

    @abstractmethod
    def unload(self) -> None:
        """Release model resources. Called on shutdown or engine swap."""

    @property
    @abstractmethod
    def supported_languages(self) -> list[str]:
        """ISO 639-1 codes this engine supports."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Engine identifier (e.g., 'piper', 'kokoro', 'melo')."""
```

| Engine | Role | Languages | Latency | When used |
|--------|------|-----------|---------|-----------|
| **Piper** (existing) | Fast mode | English + limited | ~100ms | Short replies, push-to-talk |
| **Kokoro** (new) | Quality mode | 8 languages (EN, FR, ES, +5) | ~400ms | Longer responses, conversational |
| **MeloTTS** (new) | Multilingual mode | Broad coverage including Indic | ~300ms | Tamil, Telugu, Malayalam, auto-detected |

**Auto-selection logic (default):**
1. Response < 20 words → Piper (fast)
2. Detected language not English → MeloTTS (multilingual)
3. Otherwise → Kokoro (quality)

Manual override via config or voice command ("use quality voice"). Config key: `voice.tts_mode: auto | fast | quality | multilingual`

### Audio I/O: Standardized on sounddevice

All audio I/O uses `sounddevice` exclusively. The existing `audio_io.py` uses `pyaudio` for recording and `sounddevice` for playback — this is unified to `sounddevice` only. `pyaudio` is removed from dependencies.

- `AudioInThread`: reads from sounddevice `RawInputStream` (16kHz, mono, int16, 512-sample chunks), pushes to `vad_queue`
- `AudioOutThread`: reads from `playback_queue`, writes to `RawOutputStream`
- Dither strategy: low-level noise when queue empty to keep audio device responsive
- Barge-in: VAD detects speech during SPEAKING → `should_listen` event set → AudioOutThread stops, TTS queue flushed, pipeline returns to LISTENING
- **Existing `AudioIO` class is replaced** by `AudioInThread` + `AudioOutThread`. The existing `AudioRecorder` import in `voice_pipeline.py` (which references a non-existent class) is also fixed.

### Wake Word Strategy

Two-tier approach using existing modules:

1. **Primary: `openwakeword` (audio-level detection)** — The existing `WakeWordEngine` class uses the openwakeword neural model to detect "hey homie" directly from audio, without running full STT. This is efficient and runs on CPU.
2. **Fallback: text-based matching** — If openwakeword is unavailable, fall back to the existing `WakeWordDetector` which checks STT transcript output for the wake phrase. This is less efficient (requires running STT on all audio) but works without the openwakeword dependency.

In hybrid mode, openwakeword runs continuously on the audio stream. When it detects the wake phrase, the pipeline transitions from IDLE → LISTENING without needing STT.

## VoiceManager Interface

`VoiceManager` is the central orchestrator. It owns all voice components, manages the state machine, and bridges voice I/O to the daemon's brain.

```python
class VoiceManager:
    """Orchestrates voice modes, component lifecycle, and brain integration."""

    def __init__(
        self,
        config: VoiceConfig,
        on_query: Callable[[str], Iterator[str]],  # streaming brain callback
        on_state_change: Callable[[PipelineState], None],  # overlay updates
    ) -> None:
        """
        Args:
            config: Voice configuration from homie.config.yaml
            on_query: Streaming callback — receives transcribed text, yields
                      response tokens. This is HomieDaemon._on_user_query_stream.
            on_state_change: Called when pipeline state changes (for overlay UI).
        """

    # --- Lifecycle ---
    def start(self) -> None:
        """Probe available components, start audio threads, enter IDLE state."""

    def stop(self) -> None:
        """Graceful shutdown: drain queues, send sentinels, join threads."""

    # --- Mode control ---
    def set_mode(self, mode: str) -> None:
        """Switch mode: 'hybrid', 'wake_word', 'push_to_talk', 'conversational'."""

    def enter_conversational(self) -> None:
        """Enter conversational mode (from CLI `homie voice`)."""

    def exit_conversational(self) -> None:
        """Exit conversational mode (from exit phrase or timeout)."""

    # --- Hotkey integration ---
    def on_hotkey(self) -> None:
        """Called by HotkeyListener on ctrl+8. Behavior depends on current state."""

    # --- Status ---
    @property
    def state(self) -> PipelineState:
        """Current pipeline state."""

    @property
    def available_engines(self) -> dict[str, bool]:
        """Map of component name → availability (e.g., {'silero_vad': True, 'kokoro': False})."""

    def status_report(self) -> str:
        """Human-readable status for `homie voice status`."""
```

**Streaming callback integration:** `on_query` accepts a `Callable[[str], Iterator[str]]` — it receives the transcribed text and returns a generator of response tokens. `VoiceManager` consumes this generator, simultaneously:
1. Feeding tokens to the overlay (for live text display)
2. Buffering sentences and feeding them to the TTS engine (sentence-level TTS for natural pacing)

This replaces the existing `VoicePipeline.QueryCallback = Callable[[str], str]` (synchronous) with a streaming protocol.

## Daemon & Overlay Integration

### Daemon changes (minimal)

```python
# In HomieDaemon.__init__():
if self._config.voice.enabled:
    self._voice_manager = VoiceManager(
        config=self._config.voice,
        on_query=self._on_user_query_stream,   # existing brain callback
        on_state_change=self._on_voice_state,   # for overlay updates
    )

# In HomieDaemon.start():
if self._voice_manager:
    self._voice_manager.start()

# In HomieDaemon.stop():
if self._voice_manager:
    self._voice_manager.stop()
```

The brain receives text and returns streamed tokens — it doesn't know input came from voice.

### Voice-aware prompting

When input comes from voice, a lightweight hint is injected into the system prompt:

```
User is speaking via voice. Keep responses concise and conversational.
Avoid markdown, code blocks, or visual formatting — the response will be read aloud.
```

Passed as metadata alongside the query. Text queries are unaffected.

### Overlay changes

Voice-first overlay with live transcript:

```
┌─────────────────────────────────┐
│  Listening...                   │  ← state indicator
│                                 │
│  You: "What's the weather like  │  ← live STT transcript
│        in Chennai?"             │
│                                 │
│  Homie: "It's currently 32°C   │  ← streamed response text
│   and humid in Chennai..."      │
│                                 │
│  [Type instead]    [End voice]  │  ← fallback controls
└─────────────────────────────────┘
```

- State indicator follows pipeline state
- Live transcript updates as STT produces results
- Response text streams as brain generates tokens (simultaneous with TTS)
- "Type instead" pauses voice pipeline, switches to text input
- "End voice" stops session gracefully

## Configuration

### VoiceConfig migration

The existing `VoiceConfig` Pydantic model has 6 fields: `enabled`, `wake_word`, `stt_model`, `tts_voice`, `mode`, `hotkey`. These are migrated as follows:

| Old field | New field(s) | Migration |
|---|---|---|
| `enabled` | `enabled` | Unchanged |
| `wake_word` | `wake_word` | Unchanged |
| `stt_model` | `stt_model_quality` | Renamed. Old field accepted as alias for backward compat. |
| `tts_voice` | `tts_voice_quality` | Renamed. Old field accepted as alias for backward compat. |
| `mode` | `mode` | New values added: `hybrid`. Old values (`text_only`, `audio`) mapped to `push_to_talk` and `hybrid`. |
| `hotkey` | `hotkey` | Default changes from `alt+8` to `ctrl+8` |

Pydantic `model_validator` handles backward compatibility — old config files continue to work with deprecation warnings logged.

### homie.config.yaml additions

```yaml
voice:
  enabled: false
  hotkey: ctrl+8
  wake_word: "hey homie"
  mode: hybrid                      # hybrid | wake_word | push_to_talk | conversational

  stt_engine: faster-whisper
  stt_model_fast: tiny.en
  stt_model_quality: medium
  stt_language: auto                # auto | en | ta | te | ml | fr | es

  tts_mode: auto                    # auto | fast | quality | multilingual
  tts_voice_fast: piper
  tts_voice_quality: kokoro
  tts_voice_multilingual: melo

  vad_engine: silero                # silero | webrtcvad | energy
  vad_threshold: 0.5
  vad_silence_ms: 300

  barge_in: true
  conversation_timeout: 120
  max_exit_prompts: 3               # auto-exit after N unanswered confirmations
  exit_phrases:
    - "goodbye"
    - "stop"
    - "that's all"

  device: auto                      # auto | cuda | cpu
  audio_sample_rate: 16000
  audio_chunk_size: 512
```

### CLI additions

```bash
homie voice                         # conversational session
homie voice --mode push-to-talk     # override mode
homie voice --tts quality           # override TTS
homie voice --lang en               # force language
homie voice status                  # show component status
homie voice enable / disable        # toggle voice
```

### New dependencies (pyproject.toml)

```toml
voice = [
    "faster-whisper>=1.0",
    "openwakeword>=0.6",
    "piper-tts>=1.2",
    "sounddevice>=0.4",
    "torch>=2.0",
    "kokoro>=0.9",
    "melo-tts>=0.1",
]
```

**Dependency notes:**
- `pyaudio` removed — all audio I/O standardized on `sounddevice`
- `torchaudio` removed — not needed. Silero VAD uses `torch.hub.load()`, TTS engines handle their own audio processing.
- `torch>=2.0` is required for Silero VAD. CPU-only torch suffices for VAD (2MB model). CUDA torch is needed only if `voice.device: cuda` for TTS acceleration. Install guidance: `pip install torch --index-url https://download.pytorch.org/whl/cu121` for CUDA, or `pip install torch --index-url https://download.pytorch.org/whl/cpu` for CPU-only (~200MB vs ~2GB).

## File Structure

### New and modified files

```
src/homie_core/voice/
├── __init__.py                    # exports VoiceManager, BaseTTS, PipelineState
├── audio_io.py                    # REWRITTEN: AudioInThread, AudioOutThread (sounddevice only)
├── stt.py                         # MODIFIED: fix model_size param, add language detection, model hot-switching
├── tts.py                         # MODIFIED: extract BaseTTS ABC, rename existing class to PiperTTS
├── tts_kokoro.py                  # NEW: KokoroTTS(BaseTTS)
├── tts_melo.py                    # NEW: MeloTTS(BaseTTS)
├── tts_selector.py                # NEW: TTSSelector — auto-selects engine based on context
├── vad.py                         # MODIFIED: keep VoiceActivityDetector + VAD (webrtcvad), add SileroVAD adapter
├── vad_silero.py                  # NEW: SileroVAD implementation
├── wakeword.py                    # UNCHANGED
├── voice_pipeline.py              # REWRITTEN: queue-based threading, barge-in, fix AudioRecorder import
├── voice_manager.py               # NEW: VoiceManager — mode orchestration, state machine, lifecycle
├── voice_prompts.py               # NEW: voice-aware prompt hints for brain
└── base_handler.py                # NEW: BaseHandler with queue/thread pattern (from HF repo)

src/homie_app/
├── cli.py                         # MODIFIED: add `homie voice` command group
├── daemon.py                      # MODIFIED: instantiate VoiceManager, wire callbacks
├── overlay.py                     # MODIFIED: add voice mode panel with live transcript
├── hotkey.py                      # MODIFIED: add ctrl+8 to _HOTKEY_MAP, update default, mode-aware behavior

homie.config.yaml                  # MODIFIED: expanded voice section
pyproject.toml                     # MODIFIED: updated voice dependencies
src/homie_core/config.py           # MODIFIED: expand VoiceConfig with new fields + migration validator
```

### Untouched

- `src/homie_core/brain/` — zero changes
- `src/homie_core/memory/` — unchanged
- `src/homie_core/intelligence/` — unchanged
- `src/homie_core/behavioral/` — unchanged
- All plugins, security, vault, RAG — unchanged
- All existing tests continue to pass

## Error Handling & Degradation

### Graceful degradation chain

```
Silero VAD unavailable (no torch)  → webrtcvad fallback → energy-based VAD fallback
Kokoro unavailable                 → Piper fallback
MeloTTS unavailable                → Piper fallback
faster-whisper fails to load       → voice disabled, text-only mode
Audio device not found             → voice disabled, text-only mode
openwakeword unavailable           → text-based wake word detection (STT required)
```

`VoiceManager` probes each component at startup and builds an availability map.

### Performance safeguards

| Concern | Mitigation |
|---|---|
| GPU memory contention | STT on CPU by default. TTS uses GPU only during SPEAKING, releases after. LLM has priority. |
| Audio latency spikes | Queue depth monitoring per queue. `vad_queue`: 50 items (1.6s audio). `tts_queue`: 10 items (10 sentences). `playback_queue`: 100 items (3.2s audio). Drop oldest on overflow with warning. |
| Barge-in race condition | `should_listen` is `threading.Event` (atomic). TTS thread checks every chunk (~32ms). |
| Wake word false positives | Primary: openwakeword neural detection (audio-level, no STT needed). Fallback: VAD + STT + text match. |
| Timeout during processing | Timer only runs in LISTENING state, pauses during PROCESSING/SPEAKING. |
| Thread cleanup on crash | `stop_event` + queue sentinel (`b"END"`) pattern. `atexit` handler as safety net. |
| Config hot-reload | Mid-session TTS swap: queue drain → swap engine → resume. |
| Exit confirmation loop | Max 3 unanswered exit prompts, then auto-exit. |

## Testing Strategy

### Unit tests

- **VAD tests:** Feed pre-recorded audio with known speech/silence segments, verify detection accuracy and timing for all three VAD backends (Silero, webrtcvad, energy).
- **TTS tests:** Verify each engine implements `BaseTTS` correctly. Test `synthesize()` returns valid PCM audio. Test `synthesize_stream()` yields chunks. Test `supported_languages` accuracy.
- **TTSSelector tests:** Verify auto-selection logic — short text → Piper, non-English → MeloTTS, default → Kokoro. Verify fallback when engine unavailable.
- **STT tests:** Verify `model_size` parameter fix. Test language detection returns valid codes.
- **VoiceConfig tests:** Verify backward compat — old config with `stt_model` / `tts_voice` loads correctly with deprecation warning.

### Integration tests

- **Pipeline flow:** Mock audio input → verify it flows through VAD → STT → brain callback → TTS → audio output in correct order.
- **Barge-in:** Simulate speech during SPEAKING state, verify TTS stops within 64ms (2 chunks).
- **State machine:** Verify all state transitions for each mode (wake word, push-to-talk, conversational).
- **Conversational exit:** Verify exit phrase detection, timeout behavior, max exit prompts.

### Mock strategy

- Audio devices mocked via `unittest.mock.patch('sounddevice.RawInputStream')` / `RawOutputStream`
- STT mocked to return known transcripts
- TTS mocked to return silence bytes
- Brain callback mocked to return canned responses

## Out of Scope (Future Work)

- WebSocket/network audio transport
- Progressive/streaming STT (partial transcripts during speech)
- Voice cloning / custom voice training
- Multi-speaker detection
- Noise cancellation (DeepFilterNet)
- Additional STT engines (Parakeet, Paraformer)
