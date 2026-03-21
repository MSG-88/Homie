from __future__ import annotations

import os
from enum import Enum
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field, model_validator


class ModelTier(str, Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class ModelProfile(BaseModel):
    name: str
    tier: ModelTier
    context_length: int = 4096
    supports_tools: bool = True
    location: str = "local"  # "local" | "lan" | "qubrid"
    priority: int = 0


class LLMConfig(BaseModel):
    model_path: str = str(Path.home() / ".lmstudio" / "models" / "lmstudio-community" / "Qwen3.5-35B-A3B-GGUF" / "Qwen3.5-35B-A3B-Q4_K_M.gguf")
    backend: str = "gguf"
    context_length: int = 65536
    temperature: float = 0.7
    max_tokens: int = 2048
    gpu_layers: int = -1
    repo_id: str = "lmstudio-community/Qwen3.5-35B-A3B-GGUF"
    api_key: str = ""
    api_base_url: str = ""


class VoiceConfig(BaseModel):
    enabled: bool = False
    hotkey: str = "ctrl+8"
    wake_word: str = "hey homie"
    mode: str = "hybrid"

    stt_engine: str = "faster-whisper"
    stt_model_fast: str = "tiny.en"
    stt_model_quality: str = "medium"
    stt_language: str = "auto"

    tts_mode: str = "auto"
    tts_voice_fast: str = "piper"
    tts_voice_quality: str = "kokoro"
    tts_voice_multilingual: str = "melo"

    vad_engine: str = "silero"
    vad_threshold: float = 0.5
    vad_silence_ms: int = 300

    barge_in: bool = True
    conversation_timeout: int = 120
    max_exit_prompts: int = 3
    exit_phrases: list[str] = ["goodbye", "stop", "that's all"]

    device: str = "auto"
    audio_sample_rate: int = 16000
    audio_chunk_size: int = 512

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_fields(cls, data: dict) -> dict:
        if isinstance(data, dict):
            if "stt_model" in data and "stt_model_quality" not in data:
                data["stt_model_quality"] = data.pop("stt_model")
            elif "stt_model" in data:
                data.pop("stt_model")
            if "tts_voice" in data and "tts_voice_quality" not in data:
                data["tts_voice_quality"] = data.pop("tts_voice")
            elif "tts_voice" in data:
                data.pop("tts_voice")
            mode_map = {"text_only": "push_to_talk", "audio": "hybrid"}
            if "mode" in data and data["mode"] in mode_map:
                data["mode"] = mode_map[data["mode"]]
        return data


class StorageConfig(BaseModel):
    path: str = Field(default_factory=lambda: str(Path.home() / ".homie"))
    db_name: str = "homie.db"
    chroma_dir: str = "chroma"
    models_dir: str = "models"
    backup_dir: str = "backups"
    log_dir: str = "logs"


class PrivacyConfig(BaseModel):
    data_retention_days: int = 30
    max_storage_mb: int = 512
    screen_reader_consent: bool = False
    observers: dict[str, bool] = Field(default_factory=lambda: {
        "media": False,
        "browsing": False,
        "work": True,
        "social": False,
        "routine": True,
        "emotional": False,
    })


class PluginConfig(BaseModel):
    enabled: list[str] = Field(default_factory=list)
    plugin_dirs: list[str] = Field(default_factory=list)


class UserProfileConfig(BaseModel):
    name: str = "Master"
    language: str = "en"
    timezone: str = "auto"
    work_hours_start: str = "09:00"
    work_hours_end: str = "18:00"
    work_days: list[str] = ["mon", "tue", "wed", "thu", "fri"]


class ScreenReaderConfig(BaseModel):
    enabled: bool = False
    level: int = Field(default=1, ge=1, le=3)  # 1=window titles, 2=+OCR, 3=+visual
    poll_interval_t1: int = 5
    poll_interval_t2: int = 30
    poll_interval_t3: int = 60
    event_driven: bool = True
    analysis_engine: str = "cloud"  # cloud or local
    pii_filter: bool = True
    blocklist: list[str] = [
        "*password*", "*banking*", "*incognito*", "*private*",
        "*1Password*", "*KeePass*", "*LastPass*",
    ]
    dnd: bool = False


class ServiceConfig(BaseModel):
    mode: str = "on_demand"  # on_demand or windows_service
    start_on_login: bool = False
    restart_on_failure: bool = True
    max_retries: int = 3


class NotificationConfig(BaseModel):
    enabled: bool = True
    categories: dict[str, bool] = Field(default_factory=lambda: {
        "task_reminders": True,
        "email_digest": True,
        "email_priority": True,
        "social_mentions": True,
        "context_suggestions": True,
        "system_alerts": True,
        "proactive": True,
    })
    dnd_schedule_enabled: bool = False
    dnd_schedule_start: str = "22:00"
    dnd_schedule_end: str = "07:00"


class ConnectionState(BaseModel):
    connected: bool = False


class WhatsAppConnection(BaseModel):
    connected: bool = False
    experimental: bool = True


class PhoneLinkConnection(BaseModel):
    connected: bool = False
    read_only: bool = True


class BlogConnection(BaseModel):
    connected: bool = False
    feed_url: str = ""


class ConnectionsConfig(BaseModel):
    gmail: ConnectionState = Field(default_factory=ConnectionState)
    twitter: ConnectionState = Field(default_factory=ConnectionState)
    reddit: ConnectionState = Field(default_factory=ConnectionState)
    telegram: ConnectionState = Field(default_factory=ConnectionState)
    slack: ConnectionState = Field(default_factory=ConnectionState)
    facebook: ConnectionState = Field(default_factory=ConnectionState)
    instagram: ConnectionState = Field(default_factory=ConnectionState)
    linkedin: ConnectionState = Field(default_factory=ConnectionState)
    whatsapp: WhatsAppConnection = Field(default_factory=WhatsAppConnection)
    phone_link: PhoneLinkConnection = Field(default_factory=PhoneLinkConnection)
    blog: BlogConnection = Field(default_factory=BlogConnection)


class LocationConfig(BaseModel):
    city: str = ""
    region: str = ""
    country: str = ""
    timezone: str = ""


class QubridConfig(BaseModel):
    enabled: bool = True
    model: str = "Qwen/Qwen3.5-Flash"
    base_url: str = "https://platform.qubrid.com/v1"
    timeout: int = 30


class LANInferenceConfig(BaseModel):
    prefer_desktop: bool = True
    max_latency_ms: int = 500


class InferenceConfig(BaseModel):
    priority: list[str] = ["local", "lan", "qubrid"]
    qubrid: QubridConfig = QubridConfig()
    lan: LANInferenceConfig = LANInferenceConfig()


class NetworkConfig(BaseModel):
    enabled: bool = True
    sync_port: int = 8765
    auto_discover: bool = True
    sync_scope: str = "all"  # "all", "conversations", "manual"


class ContextConfig(BaseModel):
    summarize_trigger_pct: float = 0.85
    summarize_keep_pct: float = 0.10
    manual_compact_pct: float = 0.50
    large_result_threshold: int = 80000
    long_line_threshold: int = 5000
    arg_truncation_threshold: int = 2000


class EmbeddingConfig(BaseModel):
    model_name: str = "BAAI/bge-base-en-v1.5"
    dimensions: int = 768
    batch_size: int = 32


class ImprovementConfig(BaseModel):
    enabled: bool = True
    max_mutations_per_day: int = 10
    monitoring_window: int = 300
    rollback_error_threshold: float = 0.20
    rollback_latency_threshold: float = 0.50


class RecoveryConfig(BaseModel):
    max_tier: int = 4
    preemptive: bool = True
    pattern_threshold: int = 3


class HealthLogConfig(BaseModel):
    retention_days: int = 30
    digest_enabled: bool = True


class SelfHealingConfig(BaseModel):
    enabled: bool = True
    probe_interval: int = 30
    critical_probe_interval: int = 10
    improvement: ImprovementConfig = ImprovementConfig()
    recovery: RecoveryConfig = RecoveryConfig()
    health_log: HealthLogConfig = HealthLogConfig()
    guardian_enabled: bool = True
    core_lock: list[str] = [
        "self_healing/improvement/rollback.py",
        "self_healing/guardian.py",
        "security/",
    ]


class HomieConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    user: UserProfileConfig = Field(default_factory=UserProfileConfig)
    screen_reader: ScreenReaderConfig = Field(default_factory=ScreenReaderConfig)
    service: ServiceConfig = Field(default_factory=ServiceConfig)
    notifications: NotificationConfig = Field(default_factory=NotificationConfig)
    connections: ConnectionsConfig = Field(default_factory=ConnectionsConfig)
    location: Optional[LocationConfig] = None
    user_name: str = ""
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    context: ContextConfig = Field(default_factory=ContextConfig)
    embedding: EmbeddingConfig = Field(default_factory=EmbeddingConfig)
    self_healing: SelfHealingConfig = Field(default_factory=SelfHealingConfig)


def _apply_env_overrides(cfg: HomieConfig) -> HomieConfig:
    env_map = {
        "HOMIE_LLM_BACKEND": ("llm", "backend"),
        "HOMIE_LLM_MODEL_PATH": ("llm", "model_path"),
        "HOMIE_LLM_GPU_LAYERS": ("llm", "gpu_layers"),
        "HOMIE_VOICE_ENABLED": ("voice", "enabled"),
        "HOMIE_STORAGE_PATH": ("storage", "path"),
        "HOMIE_API_KEY": ("llm", "api_key"),
        "HOMIE_API_BASE_URL": ("llm", "api_base_url"),
        "HOMIE_USER_NAME": ("user_name",),
        "HF_KEY": ("llm", "api_key"),
    }

    # Auto-detect HF backend when HF_KEY is present and no explicit config
    hf_key = os.environ.get("HF_KEY", "")
    if hf_key and not cfg.llm.api_key and cfg.llm.backend == "gguf":
        cfg.llm.api_key = hf_key
        cfg.llm.backend = "hf"
        cfg.llm.model_path = "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
    for env_var, path in env_map.items():
        val = os.environ.get(env_var)
        if val is None:
            continue
        obj = cfg
        for key in path[:-1]:
            obj = getattr(obj, key)
        field = path[-1]
        field_type = type(getattr(obj, field))
        if field_type is bool:
            val = val.lower() in ("true", "1", "yes")
        setattr(obj, field, field_type(val))
    return cfg


def load_config(path: Optional[Path | str] = None) -> HomieConfig:
    data = {}
    if path is not None:
        path = Path(path)
        if path.exists():
            data = yaml.safe_load(path.read_text()) or {}
    elif Path("homie.config.yaml").exists():
        data = yaml.safe_load(Path("homie.config.yaml").read_text()) or {}

    cfg = HomieConfig(**data)
    cfg = _apply_env_overrides(cfg)

    storage_path = Path(cfg.storage.path)
    storage_path.mkdir(parents=True, exist_ok=True)

    return cfg
