from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import yaml
from pydantic import BaseModel, Field


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
    wake_word: str = "hey homie"
    stt_model: str = "large-v3"
    tts_voice: str = "default"
    mode: str = "text_only"
    hotkey: str = "F9"


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


class HomieConfig(BaseModel):
    llm: LLMConfig = Field(default_factory=LLMConfig)
    voice: VoiceConfig = Field(default_factory=VoiceConfig)
    storage: StorageConfig = Field(default_factory=StorageConfig)
    privacy: PrivacyConfig = Field(default_factory=PrivacyConfig)
    plugins: PluginConfig = Field(default_factory=PluginConfig)
    user_name: str = ""


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
    }
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
