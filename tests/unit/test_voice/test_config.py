from homie_core.config import VoiceConfig


def test_voice_config_defaults():
    cfg = VoiceConfig()
    assert cfg.enabled is False
    assert cfg.hotkey == "ctrl+8"
    assert cfg.mode == "hybrid"
    assert cfg.stt_model_fast == "tiny.en"
    assert cfg.stt_model_quality == "medium"
    assert cfg.tts_mode == "auto"
    assert cfg.vad_engine == "silero"
    assert cfg.barge_in is True
    assert cfg.conversation_timeout == 120
    assert cfg.max_exit_prompts == 3
    assert cfg.audio_sample_rate == 16000


def test_voice_config_backward_compat():
    cfg = VoiceConfig(stt_model="base", tts_voice="custom", mode="audio", hotkey="alt+8")
    assert cfg.stt_model_quality == "base"
    assert cfg.tts_voice_quality == "custom"
    assert cfg.mode == "hybrid"
    assert cfg.hotkey == "alt+8"
