from unittest.mock import MagicMock
from homie_core.voice.tts_selector import TTSSelector


def _make_engine(name, langs):
    e = MagicMock()
    e.name = name
    e.supported_languages = langs
    e.is_loaded = True
    return e


def test_auto_short_uses_fast():
    selector = TTSSelector(fast=_make_engine("piper", ["en"]),
                           quality=_make_engine("kokoro", ["en"]),
                           multilingual=_make_engine("melo", ["en", "ta"]))
    assert selector.select("Hi there", detected_lang="en").name == "piper"


def test_auto_non_english_uses_multilingual():
    selector = TTSSelector(fast=_make_engine("piper", ["en"]),
                           quality=_make_engine("kokoro", ["en"]),
                           multilingual=_make_engine("melo", ["en", "ta"]))
    assert selector.select("vanakkam", detected_lang="ta").name == "melo"


def test_auto_long_english_uses_quality():
    selector = TTSSelector(fast=_make_engine("piper", ["en"]),
                           quality=_make_engine("kokoro", ["en"]),
                           multilingual=_make_engine("melo", ["en"]))
    assert selector.select(" ".join(["word"] * 25), detected_lang="en").name == "kokoro"


def test_fallback_when_quality_unavailable():
    kokoro = _make_engine("kokoro", ["en"])
    kokoro.is_loaded = False
    selector = TTSSelector(fast=_make_engine("piper", ["en"]), quality=kokoro)
    assert selector.select(" ".join(["word"] * 25), detected_lang="en").name == "piper"


def test_forced_mode():
    selector = TTSSelector(fast=_make_engine("piper", ["en"]),
                           quality=_make_engine("kokoro", ["en"]),
                           multilingual=_make_engine("melo", ["en"]),
                           mode="quality")
    assert selector.select("Hi", detected_lang="en").name == "kokoro"
