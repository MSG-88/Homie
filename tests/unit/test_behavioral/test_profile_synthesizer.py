import pytest
from homie_core.behavioral.base import BaseObserver
from homie_core.behavioral.profile_synthesizer import ProfileSynthesizer
from homie_core.memory.semantic import SemanticMemory
from homie_core.storage.database import Database


class MockObserver(BaseObserver):
    def tick(self):
        return {}
    def get_profile_updates(self):
        return {"favorite": "Python"}


@pytest.fixture
def synthesizer(tmp_path):
    db = Database(tmp_path / "test.db")
    db.initialize()
    sm = SemanticMemory(db=db)
    return ProfileSynthesizer(semantic_memory=sm, observers=[MockObserver(name="work")])


def test_synthesize_updates_profiles(synthesizer):
    result = synthesizer.synthesize()
    assert "work" in result
    assert result["work"]["favorite"] == "Python"


def test_get_full_profile(synthesizer):
    synthesizer.synthesize()
    profile = synthesizer.get_full_profile()
    assert "work" in profile
