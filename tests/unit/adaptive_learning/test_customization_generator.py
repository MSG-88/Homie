# tests/unit/adaptive_learning/test_customization_generator.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.customization.generator import CustomizationGenerator


class TestCustomizationGenerator:
    def _make_generator(self, inference_fn=None, evolver=None, rollback=None):
        return CustomizationGenerator(
            inference_fn=inference_fn or MagicMock(return_value='class MyMiddleware:\n    pass'),
            evolver=evolver or MagicMock(),
            rollback=rollback or MagicMock(),
            project_root="/fake/root",
        )

    def test_analyze_request(self):
        gen = self._make_generator()
        analysis = gen.analyze_request("When I say /standup, show me git and calendar")
        assert "intent" in analysis or isinstance(analysis, str)

    def test_generate_code(self):
        gen = self._make_generator()
        code = gen.generate_code("Create a greeting middleware", analysis="middleware that greets user")
        assert isinstance(code, str)
        assert len(code) > 0

    def test_apply_customization(self):
        evolver = MagicMock(return_value="v-123")
        rollback = MagicMock()
        gen = self._make_generator(evolver=evolver, rollback=rollback)
        version_id = gen.apply("test_custom.py", "class Custom:\n    pass", reason="test")
        evolver.create_module.assert_called_once()

    def test_rejects_locked_path(self):
        evolver = MagicMock()
        evolver.create_module.side_effect = PermissionError("locked")
        gen = self._make_generator(evolver=evolver)
        with pytest.raises(PermissionError):
            gen.apply("security/bad.py", "evil", reason="nope")
