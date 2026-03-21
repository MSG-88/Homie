# tests/unit/adaptive_learning/test_customization_manager.py
import pytest
from unittest.mock import MagicMock
from homie_core.adaptive_learning.customization.manager import CustomizationManager


class TestCustomizationManager:
    def _make_manager(self, storage=None, generator=None):
        storage = storage or MagicMock()
        generator = generator or MagicMock()
        generator.analyze_request.return_value = "create a greeting middleware"
        generator.generate_code.return_value = "class Greeting:\n    pass"
        generator.apply.return_value = "v-123"
        return CustomizationManager(storage=storage, generator=generator)

    def test_create_customization(self):
        mgr = self._make_manager()
        result = mgr.create("Greet me with a joke each morning")
        assert result["status"] == "active"
        assert result["version_id"] == "v-123"

    def test_list_customizations(self):
        storage = MagicMock()
        storage.query_customizations.return_value = [
            {"id": 1, "request_text": "test", "status": "active"}
        ]
        mgr = self._make_manager(storage=storage)
        items = mgr.list_customizations()
        assert len(items) == 1

    def test_disable_customization(self):
        storage = MagicMock()
        mgr = self._make_manager(storage=storage)
        mgr.disable(customization_id=1)
        storage.update_customization_status.assert_called_with(1, "disabled")

    def test_enable_customization(self):
        storage = MagicMock()
        mgr = self._make_manager(storage=storage)
        mgr.enable(customization_id=1)
        storage.update_customization_status.assert_called_with(1, "active")
