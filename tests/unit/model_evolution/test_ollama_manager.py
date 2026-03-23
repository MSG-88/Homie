import pytest
from unittest.mock import MagicMock, patch
from homie_core.model_evolution.ollama_manager import OllamaManager


class TestOllamaManager:
    def test_pull_success(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="success")
            assert mgr.pull("lfm2") is True
            mock_run.assert_called_once()

    def test_pull_failure(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=1, stderr="error")
            assert mgr.pull("nonexistent") is False

    def test_create_with_modelfile(self, tmp_path):
        modelfile = tmp_path / "Modelfile"
        modelfile.write_text("FROM lfm2\nSYSTEM You are Homie.")
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="created")
            assert mgr.create("MSG-88/Homie", modelfile) is True

    def test_push_with_api_key(self):
        mgr = OllamaManager(api_key="test-key-123")
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="pushed")
            assert mgr.push("MSG-88/Homie") is True
            # Verify OLLAMA_API_KEY was in the environment
            call_kwargs = mock_run.call_args[1]
            assert "env" in call_kwargs
            assert call_kwargs["env"]["OLLAMA_API_KEY"] == "test-key-123"

    def test_list_models(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="NAME\nlfm2:latest\nglm-4.7-flash:latest\n")
            models = mgr.list_models()
            assert len(models) >= 2

    def test_show_model(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0, stdout="Model: lfm2\nParameters: 7B")
            info = mgr.show("lfm2")
            assert isinstance(info, str)
            assert "lfm2" in info

    def test_remove_model(self):
        mgr = OllamaManager()
        with patch("homie_core.model_evolution.ollama_manager.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            assert mgr.remove("old-model") is True
