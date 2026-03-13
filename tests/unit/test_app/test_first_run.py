"""Tests for first-run wizard detection and inline wizard flow."""
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from pathlib import Path


def test_no_config_triggers_wizard(tmp_path):
    """When config has no model_path and no user_name, wizard should run."""
    from homie_app.console.console import Console

    cfg = MagicMock()
    cfg.user_name = ""
    cfg.llm.model_path = ""
    cfg.storage.path = str(tmp_path)

    with patch.object(Console, "_run_wizard") as mock_wizard:
        with patch.object(Console, "_bootstrap"):
            console = Console(config=cfg, skip_init=True)
            console._config = cfg
            assert console._needs_wizard()
            mock_wizard.return_value = None


def test_complete_config_skips_wizard(tmp_path):
    """When config is complete, wizard should not run."""
    from homie_app.console.console import Console

    cfg = MagicMock()
    cfg.user_name = "Master"
    cfg.llm.model_path = "/some/model.gguf"
    cfg.storage.path = str(tmp_path)

    with patch.object(Console, "_bootstrap"):
        console = Console(config=cfg, skip_init=True)
        console._config = cfg
        assert not console._needs_wizard()


def test_wizard_reloads_config_on_success(tmp_path):
    """After wizard completes, config should be reloaded."""
    from homie_app.console.console import Console

    cfg = MagicMock()
    cfg.user_name = ""
    cfg.llm.model_path = ""

    new_cfg = MagicMock()
    new_cfg.user_name = "NewUser"

    with patch.object(Console, "_bootstrap"):
        console = Console(config=cfg, skip_init=True)
        console._config = cfg
        console._config_path = str(tmp_path / "config.yaml")

        with patch("homie_app.console.console.Console._print"):
            with patch("homie_app.init.run_init"):
                with patch("homie_core.config.load_config", return_value=new_cfg):
                    console._run_wizard()
                    assert console._config == new_cfg
                    assert console._user_name == "NewUser"
