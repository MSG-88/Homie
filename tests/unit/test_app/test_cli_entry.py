"""Tests for simplified CLI entry point."""
import pytest
from unittest.mock import patch, MagicMock


def test_no_args_launches_console():
    """Running 'homie' with no args should launch the console."""
    with patch("homie_app.console.console.Console") as MockConsole:
        mock_instance = MockConsole.return_value
        from homie_app.cli import main
        with patch("homie_core.config.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            main([])
        MockConsole.assert_called_once()
        mock_instance.run.assert_called_once()


def test_start_launches_console():
    """Running 'homie start' should launch the console."""
    with patch("homie_app.console.console.Console") as MockConsole:
        mock_instance = MockConsole.return_value
        from homie_app.cli import main
        with patch("homie_core.config.load_config") as mock_load:
            mock_load.return_value = MagicMock()
            main(["start"])
        MockConsole.assert_called_once()
        mock_instance.run.assert_called_once()


def test_version_flag():
    """Running 'homie --version' should print version."""
    from homie_app.cli import main
    with pytest.raises(SystemExit):
        main(["--version"])
