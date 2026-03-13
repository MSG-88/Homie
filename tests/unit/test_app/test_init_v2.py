from unittest.mock import patch, MagicMock, call
from homie_core.config import HomieConfig


class TestInitWizardV2:
    @patch("homie_app.init.input", side_effect=["Muthu"])
    @patch("homie_app.init._ask_choice", return_value=0)
    def test_step5_user_profile(self, mock_choice, mock_input):
        from homie_app.init import _step_user_profile
        cfg = HomieConfig()
        _step_user_profile(cfg)
        assert cfg.user.name == "Muthu"

    @patch("homie_app.init._ask_choice", return_value=0)  # level 1
    def test_step6_screen_reader_consent(self, mock_choice):
        from homie_app.init import _step_screen_reader
        cfg = HomieConfig()
        _step_screen_reader(cfg)
        assert cfg.screen_reader.enabled is True
        assert cfg.screen_reader.level == 1

    @patch("homie_app.init._ask_choice", return_value=3)  # off
    def test_step6_screen_reader_off(self, mock_choice):
        from homie_app.init import _step_screen_reader
        cfg = HomieConfig()
        _step_screen_reader(cfg)
        assert cfg.screen_reader.enabled is False

    @patch("homie_app.init.input", side_effect=["n"])  # skip gmail
    def test_step7_skip_email(self, mock_input):
        from homie_app.init import _step_email
        cfg = HomieConfig()
        _step_email(cfg)
        assert cfg.connections.gmail.connected is False

    @patch("homie_app.init._ask_choice", return_value=0)  # on_demand
    def test_step12_service_mode(self, mock_choice):
        from homie_app.init import _step_service_mode
        cfg = HomieConfig()
        _step_service_mode(cfg)
        assert cfg.service.mode == "on_demand"

    @patch("homie_app.init._ask_choice", return_value=1)  # windows_service
    def test_step12_windows_service(self, mock_choice):
        from homie_app.init import _step_service_mode
        cfg = HomieConfig()
        _step_service_mode(cfg)
        assert cfg.service.mode == "windows_service"

    def test_existing_config_detection(self, tmp_path):
        import yaml
        config_file = tmp_path / "homie.config.yaml"
        config_file.write_text(yaml.dump({"user_name": "OldUser"}))
        from homie_app.init import _detect_existing_config
        exists, data = _detect_existing_config(str(config_file))
        assert exists is True
        assert data["user_name"] == "OldUser"
