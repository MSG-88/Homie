import pytest
from homie_core.config import UserProfileConfig, HomieConfig


class TestUserProfileConfig:
    def test_defaults(self):
        cfg = UserProfileConfig()
        assert cfg.name == "Master"
        assert cfg.language == "en"
        assert cfg.timezone == "auto"
        assert cfg.work_hours_start == "09:00"
        assert cfg.work_hours_end == "18:00"
        assert cfg.work_days == ["mon", "tue", "wed", "thu", "fri"]

    def test_custom_values(self):
        cfg = UserProfileConfig(
            name="Muthu",
            language="ta",
            timezone="Asia/Kolkata",
            work_hours_start="10:00",
            work_hours_end="19:00",
            work_days=["mon", "tue", "wed", "thu"],
        )
        assert cfg.name == "Muthu"
        assert cfg.timezone == "Asia/Kolkata"

    def test_homie_config_includes_user(self):
        cfg = HomieConfig()
        assert hasattr(cfg, "user")
        assert cfg.user.name == "Master"


class TestScreenReaderConfig:
    def test_defaults(self):
        from homie_core.config import ScreenReaderConfig
        cfg = ScreenReaderConfig()
        assert cfg.enabled is False
        assert cfg.level == 1
        assert cfg.poll_interval_t1 == 5
        assert cfg.poll_interval_t2 == 30
        assert cfg.poll_interval_t3 == 60
        assert cfg.event_driven is True
        assert cfg.analysis_engine == "cloud"
        assert cfg.pii_filter is True
        assert "*password*" in cfg.blocklist
        assert "*1Password*" in cfg.blocklist
        assert cfg.dnd is False

    def test_level_range(self):
        from homie_core.config import ScreenReaderConfig
        cfg = ScreenReaderConfig(level=3)
        assert cfg.level == 3

    def test_level_validation(self):
        from homie_core.config import ScreenReaderConfig
        import pytest
        with pytest.raises(Exception):  # Pydantic validation error
            ScreenReaderConfig(level=0)
        with pytest.raises(Exception):
            ScreenReaderConfig(level=4)

    def test_homie_config_includes_screen_reader(self):
        from homie_core.config import HomieConfig
        cfg = HomieConfig()
        assert hasattr(cfg, "screen_reader")
        assert cfg.screen_reader.enabled is False
