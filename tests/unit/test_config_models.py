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


class TestServiceConfig:
    def test_defaults(self):
        from homie_core.config import ServiceConfig
        cfg = ServiceConfig()
        assert cfg.mode == "on_demand"
        assert cfg.start_on_login is False
        assert cfg.restart_on_failure is True
        assert cfg.max_retries == 3


class TestNotificationConfig:
    def test_defaults(self):
        from homie_core.config import NotificationConfig
        cfg = NotificationConfig()
        assert cfg.enabled is True
        assert cfg.categories["task_reminders"] is True
        assert cfg.categories["email_digest"] is True
        assert cfg.categories["social_mentions"] is True
        assert cfg.categories["context_suggestions"] is True
        assert cfg.categories["system_alerts"] is True
        assert cfg.dnd_schedule_enabled is False
        assert cfg.dnd_schedule_start == "22:00"
        assert cfg.dnd_schedule_end == "07:00"

    def test_categories_not_shared_between_instances(self):
        from homie_core.config import NotificationConfig
        a = NotificationConfig()
        b = NotificationConfig()
        a.categories["email_digest"] = False
        assert b.categories["email_digest"] is True  # Must not be shared


class TestConnectionsConfig:
    def test_defaults(self):
        from homie_core.config import ConnectionsConfig
        cfg = ConnectionsConfig()
        assert cfg.gmail.connected is False
        assert cfg.twitter.connected is False
        assert cfg.telegram.connected is False
        assert cfg.whatsapp.connected is False
        assert cfg.whatsapp.experimental is True
        assert cfg.phone_link.connected is False
        assert cfg.phone_link.read_only is True
        assert cfg.blog.feed_url == ""

    def test_homie_config_includes_all(self):
        from homie_core.config import HomieConfig
        cfg = HomieConfig()
        assert hasattr(cfg, "service")
        assert hasattr(cfg, "notifications")
        assert hasattr(cfg, "connections")


class TestConfigBackwardsCompat:
    def test_old_config_loads_without_new_sections(self, tmp_path, monkeypatch):
        """Old config files without new sections should load with defaults."""
        import yaml
        # Isolate from any HF_KEY / HOMIE_* environment overrides
        monkeypatch.delenv("HF_KEY", raising=False)
        monkeypatch.delenv("HOMIE_LLM_BACKEND", raising=False)
        monkeypatch.delenv("HOMIE_LLM_MODEL_PATH", raising=False)
        old_config = {
            "llm": {"backend": "gguf", "model_path": "/some/model.gguf"},
            "voice": {"enabled": False},
            "storage": {"path": str(tmp_path / "homie_storage")},
            "privacy": {"data_retention_days": 30, "max_storage_mb": 512,
                        "observers": {"work": True}},
            "plugins": {"enabled": []},
            "user_name": "TestUser",
        }
        config_file = tmp_path / "homie.config.yaml"
        config_file.write_text(yaml.dump(old_config))

        from homie_core.config import load_config
        cfg = load_config(str(config_file))
        # Old fields preserved
        assert cfg.user_name == "TestUser"
        assert cfg.llm.backend == "gguf"
        # New fields get defaults
        assert cfg.user.name == "Master"
        assert cfg.screen_reader.enabled is False
        assert cfg.service.mode == "on_demand"
        assert cfg.notifications.enabled is True
        assert cfg.connections.gmail.connected is False

    def test_new_config_round_trips(self, tmp_path):
        """Config with new sections saves and reloads correctly."""
        import yaml
        from homie_core.config import HomieConfig, load_config
        cfg = HomieConfig()
        cfg.user.name = "Muthu"
        cfg.screen_reader.enabled = True
        cfg.screen_reader.level = 2

        config_file = tmp_path / "homie.config.yaml"
        data = cfg.model_dump()
        # Override storage path to tmp_path so load_config can create it
        data["storage"]["path"] = str(tmp_path / "homie_storage")
        config_file.write_text(yaml.dump(data, default_flow_style=False))

        loaded = load_config(str(config_file))
        assert loaded.user.name == "Muthu"
        assert loaded.screen_reader.enabled is True
        assert loaded.screen_reader.level == 2
