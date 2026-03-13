import sqlite3
from homie_core.messaging.phone_link_reader import PhoneLinkReader


class TestPhoneLinkReader:
    def test_detect_not_installed(self, tmp_path):
        reader = PhoneLinkReader(base_path=str(tmp_path / "nonexistent"))
        assert reader.is_available() is False

    def test_detect_installed(self, tmp_path):
        db_dir = tmp_path / "Indexed" / "FAKE-GUID" / "System" / "Database"
        db_dir.mkdir(parents=True)
        db_path = db_dir / "phone.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("CREATE TABLE messages (id INTEGER, body TEXT, timestamp TEXT)")
        conn.execute("INSERT INTO messages VALUES (1, 'Hello', '2026-03-13')")
        conn.commit()
        conn.close()
        reader = PhoneLinkReader(base_path=str(tmp_path))
        assert reader.is_available() is True

    def test_discover_guids(self, tmp_path):
        (tmp_path / "Indexed" / "GUID-1" / "System" / "Database").mkdir(parents=True)
        (tmp_path / "Indexed" / "GUID-2" / "System" / "Database").mkdir(parents=True)
        reader = PhoneLinkReader(base_path=str(tmp_path))
        guids = reader.discover_devices()
        assert len(guids) == 2

    def test_read_messages_graceful_failure(self):
        reader = PhoneLinkReader(base_path="/nonexistent")
        messages = reader.read_messages()
        assert messages == []
