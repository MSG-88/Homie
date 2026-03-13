import time
from unittest.mock import MagicMock
from homie_core.notifications.router import NotificationRouter, Notification
from homie_core.config import NotificationConfig


class TestTokenExpiryCheck:
    def test_warns_when_linkedin_expires_soon(self):
        from homie_app.daemon import _check_token_expiry
        vault = MagicMock()
        # Token expires in 5 days (within 7-day warning window)
        vault.get_credential.return_value = MagicMock(
            expires_at=time.time() + (5 * 86400),
            provider="linkedin",
        )
        router = MagicMock()
        _check_token_expiry(vault, router, "linkedin")
        router.route.assert_called_once()
        notification = router.route.call_args[0][0]
        assert notification.category == "system_alerts"
        assert "linkedin" in notification.title.lower() or "linkedin" in notification.body.lower()

    def test_no_warning_when_token_fresh(self):
        from homie_app.daemon import _check_token_expiry
        vault = MagicMock()
        # Token expires in 30 days (outside warning window)
        vault.get_credential.return_value = MagicMock(
            expires_at=time.time() + (30 * 86400),
            provider="linkedin",
        )
        router = MagicMock()
        _check_token_expiry(vault, router, "linkedin")
        router.route.assert_not_called()
