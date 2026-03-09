from homie_core.plugins.permissions import PermissionManager


def test_grant_and_check():
    pm = PermissionManager()
    pm.grant("email", ["read_email", "send_email"])
    assert pm.has_permission("email", "read_email") is True
    assert pm.has_permission("email", "delete_email") is False


def test_revoke():
    pm = PermissionManager()
    pm.grant("email", ["read_email", "send_email"])
    pm.revoke("email", "send_email")
    assert pm.has_permission("email", "send_email") is False
    assert pm.has_permission("email", "read_email") is True


def test_revoke_all():
    pm = PermissionManager()
    pm.grant("email", ["read_email", "send_email"])
    pm.revoke_all("email")
    assert pm.get_granted("email") == []


def test_check_all():
    pm = PermissionManager()
    pm.grant("email", ["read_email", "send_email"])
    assert pm.check_all("email", ["read_email", "send_email"]) is True
    assert pm.check_all("email", ["read_email", "delete_email"]) is False
