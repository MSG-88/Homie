from homie_core.mesh.auth import Role, MeshUser, AuthStore, hash_api_key, generate_api_key

def test_roles_ordering():
    assert Role.VIEWER < Role.USER < Role.ADMIN

def test_user_permissions_viewer():
    u = MeshUser(user_id="1", username="viewer", role=Role.VIEWER, node_id="n1")
    assert u.can_view_dashboard() and not u.can_execute_tasks() and not u.can_manage_nodes()

def test_user_permissions_user():
    u = MeshUser(user_id="1", username="user", role=Role.USER, node_id="n1")
    assert u.can_view_dashboard() and u.can_execute_tasks() and not u.can_manage_nodes()

def test_user_permissions_admin():
    u = MeshUser(user_id="1", username="admin", role=Role.ADMIN, node_id="n1")
    assert u.can_view_dashboard() and u.can_execute_tasks() and u.can_manage_nodes() and u.can_manage_users()

def test_create_and_authenticate(tmp_path):
    store = AuthStore(tmp_path / "auth.db"); store.initialize()
    user, api_key = store.create_user("muthu", Role.ADMIN, "desktop")
    assert user.username == "muthu" and user.role == Role.ADMIN
    authed = store.authenticate(api_key)
    assert authed is not None and authed.user_id == user.user_id

def test_authenticate_bad_key(tmp_path):
    store = AuthStore(tmp_path / "auth.db"); store.initialize()
    store.create_user("test", Role.USER)
    assert store.authenticate("bad_key") is None

def test_deactivate_user(tmp_path):
    store = AuthStore(tmp_path / "auth.db"); store.initialize()
    user, key = store.create_user("test", Role.USER)
    store.deactivate_user(user.user_id)
    assert store.authenticate(key) is None

def test_list_users(tmp_path):
    store = AuthStore(tmp_path / "auth.db"); store.initialize()
    store.create_user("a", Role.USER); store.create_user("b", Role.ADMIN)
    assert len(store.list_users()) == 2

def test_audit_log(tmp_path):
    store = AuthStore(tmp_path / "auth.db"); store.initialize()
    user, _ = store.create_user("admin", Role.ADMIN)
    store.log_action(user.user_id, "task_dispatch", target="laptop", details={"command": "echo hi"})
    log = store.get_audit_log()
    assert len(log) == 1 and log[0]["action"] == "task_dispatch"

def test_user_to_dict_roundtrip():
    u = MeshUser(user_id="1", username="test", role=Role.ADMIN, node_id="n1")
    r = MeshUser.from_dict(u.to_dict())
    assert r.user_id == "1" and r.role == Role.ADMIN

def test_generate_api_key():
    key = generate_api_key()
    assert key.startswith("homie_") and len(key) > 20
