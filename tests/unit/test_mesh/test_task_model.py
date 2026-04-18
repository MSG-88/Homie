from homie_core.mesh.task_model import MeshTask, TaskState

def test_task_states():
    assert TaskState.PENDING == "pending" and TaskState.RUNNING == "running"
    assert TaskState.COMPLETED == "completed" and TaskState.FAILED == "failed" and TaskState.REJECTED == "rejected"

def test_task_creation():
    t = MeshTask(source_node="desktop", target_node="laptop", command="ls ~/Downloads", reason="organize files")
    assert t.task_id and t.source_node == "desktop" and t.state == TaskState.PENDING

def test_task_to_dict():
    d = MeshTask(source_node="a", target_node="b", command="echo hi").to_dict()
    assert d["source_node"] == "a" and d["command"] == "echo hi" and d["state"] == "pending"

def test_task_from_dict():
    t = MeshTask(source_node="a", target_node="b", command="pwd", reason="test")
    r = MeshTask.from_dict(t.to_dict())
    assert r.task_id == t.task_id and r.command == "pwd" and r.reason == "test"

def test_task_complete():
    t = MeshTask(source_node="a", target_node="b", command="echo ok")
    t.complete(stdout="ok\n", exit_code=0)
    assert t.state == TaskState.COMPLETED and t.stdout == "ok\n" and t.exit_code == 0

def test_task_fail():
    t = MeshTask(source_node="a", target_node="b", command="bad")
    t.fail(error="command not found", exit_code=127)
    assert t.state == TaskState.FAILED and t.error == "command not found"

def test_task_reject():
    t = MeshTask(source_node="a", target_node="b", command="rm -rf /")
    t.reject(reason="blocked by safety")
    assert t.state == TaskState.REJECTED and t.error == "blocked by safety"

def test_task_is_destructive():
    assert MeshTask(source_node="a", target_node="b", command="rm -rf /tmp").is_destructive is True
    assert MeshTask(source_node="a", target_node="b", command="ls /tmp").is_destructive is False
    assert MeshTask(source_node="a", target_node="b", command="DROP TABLE users").is_destructive is True
    assert MeshTask(source_node="a", target_node="b", command="format C:").is_destructive is True
