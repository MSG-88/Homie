import sys
from homie_core.mesh.task_executor import MeshTaskExecutor
from homie_core.mesh.task_model import MeshTask, TaskState


def test_execute_simple_command():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local", command="echo hello")
    executor.execute(task)
    assert task.state == TaskState.COMPLETED and "hello" in task.stdout and task.exit_code == 0


def test_execute_failing_command():
    executor = MeshTaskExecutor()
    cmd = "exit 1" if sys.platform != "win32" else "cmd /c exit 1"
    task = MeshTask(source_node="hub", target_node="local", command=cmd)
    executor.execute(task)
    assert task.state == TaskState.FAILED and task.exit_code == 1


def test_reject_destructive_command():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local", command="rm -rf /")
    executor.execute(task)
    assert task.state == TaskState.REJECTED and "destructive" in task.error.lower()


def test_dry_run_does_not_execute():
    executor = MeshTaskExecutor()
    task = MeshTask(source_node="hub", target_node="local", command="echo should_not_run", dry_run=True)
    executor.execute(task)
    assert task.state == TaskState.COMPLETED and task.stdout == ""


def test_allowlist_permits_destructive():
    executor = MeshTaskExecutor(allowlist=["rm -rf /tmp/test"])
    task = MeshTask(source_node="hub", target_node="local", command="rm -rf /tmp/test")
    executor.execute(task)
    assert task.state != TaskState.REJECTED


def test_timeout_enforcement():
    executor = MeshTaskExecutor(timeout_seconds=1)
    cmd = "sleep 10" if sys.platform != "win32" else "ping -n 11 127.0.0.1"
    task = MeshTask(source_node="hub", target_node="local", command=cmd)
    executor.execute(task)
    assert task.state == TaskState.FAILED and "timeout" in task.error.lower()
