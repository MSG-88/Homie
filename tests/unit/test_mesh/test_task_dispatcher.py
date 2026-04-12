from homie_core.mesh.task_dispatcher import TaskDispatcher
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor


def test_dispatch_local_task():
    d = TaskDispatcher(local_node_id="node-1", executor=MeshTaskExecutor())
    r = d.dispatch(MeshTask(source_node="node-1", target_node="node-1", command="echo local"))
    assert r.state == TaskState.COMPLETED and "local" in r.stdout


def test_dispatch_remote_returns_pending():
    d = TaskDispatcher(local_node_id="node-1", executor=MeshTaskExecutor())
    r = d.dispatch(MeshTask(source_node="node-1", target_node="node-2", command="echo remote"))
    assert r.state in (TaskState.PENDING, TaskState.FAILED)


def test_dispatch_with_remote_handler():
    results = []

    def fake_handler(task):
        task.complete(stdout="remote result", exit_code=0)
        results.append(task)
        return task

    d = TaskDispatcher(local_node_id="node-1", executor=MeshTaskExecutor(), remote_handler=fake_handler)
    r = d.dispatch(MeshTask(source_node="node-1", target_node="node-2", command="echo hi"))
    assert r.state == TaskState.COMPLETED and r.stdout == "remote result"


def test_dispatch_records_history():
    d = TaskDispatcher(local_node_id="node-1", executor=MeshTaskExecutor())
    d.dispatch(MeshTask(source_node="n1", target_node="node-1", command="echo a"))
    d.dispatch(MeshTask(source_node="n1", target_node="node-1", command="echo b"))
    assert len(d.task_history()) == 2


def test_dispatch_all_nodes():
    d = TaskDispatcher(local_node_id="node-1", executor=MeshTaskExecutor())
    results = d.dispatch_all(source_node="node-1", target_nodes=["node-1"], command="echo multi", reason="test")
    assert len(results) == 1 and results[0].state == TaskState.COMPLETED
