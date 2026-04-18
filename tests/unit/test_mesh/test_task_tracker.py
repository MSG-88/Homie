"""Tests for TaskTracker — event-based task lifecycle tracking."""
from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_tracker import TaskTracker


def test_track_task_emits_events(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)
    task = MeshTask(source_node=identity.node_id, target_node="remote", command="echo test", reason="testing")
    tracker.track_created(task)
    assert mgr.event_count() == 1
    events = mgr.events_since(None)
    assert events[0].category == "task" and events[0].event_type == "task_created"


def test_track_completed(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)
    task = MeshTask(source_node=identity.node_id, target_node="remote", command="echo ok")
    task.state = TaskState.COMPLETED
    task.stdout = "ok\n"
    task.exit_code = 0
    tracker.track_completed(task)
    assert mgr.events_since(None)[0].event_type == "task_completed"


def test_track_failed(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)
    task = MeshTask(source_node=identity.node_id, target_node="remote", command="bad")
    task.state = TaskState.FAILED
    task.error = "not found"
    task.exit_code = 127
    tracker.track_failed(task)
    assert "not found" in mgr.events_since(None)[0].payload["error"]


def test_get_task_history(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    tracker = TaskTracker(mesh_manager=mgr)
    for i in range(3):
        tracker.track_created(MeshTask(source_node=identity.node_id, target_node="r", command=f"echo {i}"))
    assert len(tracker.get_task_events()) == 3
