"""Integration tests for distributed task execution, tracking, and sync."""
from __future__ import annotations

import pytest

from homie_core.mesh.identity import NodeIdentity
from homie_core.mesh.mesh_manager import MeshManager
from homie_core.mesh.task_model import MeshTask, TaskState
from homie_core.mesh.task_executor import MeshTaskExecutor
from homie_core.mesh.task_dispatcher import TaskDispatcher
from homie_core.mesh.task_tracker import TaskTracker
from homie_core.mesh.sync_protocol import SyncRequest


def test_local_task_execution_and_tracking(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=executor)
    tracker = TaskTracker(mesh_manager=mgr)
    task = MeshTask(source_node=identity.node_id, target_node=identity.node_id,
                    command="echo distributed_task_test", reason="integration test")
    tracker.track_created(task)
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.COMPLETED and "distributed_task_test" in result.stdout
    tracker.track_completed(result)
    events = tracker.get_task_events()
    assert len(events) == 2
    assert events[0].event_type == "task_created" and events[1].event_type == "task_completed"


def test_destructive_task_rejected_and_tracked(tmp_path):
    identity = NodeIdentity.generate()
    mgr = MeshManager(identity=identity, data_dir=tmp_path)
    executor = MeshTaskExecutor()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=executor)
    tracker = TaskTracker(mesh_manager=mgr)
    task = MeshTask(source_node=identity.node_id, target_node=identity.node_id,
                    command="rm -rf /important", reason="bad idea")
    tracker.track_created(task)
    result = dispatcher.dispatch(task)
    assert result.state == TaskState.REJECTED
    tracker.track_rejected(result)
    events = tracker.get_task_events()
    assert len(events) == 2 and events[1].event_type == "task_rejected"


def test_task_events_sync_between_nodes(tmp_path):
    node_a = NodeIdentity.generate()
    node_b = NodeIdentity.generate()
    mgr_a = MeshManager(identity=node_a, data_dir=tmp_path / "a")
    mgr_b = MeshManager(identity=node_b, data_dir=tmp_path / "b")
    tracker = TaskTracker(mesh_manager=mgr_a)
    task = MeshTask(source_node=node_a.node_id, target_node=node_b.node_id, command="echo sync_test")
    tracker.track_created(task)
    req = SyncRequest(node_id=node_b.node_id, last_event_id=None, vector_clock={})
    resp = mgr_a.handle_sync_request(req)
    mgr_b.apply_sync_response(resp)
    events_on_b = mgr_b.events_since(None)
    assert len(events_on_b) == 1 and events_on_b[0].event_type == "task_created"


def test_multi_node_dispatch(tmp_path):
    identity = NodeIdentity.generate()
    dispatcher = TaskDispatcher(local_node_id=identity.node_id, executor=MeshTaskExecutor())
    results = dispatcher.dispatch_all(source_node=identity.node_id, target_nodes=[identity.node_id],
                                      command="echo multi", reason="broadcast")
    assert len(results) == 1 and all(r.state == TaskState.COMPLETED for r in results)
