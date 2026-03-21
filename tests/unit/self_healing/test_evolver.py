# tests/unit/self_healing/test_evolver.py
import pytest
from homie_core.self_healing.improvement.evolver import ArchitectureEvolver
from homie_core.self_healing.improvement.rollback import RollbackManager


class TestArchitectureEvolver:
    def test_create_module(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        path = tmp_path / "src" / "new_module.py"
        version_id = evolver.create_module(
            file_path=path,
            content='"""New module."""\n\ndef new_function():\n    pass\n',
            reason="add caching layer",
        )
        assert path.exists()
        assert "new_function" in path.read_text()
        assert version_id is not None

    def test_create_module_rejects_locked_path(self, tmp_path):
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path, locked_paths=["security/"])
        with pytest.raises(PermissionError):
            evolver.create_module(
                file_path=tmp_path / "security" / "backdoor.py",
                content="evil",
                reason="nope",
            )

    def test_remove_module(self, tmp_path):
        target = tmp_path / "dead_module.py"
        target.write_text("# deprecated")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        version_id = evolver.remove_module(file_path=target, reason="deprecated")
        assert not target.exists()
        # Can rollback
        rm.rollback(version_id)
        assert target.exists()

    def test_split_module_creates_new_files(self, tmp_path):
        original = tmp_path / "big_module.py"
        original.write_text("# Part A\ndef func_a(): pass\n\n# Part B\ndef func_b(): pass\n")
        rm = RollbackManager(snapshot_dir=tmp_path / "snapshots")
        evolver = ArchitectureEvolver(rollback_manager=rm, project_root=tmp_path)
        version_id = evolver.split_module(
            source=original,
            targets={
                tmp_path / "part_a.py": "# Part A\ndef func_a(): pass\n",
                tmp_path / "part_b.py": "# Part B\ndef func_b(): pass\n",
            },
            reason="split oversized module",
        )
        assert (tmp_path / "part_a.py").exists()
        assert (tmp_path / "part_b.py").exists()
        assert not original.exists()
