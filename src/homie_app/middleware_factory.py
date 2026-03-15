from __future__ import annotations
from pathlib import Path
from typing import Optional

from homie_core.config import HomieConfig
from homie_core.middleware import MiddlewareStack
from homie_core.middleware.summarization import SummarizationMiddleware
from homie_core.middleware.arg_truncation import ArgTruncationMiddleware
from homie_core.middleware.long_line_split import LongLineSplitMiddleware
from homie_core.middleware.large_result_eviction import LargeResultEvictionMiddleware
from homie_core.middleware.context_overflow import ContextOverflowRecoveryMiddleware
from homie_core.middleware.dangling_tool_repair import DanglingToolCallMiddleware
from homie_core.middleware.todo import TodoMiddleware
from homie_core.middleware.shell_allowlist import ShellAllowlistMiddleware
from homie_core.middleware.skills import SkillsMiddleware
from homie_core.middleware.memory import MemoryMiddleware
from homie_core.backend.local_filesystem import LocalFilesystemBackend
from homie_core.memory.working import WorkingMemory


def build_middleware_stack(
    config: HomieConfig,
    working_memory: WorkingMemory,
    backend: Optional[LocalFilesystemBackend] = None,
    skill_paths: Optional[list[str | Path]] = None,
    memory_paths: Optional[list[str | Path]] = None,
) -> MiddlewareStack:
    """Build the default middleware stack for Homie.

    Creates all Phase 1-5 middleware with sensible defaults.
    Pass backend=None to auto-create from config.storage.path.
    """
    if backend is None:
        backend = LocalFilesystemBackend(root_dir=config.storage.path)

    middleware = [
        # Phase 2: Context Intelligence
        ContextOverflowRecoveryMiddleware(working_memory=working_memory),
        SummarizationMiddleware(config=config, backend=backend, working_memory=working_memory),
        ArgTruncationMiddleware(config=config, working_memory=working_memory),
        LongLineSplitMiddleware(config=config),
        LargeResultEvictionMiddleware(config=config, backend=backend),
        # Phase 3: Multi-Agent & Planning
        DanglingToolCallMiddleware(working_memory=working_memory),
        TodoMiddleware(),
        # Phase 4: Safety & Control
        ShellAllowlistMiddleware(),
    ]

    # Phase 5: Extensibility (optional — paths may be None)
    if skill_paths:
        middleware.append(SkillsMiddleware(skill_paths=skill_paths))

    # Load AGENTS.md if it exists in the homie directory
    agents_md = Path(config.storage.path) / "AGENTS.md"
    memory_files = list(memory_paths or [])
    if agents_md.exists() and str(agents_md) not in [str(p) for p in memory_files]:
        memory_files.append(agents_md)
    if memory_files:
        middleware.append(MemoryMiddleware(memory_paths=memory_files))

    return MiddlewareStack(middleware)
