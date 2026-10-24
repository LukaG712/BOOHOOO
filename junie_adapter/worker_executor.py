"""Single-worker execution with worktree isolation and metadata capture.

Wraps run_junie_task() to provide worker-specific metadata, worktree-based
isolation, output capture, timeout enforcement, and changed-file detection.

This module is opt-in — the existing run_junie_task() API is unchanged.
"""

import asyncio
import json
import logging
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .config import IsolationMode, OrchestratorConfig
from .models import WorkerExecution, WorkerStatus
from .persistent_store import PersistentStore
from .worktree_manager import (
    WorktreeInfo,
    create_worktree,
    remove_worktree,
    validate_git_repo,
)


def _worker_execution_to_task_record(execution: WorkerExecution):
    """Convert a WorkerExecution to a TaskRecord for persistence."""
    from .persistent_store import TaskRecord

    duration = 0.0
    if execution.end_time and execution.start_time:
        duration = (execution.end_time - execution.start_time).total_seconds()
    return TaskRecord(
        id=execution.task_id,
        project_id=None,
        worker_id=execution.worker_id,
        parent_task_id=execution.parent_task_id,
        task_description=execution.task_description,
        project_path=execution.project_path,
        worktree_path=execution.worktree_path,
        status=execution.status.value,
        exit_code=execution.exit_code,
        stdout=execution.stdout,
        stderr=execution.stderr,
        result_summary=execution.result_summary,
        changed_files=json.dumps(execution.changed_files)
        if execution.changed_files
        else "[]",
        error=execution.error,
        duration_seconds=duration,
        start_time=execution.start_time,
        end_time=execution.end_time,
    )


runs_logger = logging.getLogger("runs")

# Default memory limit for output capture (10 MB)
DEFAULT_OUTPUT_LIMIT = 10 * 1024 * 1024


def _truncate_output(text: str, limit: int = DEFAULT_OUTPUT_LIMIT) -> str:
    """Truncate output to prevent unbounded memory growth."""
    if len(text) > limit:
        truncated = text[:limit]
        return f"{truncated}\n\n[Output truncated: {len(text)} bytes total, showing first {limit} bytes]"
    return text


class WorkerSemaphore:
    """Asyncio semaphore-based concurrency limiter for workers."""

    def __init__(self, max_workers: int = 1):
        self._semaphore = asyncio.Semaphore(max_workers)
        self._active_count = 0
        self._lock = asyncio.Lock()

    async def acquire(self):
        await self._semaphore.acquire()
        async with self._lock:
            self._active_count += 1

    async def release(self):
        async with self._lock:
            self._active_count -= 1
        self._semaphore.release()

    @property
    def active_count(self) -> int:
        return self._active_count


_worker_semaphore = WorkerSemaphore(max_workers=1)


class WorkerRegistry:
    """Tracks active workers for monitoring and cancellation.

    Provides a thread-safe mapping from worker_id to WorkerExecution,
    allowing external code to query active workers and attempt cancellation.
    """

    def __init__(self):
        self._workers: dict[str, WorkerExecution] = {}
        self._lock = asyncio.Lock()

    async def register(self, worker_id: str, execution: WorkerExecution) -> None:
        """Register a worker as active.

        Args:
            worker_id: Unique worker identifier.
            execution: WorkerExecution object with initial RUNNING status.
        """
        async with self._lock:
            execution.status = WorkerStatus.RUNNING
            self._workers[worker_id] = execution

    async def complete(self, worker_id: str, execution: WorkerExecution) -> None:
        """Mark a worker as completed and remove from active registry.

        Args:
            worker_id: Unique worker identifier.
            execution: Final WorkerExecution object with result status.
        """
        async with self._lock:
            self._workers[worker_id] = execution

    async def get(self, worker_id: str) -> WorkerExecution | None:
        """Get a worker's execution info by ID.

        Args:
            worker_id: Unique worker identifier.

        Returns:
            WorkerExecution if found, None otherwise.
        """
        async with self._lock:
            return self._workers.get(worker_id)

    async def get_active(self) -> list[WorkerExecution]:
        """Get all currently active workers.

        Returns:
            List of WorkerExecution objects for active workers.
        """
        async with self._lock:
            return list(self._workers.values())

    async def remove(self, worker_id: str) -> bool:
        """Remove a worker from the registry.

        Args:
            worker_id: Unique worker identifier.

        Returns:
            True if the worker was found and removed, False otherwise.
        """
        async with self._lock:
            if worker_id in self._workers:
                del self._workers[worker_id]
                return True
            return False

    @property
    def active_count(self) -> int:
        """Number of active workers (synchronous approximation)."""
        return len(self._workers)


# Global worker registry instance
_worker_registry = WorkerRegistry()


def get_worker_registry() -> WorkerRegistry:
    """Get the global worker registry instance.

    Returns:
        The WorkerRegistry singleton.
    """
    return _worker_registry


def set_max_concurrent_workers(max_workers: int) -> None:
    """Set the maximum number of concurrent workers.

    Args:
        max_workers: Maximum concurrent workers (default 1 per JUNIE-3394).
    """
    _worker_semaphore._semaphore = asyncio.Semaphore(max_workers)


async def execute_worker(
    task_description: str,
    config: OrchestratorConfig,
    worker_id: str,
    task_id: str,
    parent_task_id: str | None = None,
    use_worktree: bool = True,
    timeout: int | None = None,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    store: Optional["PersistentStore"] = None,
) -> WorkerExecution:
    """Execute a single worker task with optional worktree isolation.

    Creates a Git worktree for isolation (if available), runs Junie CLI,
    captures output and changed files, then returns a WorkerExecution result.

    Args:
        task_description: The task to execute.
        config: Orchestrator configuration.
        worker_id: Unique identifier for this worker.
        task_id: Unique identifier for this task.
        parent_task_id: Optional parent task ID for task graph tracking.
        use_worktree: Whether to use worktree isolation. Defaults to True.
        timeout: Per-task timeout in seconds. Defaults to config.junie_timeout.
        output_limit: Maximum output size in bytes. Defaults to 10 MB.
        store: Optional PersistentStore to save task records. If provided,
            the task will be persisted at each lifecycle stage (queued, running,
            completed/failed/timed_out).

    Returns:
        WorkerExecution with full metadata about the execution.

    Raises:
        RuntimeError: If isolation mode is REQUIRED and the project is not a Git repo.
    """
    start_time = datetime.now(timezone.utc)
    timeout = timeout or config.junie_timeout

    # Enforce concurrency limit
    await _worker_semaphore.acquire()

    # Determine the project path
    project_path = config.workspace_dir
    worktree_info: WorktreeInfo | None = None
    actual_project_path = project_path
    pid = None

    # Create initial execution record for registry
    initial_execution = WorkerExecution(
        worker_id=worker_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        task_description=task_description,
        project_path=str(project_path),
        worktree_path=None,
        status=WorkerStatus.QUEUED,
        pid=pid,
        start_time=start_time,
        end_time=None,
        stdout="",
        stderr="",
        exit_code=None,
        result_summary=None,
        changed_files=[],
        error=None,
    )

    # Persist initial task record as "queued"
    if store is not None:
        try:
            task_record = _worker_execution_to_task_record(initial_execution)
            store.save_task(task_record)
            runs_logger.debug(f"Persisted task {task_id} as queued")
        except Exception as e:
            runs_logger.warning(f"Failed to persist task {task_id} as queued: {e}")

    try:
        # Handle isolation mode
        if config.isolation_mode == IsolationMode.REQUIRED:
            if not validate_git_repo(project_path):
                raise RuntimeError(
                    f"Isolation mode is REQUIRED but {project_path} is not a Git repo. "
                    "Cannot execute worker without worktree isolation."
                )
            worktree_info = create_worktree(project_path, worker_id)
            actual_project_path = worktree_info.worktree_path
            runs_logger.info(
                f"Worker {worker_id} using worktree: {worktree_info.worktree_path}"
            )
        elif config.isolation_mode == IsolationMode.WORKTREE:
            if use_worktree and validate_git_repo(project_path):
                try:
                    worktree_info = create_worktree(project_path, worker_id)
                    actual_project_path = worktree_info.worktree_path
                    runs_logger.info(
                        f"Worker {worker_id} using worktree: {worktree_info.worktree_path}"
                    )
                except RuntimeError as e:
                    runs_logger.warning(
                        f"Failed to create worktree for worker {worker_id}: {e}. "
                        "Falling back to project directory."
                    )
                    actual_project_path = project_path
            elif use_worktree:
                runs_logger.warning(
                    f"Worktree requested but {project_path} is not a Git repo. "
                    "Running without worktree isolation."
                )
        # IsolationMode.NONE: no worktree, use project_path directly

        # Use a local config copy — do NOT mutate the shared config object
        worker_config = replace(config, workspace_path=str(actual_project_path))

        # Register worker in the registry
        initial_execution.worktree_path = (
            str(worktree_info.worktree_path) if worktree_info else None
        )
        await _worker_registry.register(worker_id, initial_execution)

        # Persist task as "running"
        if store is not None:
            try:
                running_record = _worker_execution_to_task_record(initial_execution)
                store.save_task(running_record)
                runs_logger.debug(f"Persisted task {task_id} as running")
            except Exception as e:
                runs_logger.warning(f"Failed to persist task {task_id} as running: {e}")

        # Run the Junie task with timeout
        try:
            from .junie_runner import run_junie_task

            result = await asyncio.wait_for(
                run_junie_task(task_description, worker_config),
                timeout=timeout,
            )

            end_time = datetime.now(timezone.utc)
            stdout = result.answer or ""
            stderr = result.error or ""
            exit_code = 0 if result.status.value == "completed" else 1

            # Detect changed files in the worktree
            changed_files: list[str] = []
            if worktree_info:
                try:
                    from .worktree_manager import (
                        get_changed_files as _get_changed_files,
                    )

                    changed_files = _get_changed_files(worktree_info.worktree_path)
                except Exception as e:
                    runs_logger.warning(f"Failed to detect changed files: {e}")

            # Determine worker status
            if result.status.value == "completed":
                worker_status = WorkerStatus.COMPLETED
            elif result.status.value == "timed_out":
                worker_status = WorkerStatus.TIMED_OUT
            else:
                worker_status = WorkerStatus.FAILED

            final_result = WorkerExecution(
                worker_id=worker_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                task_description=task_description,
                project_path=str(project_path),
                worktree_path=str(worktree_info.worktree_path)
                if worktree_info
                else None,
                status=worker_status,
                pid=pid,
                start_time=start_time,
                end_time=end_time,
                stdout=_truncate_output(stdout, output_limit),
                stderr=_truncate_output(stderr, output_limit),
                exit_code=exit_code,
                result_summary=result.answer[:500] if result.answer else None,
                changed_files=changed_files,
                error=result.error,
            )
            await _worker_registry.complete(worker_id, final_result)

            # Persist final task record
            if store is not None:
                try:
                    final_record = _worker_execution_to_task_record(final_result)
                    store.save_task(final_record)
                    runs_logger.debug(
                        f"Persisted task {task_id} as {worker_status.value}"
                    )
                except Exception as e:
                    runs_logger.warning(
                        f"Failed to persist task {task_id} as {worker_status.value}: {e}"
                    )

            return final_result

        except asyncio.TimeoutError:
            end_time = datetime.now(timezone.utc)
            runs_logger.error(f"Worker {worker_id} timed out after {timeout}s")
            timeout_result = WorkerExecution(
                worker_id=worker_id,
                task_id=task_id,
                parent_task_id=parent_task_id,
                task_description=task_description,
                project_path=str(project_path),
                worktree_path=str(worktree_info.worktree_path)
                if worktree_info
                else None,
                status=WorkerStatus.TIMED_OUT,
                pid=pid,
                start_time=start_time,
                end_time=end_time,
                stdout="",
                stderr=f"Worker timed out after {timeout}s",
                exit_code=None,
                result_summary=None,
                changed_files=[],
                error=f"Timeout after {timeout}s",
            )
            await _worker_registry.complete(worker_id, timeout_result)

            # Persist timeout task record
            if store is not None:
                try:
                    timeout_record = _worker_execution_to_task_record(timeout_result)
                    store.save_task(timeout_record)
                    runs_logger.debug(f"Persisted task {task_id} as timed_out")
                except Exception as e:
                    runs_logger.warning(
                        f"Failed to persist task {task_id} as timed_out: {e}"
                    )

            return timeout_result

    finally:
        # Log worktree location for review — do NOT mutate shared config
        if worktree_info:
            runs_logger.info(
                f"Worker {worker_id} completed. Worktree preserved at: "
                f"{worktree_info.worktree_path} for review."
            )

        # Release concurrency slot
        await _worker_semaphore.release()


async def execute_worker_and_cleanup(
    task_description: str,
    config: OrchestratorConfig,
    worker_id: str,
    task_id: str,
    parent_task_id: str | None = None,
    use_worktree: bool = True,
    timeout: int | None = None,
    output_limit: int = DEFAULT_OUTPUT_LIMIT,
    store: Optional["PersistentStore"] = None,
) -> WorkerExecution:
    """Execute a worker task and automatically clean up the worktree.

    Similar to execute_worker but removes the worktree after execution.
    Use this when you don't need to preserve the worktree for review.

    Args:
        task_description: The task to execute.
        config: Orchestrator configuration.
        worker_id: Unique identifier for this worker.
        task_id: Unique identifier for this task.
        parent_task_id: Optional parent task ID.
        use_worktree: Whether to use worktree isolation.
        timeout: Per-task timeout in seconds.
        output_limit: Maximum output size in bytes.
        store: Optional PersistentStore to save task records.

    Returns:
        WorkerExecution with full metadata.
    """
    result = await execute_worker(
        task_description=task_description,
        config=config,
        worker_id=worker_id,
        task_id=task_id,
        parent_task_id=parent_task_id,
        use_worktree=use_worktree,
        timeout=timeout,
        output_limit=output_limit,
        store=store,
    )

    # Clean up worktree if it was created
    if result.worktree_path:
        try:
            worktree_path = Path(result.worktree_path)
            remove_worktree(worktree_path, force=True)
            runs_logger.info(f"Cleaned up worktree for worker {worker_id}")
        except Exception as e:
            runs_logger.warning(
                f"Failed to clean up worktree for worker {worker_id}: {e}"
            )

    return result
