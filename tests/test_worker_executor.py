"""Tests for single-worker execution with worktree isolation.

Uses mocked Junie (fake_junie.py) and mocked worktree manager — no live Junie required.
"""

import asyncio
import subprocess
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from junie_adapter.config import OrchestratorConfig
from junie_adapter.models import WorkerExecution, WorkerStatus
from junie_adapter.worker_executor import (
    DEFAULT_OUTPUT_LIMIT,
    _truncate_output,
    execute_worker,
    execute_worker_and_cleanup,
)


@pytest.fixture
def config(tmp_path: Path):
    """Create a test config with fake Junie binary."""
    workspace = tmp_path / "workspace"
    workspace.mkdir(parents=True, exist_ok=True)
    return OrchestratorConfig(
        workspace_path=str(workspace),
        junie_bin=str(Path(__file__).parent / "fake_junie.py"),
        junie_timeout=30,
        ollama_url="http://localhost:11434",
        ollama_model="llama3",
        refine_prompt=False,
        api_key="test-key",
        vault_path=None,
        vault_map_path=None,
        mirror_logs_to_vault=False,
    )


def _init_git_repo(path: Path) -> Path:
    """Initialize a Git repo at the given path with one commit on the master branch."""
    path.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "init", "-b", "master"], cwd=str(path), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.com"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test User"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    (path / "README.md").write_text("# Test Project\n", encoding="utf-8")
    subprocess.run(
        ["git", "add", "README.md"], cwd=str(path), check=True, capture_output=True
    )
    subprocess.run(
        ["git", "commit", "-m", "Initial commit"],
        cwd=str(path),
        check=True,
        capture_output=True,
    )
    return path


class TestTruncateOutput:
    def test_no_truncation_needed(self):
        text = "Short output"
        assert _truncate_output(text, limit=1000) == text

    def test_truncation_applied(self):
        text = "x" * 2000
        result = _truncate_output(text, limit=100)
        assert len(result) > 100  # Includes truncation message
        assert "[Output truncated:" in result

    def test_default_limit(self):
        text = "x" * (DEFAULT_OUTPUT_LIMIT + 1)
        result = _truncate_output(text)
        assert "[Output truncated:" in result


class TestExecuteWorker:
    @pytest.mark.asyncio
    async def test_successful_execution(self, config: OrchestratorConfig):
        """Test that a successful worker execution returns correct metadata."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="Test task",
                config=config,
                worker_id="worker-1",
                task_id="task-1",
            )

        assert isinstance(result, WorkerExecution)
        assert result.worker_id == "worker-1"
        assert result.task_id == "task-1"
        assert result.parent_task_id is None
        assert result.task_description == "Test task"
        assert result.status == WorkerStatus.COMPLETED
        assert result.start_time <= result.end_time
        assert result.exit_code == 0
        assert result.worktree_path is None

    @pytest.mark.asyncio
    async def test_execution_with_parent_task_id(self, config: OrchestratorConfig):
        """Test that parent_task_id is correctly passed through."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="Subtask",
                config=config,
                worker_id="worker-2",
                task_id="task-2",
                parent_task_id="parent-1",
            )

        assert result.parent_task_id == "parent-1"

    @pytest.mark.asyncio
    async def test_execution_with_worktree(
        self, config: OrchestratorConfig, tmp_path: Path
    ):
        """Test execution with worktree isolation."""
        _init_git_repo(config.workspace_dir)

        mock_info = MagicMock()
        mock_info.worktree_path = tmp_path / "worktree"
        mock_info.worktree_path.mkdir(parents=True, exist_ok=True)

        from junie_adapter.models import RunResult, TaskStatus

        mock_result = RunResult(
            task="Test with worktree",
            answer="Test answer",
            status=TaskStatus.COMPLETED,
            error=None,
            duration_seconds=1.0,
        )

        with (
            patch("junie_adapter.worker_executor.validate_git_repo", return_value=True),
            patch(
                "junie_adapter.worker_executor.create_worktree", return_value=mock_info
            ),
            patch(
                "junie_adapter.worktree_manager.get_changed_files",
                return_value=["file1.py"],
            ),
            patch(
                "junie_adapter.junie_runner.run_junie_task",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            result = await execute_worker(
                task_description="Test with worktree",
                config=config,
                worker_id="worker-3",
                task_id="task-3",
            )

        assert result.worktree_path is not None
        assert result.changed_files == ["file1.py"]

    @pytest.mark.asyncio
    async def test_execution_timeout(self, config: OrchestratorConfig):
        """Test that timeout returns TIMED_OUT status."""
        with (
            patch(
                "junie_adapter.worker_executor.validate_git_repo", return_value=False
            ),
            patch(
                "junie_adapter.junie_runner.run_junie_task",
                new=AsyncMock(side_effect=asyncio.TimeoutError()),
            ),
        ):
            result = await execute_worker(
                task_description="Slow task",
                config=config,
                worker_id="worker-4",
                task_id="task-4",
                timeout=0.1,
            )

        assert result.status == WorkerStatus.TIMED_OUT
        assert result.error is not None
        assert "Timeout" in result.error

    @pytest.mark.asyncio
    async def test_execution_failure(self, config: OrchestratorConfig):
        """Test that a failed Junie execution returns FAILED status."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            # Use a non-existent Junie binary to trigger failure
            config.junie_bin = "/nonexistent/junie"
            result = await execute_worker(
                task_description="Failing task",
                config=config,
                worker_id="worker-5",
                task_id="task-5",
            )

        assert result.status == WorkerStatus.FAILED
        assert result.exit_code == 1

    @pytest.mark.asyncio
    async def test_non_git_fallback(self, config: OrchestratorConfig):
        """Test that non-Git projects fall back to no worktree."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="No Git task",
                config=config,
                worker_id="worker-6",
                task_id="task-6",
                use_worktree=True,
            )

        assert result.worktree_path is None
        assert result.status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_use_worktree_false(self, config: OrchestratorConfig):
        """Test that use_worktree=False skips worktree creation."""
        with (
            patch("junie_adapter.worker_executor.validate_git_repo", return_value=True),
            patch("junie_adapter.worktree_manager.create_worktree") as mock_create,
        ):
            await execute_worker(
                task_description="No worktree task",
                config=config,
                worker_id="worker-7",
                task_id="task-7",
                use_worktree=False,
            )

        mock_create.assert_not_called()


class TestExecuteWorkerAndCleanup:
    @pytest.mark.asyncio
    async def test_cleanup_returns_result(self, config: OrchestratorConfig):
        """Test that execute_worker_and_cleanup returns a WorkerExecution result."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker_and_cleanup(
                task_description="Cleanup task",
                config=config,
                worker_id="worker-8",
                task_id="task-8",
            )

        assert isinstance(result, WorkerExecution)
        assert result.status == WorkerStatus.COMPLETED
        assert result.worker_id == "worker-8"

    @pytest.mark.asyncio
    async def test_cleanup_no_worktree(self, config: OrchestratorConfig):
        """Test that cleanup is skipped when no worktree was created."""
        with (
            patch(
                "junie_adapter.worker_executor.validate_git_repo", return_value=False
            ),
            patch("junie_adapter.worktree_manager.remove_worktree") as mock_remove,
        ):
            await execute_worker_and_cleanup(
                task_description="No cleanup task",
                config=config,
                worker_id="worker-9",
                task_id="task-9",
            )

        mock_remove.assert_not_called()


class TestWorkerExecutionModel:
    def test_worker_status_values(self):
        """Test that all WorkerStatus values are defined."""
        assert WorkerStatus.QUEUED.value == "queued"
        assert WorkerStatus.RUNNING.value == "running"
        assert WorkerStatus.COMPLETED.value == "completed"
        assert WorkerStatus.FAILED.value == "failed"
        assert WorkerStatus.TIMED_OUT.value == "timed_out"
        assert WorkerStatus.CANCELLED.value == "cancelled"
        assert WorkerStatus.INTERRUPTED.value == "interrupted"

    def test_worker_execution_creation(self):
        """Test that WorkerExecution can be created with all fields."""
        now = datetime.now(timezone.utc)
        we = WorkerExecution(
            worker_id="w1",
            task_id="t1",
            parent_task_id=None,
            task_description="Test",
            project_path="/path/to/project",
            worktree_path="/path/to/worktree",
            status=WorkerStatus.COMPLETED,
            pid=12345,
            start_time=now,
            end_time=now,
            stdout="output",
            stderr="",
            exit_code=0,
            result_summary="Summary",
            changed_files=["file1.py", "file2.py"],
            error=None,
        )
        assert we.worker_id == "w1"
        assert we.changed_files == ["file1.py", "file2.py"]


@pytest.mark.integration
@pytest.mark.asyncio
class TestLiveJunieIntegration:
    """Integration tests that require a live Junie CLI installation.

    These are skipped by default. Run with: pytest -m integration
    """

    async def test_live_execution(self):
        """Test execution against a real Junie CLI if available."""
        pytest.skip("Live Junie integration test — requires JUNIE_BIN env var")


class TestPersistentStoreIntegration:
    """Test that execute_worker persists task records to the PersistentStore."""

    @pytest.mark.asyncio
    async def test_persists_queued_and_completed(self, config: OrchestratorConfig):
        """Test that task is persisted as queued then completed."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="Persisted task",
                config=config,
                worker_id="worker-persist-1",
                task_id="task-persist-1",
                store=store,
            )

        assert result.status == WorkerStatus.COMPLETED

        # Verify task is in the store
        task = store.get_task("task-persist-1")
        assert task is not None
        assert task.id == "task-persist-1"
        assert task.worker_id == "worker-persist-1"
        assert task.status == "completed"
        assert task.task_description == "Persisted task"
        assert task.duration_seconds >= 0.0

    @pytest.mark.asyncio
    async def test_persists_with_parent_task_id(self, config: OrchestratorConfig):
        """Test that task with parent_task_id is persisted correctly."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="Persisted task with parent",
                config=config,
                worker_id="worker-persist-parent",
                task_id="task-persist-parent",
                parent_task_id="parent-task-1",
                store=store,
            )

        assert result.status == WorkerStatus.COMPLETED
        assert result.parent_task_id == "parent-task-1"

        # Verify task is in the store
        task = store.get_task("task-persist-parent")
        assert task is not None, "Task not found in store"
        assert task.parent_task_id == "parent-task-1", (
            f"Expected 'parent-task-1', got {task.parent_task_id!r}"
        )

    @pytest.mark.asyncio
    async def test_persists_failed_status(self, config: OrchestratorConfig):
        """Test that a failed task is persisted with FAILED status."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            config.junie_bin = "/nonexistent/junie"
            result = await execute_worker(
                task_description="Failing persisted task",
                config=config,
                worker_id="worker-persist-2",
                task_id="task-persist-2",
                store=store,
            )

        assert result.status == WorkerStatus.FAILED

        task = store.get_task("task-persist-2")
        assert task is not None
        assert task.status == "failed"

    @pytest.mark.asyncio
    async def test_persists_timed_out_status(self, config: OrchestratorConfig):
        """Test that a timed out task is persisted with TIMED_OUT status."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        with (
            patch(
                "junie_adapter.worker_executor.validate_git_repo", return_value=False
            ),
            patch(
                "junie_adapter.junie_runner.run_junie_task",
                new=AsyncMock(side_effect=asyncio.TimeoutError()),
            ),
        ):
            result = await execute_worker(
                task_description="Timed out task",
                config=config,
                worker_id="worker-persist-3",
                task_id="task-persist-3",
                timeout=0.1,
                store=store,
            )

        assert result.status == WorkerStatus.TIMED_OUT

        task = store.get_task("task-persist-3")
        assert task is not None
        assert task.status == "timed_out"
        assert task.error is not None
        assert "Timeout" in task.error

    @pytest.mark.asyncio
    async def test_no_store_skips_persistence(self, config: OrchestratorConfig):
        """Test that execution works normally when no store is provided."""
        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="No store task",
                config=config,
                worker_id="worker-persist-4",
                task_id="task-persist-4",
                store=None,
            )

        assert result.status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_persists_changed_files(
        self, config: OrchestratorConfig, tmp_path: Path
    ):
        """Test that changed files are persisted correctly."""
        from junie_adapter.models import RunResult, TaskStatus
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        _init_git_repo(config.workspace_dir)

        mock_info = MagicMock()
        mock_info.worktree_path = tmp_path / "worktree"
        mock_info.worktree_path.mkdir(parents=True, exist_ok=True)

        mock_result = RunResult(
            task="Test with changed files",
            answer="Changed files answer",
            status=TaskStatus.COMPLETED,
            error=None,
            duration_seconds=1.0,
        )

        with (
            patch("junie_adapter.worker_executor.validate_git_repo", return_value=True),
            patch(
                "junie_adapter.worker_executor.create_worktree", return_value=mock_info
            ),
            patch(
                "junie_adapter.worktree_manager.get_changed_files",
                return_value=["file1.py", "file2.py"],
            ),
            patch(
                "junie_adapter.junie_runner.run_junie_task",
                new=AsyncMock(return_value=mock_result),
            ),
        ):
            await execute_worker(
                task_description="Test with changed files",
                config=config,
                worker_id="worker-persist-6",
                task_id="task-persist-6",
                store=store,
            )

        task = store.get_task("task-persist-6")
        assert task is not None
        assert task.changed_files == ["file1.py", "file2.py"]

    @pytest.mark.asyncio
    async def test_store_error_does_not_break_execution(
        self, config: OrchestratorConfig
    ):
        """Test that a store error doesn't break task execution."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")
        # Break the store by replacing save_task with a failing mock
        store.save_task = lambda task: (_ for _ in ()).throw(RuntimeError("DB error"))

        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker(
                task_description="Store error task",
                config=config,
                worker_id="worker-persist-7",
                task_id="task-persist-7",
                store=store,
            )

        # Should still complete successfully despite store errors
        assert result.status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_execute_worker_and_cleanup_passes_store(
        self, config: OrchestratorConfig
    ):
        """Test that execute_worker_and_cleanup passes the store through."""
        from junie_adapter.persistent_store import PersistentStore

        store = PersistentStore(":memory:")

        with patch(
            "junie_adapter.worker_executor.validate_git_repo", return_value=False
        ):
            result = await execute_worker_and_cleanup(
                task_description="Cleanup with store",
                config=config,
                worker_id="worker-persist-8",
                task_id="task-persist-8",
                store=store,
            )

        assert result.status == WorkerStatus.COMPLETED
        task = store.get_task("task-persist-8")
        assert task is not None
        assert task.status == "completed"
