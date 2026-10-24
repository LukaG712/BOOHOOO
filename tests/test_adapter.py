"""Tests for the JunieAdapter interface.

Verifies that the adapter provides a clean, typed API for Greenie integration
without requiring private implementation details.
"""

import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from junie_adapter import (
    AdapterRequest,
    AdapterResult,
    IsolationMode,
    JunieAdapter,
    OrchestratorConfig,
)
from junie_adapter.models import WorkerStatus


class TestAdapterRequest:
    """Test AdapterRequest dataclass."""

    def test_auto_generates_worker_id(self):
        """Test that worker_id is auto-generated if not provided."""
        request = AdapterRequest(
            task="Test task",
            project_path="/tmp/test",
        )
        assert request.worker_id is not None
        assert request.worker_id.startswith("worker-")

    def test_auto_generates_task_id(self):
        """Test that task_id is auto-generated if not provided."""
        request = AdapterRequest(
            task="Test task",
            project_path="/tmp/test",
        )
        assert request.task_id is not None
        assert request.task_id.startswith("task-")

    def test_custom_ids_preserved(self):
        """Test that custom IDs are preserved."""
        request = AdapterRequest(
            task="Test task",
            project_path="/tmp/test",
            worker_id="my-worker",
            task_id="my-task",
        )
        assert request.worker_id == "my-worker"
        assert request.task_id == "my-task"

    def test_default_values(self):
        """Test default values for optional fields."""
        request = AdapterRequest(
            task="Test task",
            project_path="/tmp/test",
        )
        assert request.use_worktree is True
        assert request.timeout == 1800
        assert request.isolation_mode == IsolationMode.WORKTREE
        assert request.parent_task_id is None
        assert request.config_overrides is None


class TestAdapterResult:
    """Test AdapterResult dataclass."""

    def test_completed_result(self):
        """Test a completed result."""
        result = AdapterResult(
            worker_id="worker-1",
            task_id="task-1",
            status=WorkerStatus.COMPLETED,
            output="Done",
            exit_code=0,
            changed_files=["file1.py"],
            diff_summary="Modified 1 file",
            error=None,
            worktree_path="/tmp/worktree",
            branch_name="junie-worker-worker-1",
            duration_seconds=5.0,
        )
        assert result.status == WorkerStatus.COMPLETED
        assert result.exit_code == 0
        assert len(result.changed_files) == 1

    def test_failed_result(self):
        """Test a failed result."""
        result = AdapterResult(
            worker_id="worker-1",
            task_id="task-1",
            status=WorkerStatus.FAILED,
            output="",
            exit_code=1,
            changed_files=[],
            diff_summary="",
            error="Something went wrong",
            worktree_path=None,
            branch_name=None,
            duration_seconds=1.0,
        )
        assert result.status == WorkerStatus.FAILED
        assert result.error == "Something went wrong"


class TestJunieAdapter:
    """Test JunieAdapter class."""

    def test_check_health_workspace_exists(self):
        """Test health check returns True when workspace exists."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = JunieAdapter(workspace_path=tmp)
            assert adapter.check_health() is True

    def test_check_health_workspace_missing(self):
        """Test health check returns False when workspace does not exist."""
        adapter = JunieAdapter(workspace_path="/nonexistent/path")
        assert adapter.check_health() is False

    def test_check_health_with_config(self):
        """Test health check with explicit config."""
        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            adapter = JunieAdapter(config=config)
            assert adapter.check_health() is True

    @pytest.mark.asyncio
    async def test_get_changed_files_empty(self):
        """Test get_changed_files returns empty list for clean repo."""
        with tempfile.TemporaryDirectory() as tmp:
            # Initialize a Git repo using subprocess instead of os.system
            import subprocess

            subprocess.run(["git", "init", tmp], capture_output=True)
            subprocess.run(
                ["git", "config", "user.email", "test@test.com"],
                cwd=tmp,
                capture_output=True,
            )
            subprocess.run(
                ["git", "config", "user.name", "Test"],
                cwd=tmp,
                capture_output=True,
            )
            # Create and commit a file
            test_file = Path(tmp) / "test.txt"
            test_file.write_text("hello")
            subprocess.run(["git", "add", "."], cwd=tmp, capture_output=True)
            subprocess.run(
                ["git", "commit", "-m", "init"], cwd=tmp, capture_output=True
            )

            adapter = JunieAdapter(workspace_path=tmp)
            files = await adapter.get_changed_files(tmp)
            assert files == []

    @pytest.mark.asyncio
    async def test_get_changed_files_non_git(self):
        """Test get_changed_files returns empty list for non-Git directory."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = JunieAdapter(workspace_path=tmp)
            files = await adapter.get_changed_files(tmp)
            # Should not raise, returns empty list
            assert isinstance(files, list)

    @pytest.mark.asyncio
    async def test_cleanup_worktree_nonexistent(self):
        """Test cleanup_worktree returns False for nonexistent path."""
        adapter = JunieAdapter(workspace_path=tempfile.gettempdir())
        result = await adapter.cleanup_worktree("/nonexistent/worktree")
        assert result is False

    @pytest.mark.asyncio
    async def test_cancel_task(self):
        """Test cancel_task returns False (Junie CLI doesn't support cancellation)."""
        adapter = JunieAdapter(workspace_path=tempfile.gettempdir())
        result = await adapter.cancel_task("worker-1")
        assert result is False

    @pytest.mark.asyncio
    async def test_submit_task_with_mock(self):
        """Test submit_task with mocked execute_worker."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = JunieAdapter(workspace_path=tmp)
            request = AdapterRequest(
                task="Test task",
                project_path=tmp,
                use_worktree=False,
            )

            # Mock execute_worker to return a completed result
            from datetime import datetime, timezone

            from junie_adapter.models import WorkerExecution

            mock_result = WorkerExecution(
                worker_id=request.worker_id,
                task_id=request.task_id,
                parent_task_id=None,
                task_description="Test task",
                project_path=tmp,
                worktree_path=None,
                status=WorkerStatus.COMPLETED,
                pid=None,
                start_time=datetime.now(timezone.utc),
                end_time=datetime.now(timezone.utc),
                stdout="Test output",
                stderr="",
                exit_code=0,
                result_summary="Test output",
                changed_files=[],
                error=None,
            )

            with patch(
                "junie_adapter.worker_executor.execute_worker",
                new=AsyncMock(return_value=mock_result),
            ):
                result = await adapter.submit_task(request)

            assert isinstance(result, AdapterResult)
            assert result.status == WorkerStatus.COMPLETED
            assert result.output == "Test output"
            assert result.worker_id == request.worker_id

    @pytest.mark.asyncio
    async def test_submit_task_error_handling(self):
        """Test submit_task returns FAILED status on exception."""
        with tempfile.TemporaryDirectory() as tmp:
            adapter = JunieAdapter(workspace_path=tmp)
            request = AdapterRequest(
                task="Test task",
                project_path=tmp,
                use_worktree=False,
            )

            with patch(
                "junie_adapter.worker_executor.execute_worker",
                new=AsyncMock(side_effect=RuntimeError("Test error")),
            ):
                result = await adapter.submit_task(request)

            assert result.status == WorkerStatus.FAILED
            assert "Test error" in result.error
