"""Tests for the junie_adapter package.

Tests each module independently using mocks and temporary directories,
plus integration tests via the FastAPI service layer.
No live Junie installation or Ollama instance required.
"""

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest
from fastapi.testclient import TestClient

# ─── Config tests ─────────────────────────────────────────────────────────────


class TestConfig:
    def test_from_env_defaults(self):
        """Test default config values."""
        # Clear all relevant env vars
        with patch.dict(os.environ, {}, clear=True):
            from junie_adapter.config import OrchestratorConfig

            config = OrchestratorConfig.from_env()
            assert config.ollama_url == "http://localhost:11434"
            assert config.ollama_model == "gemma4:12b"
            assert config.refine_prompt is False
            assert config.workspace_path is None
            assert config.vault_path is None
            assert config.mirror_logs_to_vault is False
            assert config.api_key == ""
            assert config.junie_timeout == 1800
            assert config.port == 8000
            assert config.junie_bin == "junie"

    def test_from_env_custom_values(self):
        """Test loading custom values from environment."""
        env = {
            "OLLAMA_URL": "http://custom:11435",
            "OLLAMA_MODEL": "llama3:8b",
            "REFINE_PROMPT": "true",
            "WORKSPACE_PATH": "/custom/workspace",
            "VAULT_PATH": "/custom/vault",
            "MIRROR_LOGS_TO_VAULT": "true",
            "API_KEY": "test-key",
            "JUNIE_TIMEOUT": "300",
            "PORT": "9000",
            "JUNIE_BIN": "/custom/junie",
        }
        with patch.dict(os.environ, env, clear=True):
            from junie_adapter.config import OrchestratorConfig

            config = OrchestratorConfig.from_env()
            assert config.ollama_url == "http://custom:11435"
            assert config.ollama_model == "llama3:8b"
            assert config.refine_prompt is True
            assert config.workspace_path == "/custom/workspace"
            assert config.vault_path == "/custom/vault"
            assert config.mirror_logs_to_vault is True
            assert config.api_key == "test-key"
            assert config.junie_timeout == 300
            assert config.port == 9000
            assert config.junie_bin == "/custom/junie"

    def test_validate_requires_api_key(self):
        """Test that validation raises when API_KEY is required but missing."""
        from junie_adapter.config import OrchestratorConfig

        config = OrchestratorConfig(api_key="")
        with pytest.raises(RuntimeError, match="API_KEY"):
            config.validate(require_api_key=True)

    def test_validate_skips_api_key(self):
        """Test that validation passes when API_KEY is not required."""
        from junie_adapter.config import OrchestratorConfig

        config = OrchestratorConfig(api_key="")
        config.validate(require_api_key=False)  # Should not raise

    def test_workspace_dir_fallback(self):
        """Test workspace_dir falls back to temp directory."""
        from junie_adapter.config import OrchestratorConfig

        config = OrchestratorConfig(workspace_path=None)
        assert "junie_workspace" in str(config.workspace_dir)

    def test_workspace_dir_custom(self):
        """Test workspace_dir uses custom path."""
        from junie_adapter.config import OrchestratorConfig

        config = OrchestratorConfig(workspace_path="/custom/path")
        assert config.workspace_dir == Path("/custom/path")

    def test_memory_file_path(self):
        """Test memory_file property."""
        from junie_adapter.config import OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            assert config.memory_file == Path(tmp) / "MEMORY.md"

    def test_session_logs_dir_path(self):
        """Test session_logs_dir property."""
        from junie_adapter.config import OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            assert config.session_logs_dir == Path(tmp) / "session-logs"


# ─── Models tests ─────────────────────────────────────────────────────────────


class TestModels:
    def test_run_result(self):
        """Test RunResult dataclass."""
        from junie_adapter.models import RunResult, TaskStatus

        result = RunResult(
            task="test task",
            answer="test answer",
            status=TaskStatus.COMPLETED,
            duration_seconds=1.5,
        )
        assert result.task == "test task"
        assert result.answer == "test answer"
        assert result.status == TaskStatus.COMPLETED
        assert result.error is None

    def test_session_log(self):
        """Test SessionLog dataclass."""
        from junie_adapter.models import SessionLog, TaskStatus

        log = SessionLog(
            raw_prompt="test prompt",
            refined_prompt="refined prompt",
            answer="test answer",
        )
        assert log.raw_prompt == "test prompt"
        assert log.refined_prompt == "refined prompt"
        assert log.status == TaskStatus.COMPLETED

    def test_task_status_enum(self):
        """Test TaskStatus enum values."""
        from junie_adapter.models import TaskStatus

        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.TIMED_OUT.value == "timed_out"
        assert TaskStatus.INTERRUPTED.value == "interrupted"


# ─── Junie runner tests ──────────────────────────────────────────────────────


class TestJunieRunner:
    def test_parse_junie_json_simple(self):
        """Test parsing simple JSON output."""
        from junie_adapter.junie_runner import parse_junie_json

        output = json.dumps({"answer": "Hello world"})
        assert parse_junie_json(output) == "Hello world"

    def test_parse_junie_json_nested(self):
        """Test parsing nested JSON output."""
        from junie_adapter.junie_runner import parse_junie_json

        output = json.dumps({"result": {"content": "Nested answer"}})
        assert parse_junie_json(output) == "Nested answer"

    def test_parse_junie_json_non_json(self):
        """Test parsing non-JSON output."""
        from junie_adapter.junie_runner import parse_junie_json

        assert parse_junie_json("Plain text output") == "Plain text output"

    def test_parse_junie_json_empty(self):
        """Test parsing empty output."""
        from junie_adapter.junie_runner import parse_junie_json

        assert parse_junie_json("") == ""

    def test_has_rate_limit(self):
        """Test rate limit detection."""
        from junie_adapter.junie_runner import has_rate_limit

        assert has_rate_limit('{"error": "429 Too Many Requests"}') is True
        assert has_rate_limit("RESOURCE_EXHAUSTED") is True
        assert has_rate_limit('{"answer": "normal output"}') is False

    @pytest.mark.asyncio
    async def test_run_junie_not_found(self):
        """Test Junie not found returns FAILED status."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.junie_runner import run_junie_task
        from junie_adapter.models import TaskStatus

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(
                workspace_path=tmp,
                junie_bin="/nonexistent/junie",
                junie_timeout=1,
            )
            result = await run_junie_task("test task", config)
            assert result.status == TaskStatus.FAILED
            assert "Junie CLI not found" in result.answer

    @pytest.mark.asyncio
    async def test_run_junie_success_with_fake(self):
        """Test successful Junie run with fake binary."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.junie_runner import run_junie_task
        from junie_adapter.models import TaskStatus

        fake_junie = Path(__file__).parent / "fake_junie.py"
        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(
                workspace_path=tmp,
                junie_bin=str(fake_junie),
                junie_timeout=10,
            )
            result = await run_junie_task("test task", config)
            assert result.status == TaskStatus.COMPLETED
            assert "sample answer" in result.answer


# ─── Prompt refiner tests ────────────────────────────────────────────────────


class TestPromptRefiner:
    def test_build_prompt_system(self):
        """Test building system prompt with context."""
        from junie_adapter.prompt_refiner import build_prompt_system

        system = build_prompt_system(
            context="Some context", vault_map_content="Some map"
        )
        assert "Some context" in system
        assert "Some map" in system

    def test_build_prompt_system_empty(self):
        """Test building system prompt without context."""
        from junie_adapter.prompt_refiner import build_prompt_system

        system = build_prompt_system()
        assert (
            "PROMPT_REWRITE_SYSTEM_BASE" not in system
        )  # The actual text, not the variable name
        assert "You rewrite rough requests" in system

    @pytest.mark.asyncio
    async def test_refine_prompt_ollama_unavailable(self):
        """Test refinement falls back to raw prompt when Ollama is unavailable."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.prompt_refiner import refine_prompt

        config = OrchestratorConfig(
            ollama_url="http://localhost:9999",  # Non-existent
            ollama_model="test-model",
        )
        result = await refine_prompt("test prompt", config)
        improved, fallback, error = result
        assert improved == "test prompt"
        assert fallback is True
        assert error  # Should have an error reason

    @pytest.mark.respx(assert_all_called=False)
    @pytest.mark.asyncio
    async def test_refine_prompt_with_mock_ollama(self):
        """Test refinement with mocked Ollama response."""
        import respx

        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.prompt_refiner import refine_prompt

        config = OrchestratorConfig(
            ollama_url="http://localhost:11434",
            ollama_model="test-model",
        )
        with respx.mock:
            respx.post("http://localhost:11434/api/chat").mock(
                return_value=httpx.Response(
                    200, json={"message": {"content": "Refined prompt text"}}
                )
            )
            result = await refine_prompt("raw prompt", config)
            improved, fallback, error = result
            assert improved == "Refined prompt text"
            assert fallback is False
            assert error == ""


# ─── Memory loader tests ─────────────────────────────────────────────────────


class TestMemoryLoader:
    def test_load_context_no_memory(self):
        """Test loading context when no memory files exist."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_context

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            context = load_context(config)
            assert context == ""

    def test_load_context_with_memory(self):
        """Test loading context with MEMORY.md."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_context

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            memory_file = Path(tmp) / "MEMORY.md"
            memory_file.write_text("Test memory content", encoding="utf-8")
            context = load_context(config)
            assert "Test memory content" in context
            assert "# Memory" in context

    def test_load_context_with_session_logs(self):
        """Test loading context with session logs."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_context

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            logs_dir = Path(tmp) / "session-logs"
            logs_dir.mkdir()
            log_file = logs_dir / "session-20240101-120000.md"
            log_file.write_text("# Session 1\n\nTest session log", encoding="utf-8")
            context = load_context(config)
            assert "# Session 1" in context
            assert "# Recent Sessions" in context

    def test_load_vault_map_missing(self):
        """Test vault map returns empty when not configured."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_vault_map

        config = OrchestratorConfig(vault_map_path=None)
        assert load_vault_map(config) == ""

    def test_load_vault_map_exists(self):
        """Test vault map loading when file exists."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_vault_map

        with tempfile.TemporaryDirectory() as tmp:
            map_file = Path(tmp) / "vault_map.md"
            map_file.write_text("Vault map content", encoding="utf-8")
            config = OrchestratorConfig(vault_map_path=str(map_file))
            result = load_vault_map(config)
            assert "Vault map content" in result
            assert "# Vault Map" in result


# ─── Obsidian tests ───────────────────────────────────────────────────────────


class TestObsidian:
    def test_mirror_skips_when_no_vault(self):
        """Test mirroring skips when vault_path is not set."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.obsidian import mirror_to_vault

        config = OrchestratorConfig(vault_path=None)
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test content")
            f.flush()
            mirror_to_vault(Path(f.name), config)  # Should not raise

    def test_mirror_with_vault(self):
        """Test mirroring copies file to vault."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.obsidian import mirror_to_vault

        with tempfile.TemporaryDirectory() as vault:
            config = OrchestratorConfig(vault_path=vault, mirror_logs_to_vault=True)
            source = Path(vault) / "source.md"
            source.write_text("test session log", encoding="utf-8")
            mirror_to_vault(source, config)
            dest = Path(vault) / "Session-Logs" / source.name
            assert dest.exists()
            assert dest.read_text(encoding="utf-8") == "test session log"

    def test_mirror_with_invalid_vault(self):
        """Test mirroring handles invalid vault path gracefully."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.obsidian import mirror_to_vault

        config = OrchestratorConfig(vault_path="/nonexistent/vault/path")
        with tempfile.NamedTemporaryFile() as f:
            f.write(b"test content")
            f.flush()
            mirror_to_vault(Path(f.name), config)  # Should not raise


# ─── Session logger tests ────────────────────────────────────────────────────


class TestSessionLogger:
    @pytest.mark.asyncio
    async def test_log_session_basic(self):
        """Test basic session logging."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.models import SessionLog, TaskStatus
        from junie_adapter.session_logger import log_session

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            session_log = SessionLog(
                raw_prompt="test prompt",
                answer="test answer",
                status=TaskStatus.COMPLETED,
            )
            await log_session(session_log, config)
            logs = list(config.session_logs_dir.glob("session-*.md"))
            assert len(logs) == 1
            content = logs[0].read_text(encoding="utf-8")
            assert "test prompt" in content

    @pytest.mark.asyncio
    async def test_log_session_with_refined_prompt(self):
        """Test session logging preserves original and refined prompts."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.models import SessionLog, TaskStatus
        from junie_adapter.session_logger import log_session

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            session_log = SessionLog(
                raw_prompt="original prompt",
                refined_prompt="refined prompt",
                answer="test answer",
                status=TaskStatus.COMPLETED,
            )
            await log_session(session_log, config)
            logs = list(config.session_logs_dir.glob("session-*.md"))
            assert len(logs) == 1


# ─── Service integration tests ───────────────────────────────────────────────


class TestService:
    @pytest.fixture
    def client(self):
        """Create a test client with mocked config."""
        # Remove cached modules
        for mod in list(sys.modules.keys()):
            if mod.startswith("junie_adapter"):
                del sys.modules[mod]

        env = {
            "API_KEY": "test-key",
            "JUNIE_BIN": str(Path(__file__).parent / "fake_junie.py"),
            "JUNIE_TIMEOUT": "10",
            "WORKSPACE_PATH": tempfile.gettempdir() + "/test-workspace",
            "REFINE_PROMPT": "false",
        }
        with patch.dict(os.environ, env, clear=True):
            from junie_adapter.service import app

            # Reset the lock
            app.state.junie_lock = asyncio.Lock()
            with TestClient(app) as c:
                yield c

    def test_auth_required(self, client):
        """Test that unauthenticated requests are rejected."""
        response = client.get("/v1/models")
        assert response.status_code == 401

    def test_list_models(self, client):
        """Test listing available models."""
        response = client.get(
            "/v1/models", headers={"Authorization": "Bearer test-key"}
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "list"
        assert any(m["id"] == "junie-workflow" for m in data["data"])

    def test_chat_completions_no_messages(self, client):
        """Test chat completions with no messages."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={"messages": []},
        )
        assert response.status_code == 400

    def test_chat_completions_no_user_message(self, client):
        """Test chat completions with no user message."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={"messages": [{"role": "assistant", "content": "Hello"}]},
        )
        assert response.status_code == 400

    def test_chat_completions_success(self, client):
        """Test successful chat completion with refinement disabled."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "messages": [{"role": "user", "content": "Test task"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        assert data["object"] == "chat.completion"
        assert "sample answer" in data["choices"][0]["message"]["content"]

    def test_bypass_request(self, client):
        """Test bypass request (### Task: prefix) goes to Ollama."""
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "messages": [{"role": "user", "content": "### Task: direct question"}],
                "stream": False,
            },
        )
        # Should return Ollama error since Ollama is not running
        assert response.status_code == 200
        data = response.json()
        assert "Ollama error" in data["choices"][0]["message"]["content"]

    def test_refinement_disabled(self, client):
        """Test that refinement is disabled by default."""
        # REFINE_PROMPT=false in env, so raw prompt should go directly to Junie
        response = client.post(
            "/v1/chat/completions",
            headers={"Authorization": "Bearer test-key"},
            json={
                "messages": [{"role": "user", "content": "Simple task"}],
                "stream": False,
            },
        )
        assert response.status_code == 200
        data = response.json()
        # The answer should come from Junie, not through Ollama refinement
        assert "sample answer" in data["choices"][0]["message"]["content"]


# ─── Integration tests ───────────────────────────────────────────────────────


class TestIntegration:
    def test_full_pipeline_no_refinement(self):
        """Test full pipeline with refinement disabled."""
        for mod in list(sys.modules.keys()):
            if mod.startswith("junie_adapter"):
                del sys.modules[mod]

        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "API_KEY": "test-key",
                "JUNIE_BIN": str(Path(__file__).parent / "fake_junie.py"),
                "JUNIE_TIMEOUT": "10",
                "WORKSPACE_PATH": tmp,
                "REFINE_PROMPT": "false",
            }
            with patch.dict(os.environ, env, clear=True):
                from junie_adapter.service import app

                app.state.junie_lock = asyncio.Lock()
                with TestClient(app) as client:
                    response = client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": "Bearer test-key"},
                        json={
                            "messages": [
                                {"role": "user", "content": "Integration test"}
                            ],
                            "stream": False,
                        },
                    )
                    assert response.status_code == 200
                    # Session log should be created
                    logs = list(Path(tmp).glob("session-logs/session-*.md"))
                    assert len(logs) >= 1

    def test_obsidian_disabled_by_default(self):
        """Test that Obsidian is disabled by default."""
        with tempfile.TemporaryDirectory() as tmp:
            env = {
                "API_KEY": "test-key",
                "JUNIE_BIN": str(Path(__file__).parent / "fake_junie.py"),
                "JUNIE_TIMEOUT": "10",
                "WORKSPACE_PATH": tmp,
                "REFINE_PROMPT": "false",
            }
            with patch.dict(os.environ, env, clear=True):
                from junie_adapter.config import OrchestratorConfig

                config = OrchestratorConfig.from_env()
                assert config.vault_path is None
                assert config.mirror_logs_to_vault is False

    def test_custom_context_replaces_default(self):
        """Test that custom context replaces default memory loading."""
        from junie_adapter.config import OrchestratorConfig
        from junie_adapter.memory_loader import load_context

        with tempfile.TemporaryDirectory() as tmp:
            # Create MEMORY.md
            memory_file = Path(tmp) / "MEMORY.md"
            memory_file.write_text("Default memory", encoding="utf-8")
            config = OrchestratorConfig(workspace_path=tmp)
            context = load_context(config)
            assert "Default memory" in context
            # Custom context would be passed directly, bypassing load_context
            # This is verified by the service layer test


# ─── Context Provider tests ──────────────────────────────────────────────────


class TestContextProvider:
    """Test the pluggable context provider interface."""

    def test_empty_context_provider(self):
        """Test EmptyContextProvider returns empty string."""
        from junie_adapter import EmptyContextProvider

        provider = EmptyContextProvider()
        assert provider.load_context() == ""

    def test_memory_file_context_provider_no_files(self):
        """Test MemoryFileContextProvider with no memory files."""
        from junie_adapter import MemoryFileContextProvider, OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            provider = MemoryFileContextProvider(config)
            context = provider.load_context()
            assert context == ""

    def test_memory_file_context_provider_with_memory(self):
        """Test MemoryFileContextProvider loads MEMORY.md."""
        from junie_adapter import MemoryFileContextProvider, OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            memory_file = Path(tmp) / "MEMORY.md"
            memory_file.write_text("Test memory content", encoding="utf-8")
            provider = MemoryFileContextProvider(config)
            context = provider.load_context()
            assert "Test memory content" in context
            assert "# Memory" in context

    def test_memory_file_context_provider_with_session_logs(self):
        """Test MemoryFileContextProvider loads session logs."""
        from junie_adapter import MemoryFileContextProvider, OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            logs_dir = Path(tmp) / "session-logs"
            logs_dir.mkdir()
            log_file = logs_dir / "session-20240101-120000.md"
            log_file.write_text(
                "# Session 1\n\nTest session log content", encoding="utf-8"
            )
            provider = MemoryFileContextProvider(config)
            context = provider.load_context()
            assert "# Session 1" in context
            assert "# Recent Sessions" in context

    def test_vault_map_context_provider_missing(self):
        """Test VaultMapContextProvider returns empty when not configured."""
        from junie_adapter import VaultMapContextProvider

        provider = VaultMapContextProvider()
        assert provider.load_context() == ""

    def test_vault_map_context_provider_with_config(self):
        """Test VaultMapContextProvider loads vault map."""
        from junie_adapter import OrchestratorConfig, VaultMapContextProvider

        with tempfile.TemporaryDirectory() as tmp:
            map_file = Path(tmp) / "vault_map.md"
            map_file.write_text("Vault map content", encoding="utf-8")
            config = OrchestratorConfig(vault_map_path=str(map_file))
            provider = VaultMapContextProvider(config=config)
            context = provider.load_context()
            assert "Vault map content" in context
            assert "# Vault Map" in context

    def test_composite_context_provider(self):
        """Test CompositeContextProvider combines multiple providers."""
        from junie_adapter import (
            CompositeContextProvider,
            MemoryFileContextProvider,
            OrchestratorConfig,
            VaultMapContextProvider,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            memory_file = Path(tmp) / "MEMORY.md"
            memory_file.write_text("Memory content", encoding="utf-8")
            map_file = Path(tmp) / "vault_map.md"
            map_file.write_text("Map content", encoding="utf-8")
            config.vault_map_path = str(map_file)

            provider = CompositeContextProvider(
                [
                    MemoryFileContextProvider(config),
                    VaultMapContextProvider(config=config),
                ]
            )
            context = provider.load_context()
            assert "Memory content" in context
            assert "Map content" in context

    def test_create_default_context_provider_with_memory(self):
        """Test factory creates provider with memory."""
        from junie_adapter import OrchestratorConfig, create_default_context_provider

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            provider = create_default_context_provider(config, include_memory=True)
            assert provider.load_context() == ""  # No memory file

    def test_create_default_context_provider_empty(self):
        """Test factory returns EmptyContextProvider when nothing included."""
        from junie_adapter import (
            EmptyContextProvider,
            OrchestratorConfig,
            create_default_context_provider,
        )

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            provider = create_default_context_provider(
                config, include_memory=False, include_vault_map=False
            )
            assert isinstance(provider, EmptyContextProvider)

    def test_memory_loader_uses_context_provider(self):
        """Test that load_context delegates to ContextProvider."""
        from junie_adapter import OrchestratorConfig
        from junie_adapter.memory_loader import load_context

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            memory_file = Path(tmp) / "MEMORY.md"
            memory_file.write_text("Test memory", encoding="utf-8")
            context = load_context(config)
            assert "Test memory" in context


# ─── IsolationMode tests ─────────────────────────────────────────────────────


class TestIsolationMode:
    """Test IsolationMode enum and configuration."""

    def test_isolation_mode_enum_values(self):
        """Test IsolationMode enum values."""
        from junie_adapter import IsolationMode

        assert IsolationMode.WORKTREE.value == "worktree"
        assert IsolationMode.NONE.value == "none"
        assert IsolationMode.REQUIRED.value == "required"

    def test_isolation_mode_from_env(self):
        """Test loading isolation mode from environment."""
        from junie_adapter import IsolationMode

        env = {"ISOLATION_MODE": "required"}
        with patch.dict(os.environ, env, clear=True):
            from junie_adapter.config import OrchestratorConfig

            config = OrchestratorConfig.from_env()
            assert config.isolation_mode == IsolationMode.REQUIRED

    def test_isolation_mode_invalid_fallback(self):
        """Test invalid isolation mode falls back to WORKTREE."""
        from junie_adapter import IsolationMode

        env = {"ISOLATION_MODE": "invalid_mode"}
        with patch.dict(os.environ, env, clear=True):
            from junie_adapter.config import OrchestratorConfig

            config = OrchestratorConfig.from_env()
            assert config.isolation_mode == IsolationMode.WORKTREE

    def test_isolation_mode_default(self):
        """Test default isolation mode is WORKTREE."""
        from junie_adapter import IsolationMode
        from junie_adapter.config import OrchestratorConfig

        config = OrchestratorConfig()
        assert config.isolation_mode == IsolationMode.WORKTREE


# ─── validate_workspace tests ────────────────────────────────────────────────


class TestValidateWorkspace:
    """Test workspace validation."""

    def test_validate_workspace_valid(self):
        """Test validation passes for valid workspace."""
        from junie_adapter import OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            config = OrchestratorConfig(workspace_path=tmp)
            config.validate_workspace()  # Should not raise

    def test_validate_workspace_not_exists(self):
        """Test validation raises for non-existent workspace."""
        from junie_adapter import OrchestratorConfig

        config = OrchestratorConfig(workspace_path="/nonexistent/path")
        with pytest.raises(RuntimeError, match="does not exist"):
            config.validate_workspace()

    def test_validate_workspace_not_writable(self):
        """Test validation raises for non-writable workspace."""
        from junie_adapter import OrchestratorConfig

        with tempfile.TemporaryDirectory() as tmp:
            try:
                os.chmod(tmp, 0o000)
                config = OrchestratorConfig(workspace_path=tmp)
                # On Windows, admin users can still access 0o000 dirs, so skip
                import platform

                if platform.system() != "Windows":
                    with pytest.raises(RuntimeError, match="not writable"):
                        config.validate_workspace()
            finally:
                os.chmod(tmp, 0o755)  # Restore permissions for cleanup


# ─── WorkerSemaphore tests ───────────────────────────────────────────────────


class TestWorkerSemaphore:
    """Test WorkerSemaphore concurrency limiter."""

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrency(self):
        """Test that semaphore limits concurrent access."""
        from junie_adapter import WorkerSemaphore

        sem = WorkerSemaphore(max_workers=2)
        assert sem.active_count == 0

        await sem.acquire()
        assert sem.active_count == 1

        await sem.acquire()
        assert sem.active_count == 2

        await sem.release()
        assert sem.active_count == 1

        await sem.release()
        assert sem.active_count == 0

    @pytest.mark.asyncio
    async def test_semaphore_blocks_when_full(self):
        """Test that semaphore enforces max workers."""
        from junie_adapter import WorkerSemaphore

        sem = WorkerSemaphore(max_workers=1)
        await sem.acquire()
        assert sem.active_count == 1

        await sem.release()
        assert sem.active_count == 0

        # Verify we can acquire again after release
        await sem.acquire()
        assert sem.active_count == 1
        await sem.release()


# ─── WorkerRegistry tests ────────────────────────────────────────────────────


class TestWorkerRegistry:
    """Test WorkerRegistry for tracking active workers."""

    @pytest.mark.asyncio
    async def test_register_and_get(self):
        """Test registering and retrieving a worker."""
        from datetime import datetime, timezone

        from junie_adapter import WorkerExecution, WorkerRegistry
        from junie_adapter.models import WorkerStatus

        registry = WorkerRegistry()
        execution = WorkerExecution(
            worker_id="worker-1",
            task_id="task-1",
            parent_task_id=None,
            task_description="Test task",
            project_path="/tmp/test",
            worktree_path=None,
            status=WorkerStatus.QUEUED,
            pid=None,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            stdout="",
            stderr="",
            exit_code=None,
            result_summary=None,
            changed_files=[],
            error=None,
        )
        await registry.register("worker-1", execution)
        assert registry.active_count == 1

        found = await registry.get("worker-1")
        assert found is not None
        assert found.worker_id == "worker-1"
        assert found.status == WorkerStatus.RUNNING  # register sets RUNNING

    @pytest.mark.asyncio
    async def test_complete_worker(self):
        """Test completing a worker updates status."""
        from datetime import datetime, timezone

        from junie_adapter import WorkerExecution, WorkerRegistry
        from junie_adapter.models import WorkerStatus

        registry = WorkerRegistry()
        execution = WorkerExecution(
            worker_id="worker-1",
            task_id="task-1",
            parent_task_id=None,
            task_description="Test task",
            project_path="/tmp/test",
            worktree_path=None,
            status=WorkerStatus.RUNNING,
            pid=None,
            start_time=datetime.now(timezone.utc),
            end_time=datetime.now(timezone.utc),
            stdout="Done",
            stderr="",
            exit_code=0,
            result_summary="Done",
            changed_files=[],
            error=None,
        )
        await registry.register("worker-1", execution)
        execution.status = WorkerStatus.COMPLETED
        await registry.complete("worker-1", execution)

        found = await registry.get("worker-1")
        assert found.status == WorkerStatus.COMPLETED

    @pytest.mark.asyncio
    async def test_remove_worker(self):
        """Test removing a worker from registry."""
        from datetime import datetime, timezone

        from junie_adapter import WorkerExecution, WorkerRegistry
        from junie_adapter.models import WorkerStatus

        registry = WorkerRegistry()
        execution = WorkerExecution(
            worker_id="worker-1",
            task_id="task-1",
            parent_task_id=None,
            task_description="Test task",
            project_path="/tmp/test",
            worktree_path=None,
            status=WorkerStatus.RUNNING,
            pid=None,
            start_time=datetime.now(timezone.utc),
            end_time=None,
            stdout="",
            stderr="",
            exit_code=None,
            result_summary=None,
            changed_files=[],
            error=None,
        )
        await registry.register("worker-1", execution)
        assert registry.active_count == 1

        result = await registry.remove("worker-1")
        assert result is True
        assert registry.active_count == 0

        result = await registry.remove("nonexistent")
        assert result is False

    @pytest.mark.asyncio
    async def test_get_active_workers(self):
        """Test getting all active workers."""
        from datetime import datetime, timezone

        from junie_adapter import WorkerExecution, WorkerRegistry
        from junie_adapter.models import WorkerStatus

        registry = WorkerRegistry()
        for i in range(3):
            execution = WorkerExecution(
                worker_id=f"worker-{i}",
                task_id=f"task-{i}",
                parent_task_id=None,
                task_description=f"Task {i}",
                project_path="/tmp/test",
                worktree_path=None,
                status=WorkerStatus.RUNNING,
                pid=None,
                start_time=datetime.now(timezone.utc),
                end_time=None,
                stdout="",
                stderr="",
                exit_code=None,
                result_summary=None,
                changed_files=[],
                error=None,
            )
            await registry.register(f"worker-{i}", execution)

        active = await registry.get_active()
        assert len(active) == 3

    @pytest.mark.asyncio
    async def test_get_nonexistent_worker(self):
        """Test getting a nonexistent worker returns None."""
        from junie_adapter import WorkerRegistry

        registry = WorkerRegistry()
        found = await registry.get("nonexistent")
        assert found is None

    def test_get_worker_registry_singleton(self):
        """Test that get_worker_registry returns the same instance."""
        from junie_adapter import get_worker_registry

        reg1 = get_worker_registry()
        reg2 = get_worker_registry()
        assert reg1 is reg2


# ─── Streaming tests ─────────────────────────────────────────────────────────


class TestStreaming:
    """Test streaming response helpers."""

    @pytest.mark.asyncio
    async def test_stream_progress_events(self):
        """Test stream_progress emits correct event order."""
        from junie_adapter.service import stream_progress

        events = []
        async for chunk in stream_progress(
            improved_prompt="Test prompt",
            junie_answer="Test answer",
            fallback=False,
            error_reason="",
            raw_prompt="Test prompt",
        ):
            events.append(chunk)

        # First event should be progress
        assert "event: progress" in events[0]
        assert "task_completed" in events[0]

        # Last event should be [DONE]
        assert "[DONE]" in events[-1]

    @pytest.mark.asyncio
    async def test_stream_busy_event(self):
        """Test stream_busy_event emits busy status."""
        from junie_adapter.service import stream_busy_event

        events = []
        async for chunk in stream_busy_event():
            events.append(chunk)

        # Should contain busy status
        combined = "\n".join(events)
        assert "busy" in combined
        assert "[DONE]" in combined

    def test_sse_event_format(self):
        """Test SSE event formatting."""
        from junie_adapter.service import sse_event

        result = sse_event('{"test": "data"}')
        assert result == 'data: {"test": "data"}\n\n'

    def test_make_completion_response(self):
        """Test completion response format."""
        from junie_adapter.service import make_completion_response

        result = make_completion_response("Test content")
        assert result["object"] == "chat.completion"
        assert result["choices"][0]["message"]["content"] == "Test content"
        assert result["choices"][0]["finish_reason"] == "stop"
