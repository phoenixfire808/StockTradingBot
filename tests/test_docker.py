"""Tests for Docker / Windows-service infrastructure files.

Validates:
  * Dockerfile — required directives present, multi-stage, healthcheck, port,
    non-root user, slim base image.
  * .dockerignore — excludes heavy/runtime dirs (logs, data, venv, .git).
  * docker-compose.yml — service name, port mapping, volume mounts for
    logs/data/reports, env_file, healthcheck.
  * scripts/build-docker.sh + .ps1 — shebang / param block, image tag default.
  * scripts/run-docker.sh + .ps1 — supports the required modes
    (ui, dry-run, live, backtest) and mounts logs/data.

These tests are pure-string assertions against the files on disk — they
don't build images or invoke Docker. Run anywhere.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    """Read a repo-relative file."""
    p = REPO_ROOT / rel
    if not p.exists():
        pytest.skip(f"{rel} not present in repo")
    return p.read_text(encoding="utf-8")


# ── Dockerfile ──────────────────────────────────────────────────────────


class TestDockerfile:
    DOCKERFILE = _read("Dockerfile")

    def test_uses_python_slim_base(self):
        assert "FROM python:" in self.DOCKERFILE
        assert "-slim" in self.DOCKERFILE
        # Either literal pinned (3.X-slim) or ARG-driven (${PYTHON_VERSION}-slim).
        assert re.search(
            r"FROM python:(?:3\.\d+|\$\{PYTHON_VERSION\})-slim", self.DOCKERFILE
        ), "Dockerfile should pin a python:3.x-slim base image"

    def test_exposes_streamlit_port(self):
        assert "EXPOSE 8501" in self.DOCKERFILE

    def test_has_healthcheck(self):
        assert "HEALTHCHECK" in self.DOCKERFILE
        # Streamlit exposes /_stcore/health by default
        assert "/_stcore/health" in self.DOCKERFILE

    def test_runs_as_non_root(self):
        assert "USER app" in self.DOCKERFILE
        assert "useradd" in self.DOCKERFILE

    def test_uses_multi_stage_build(self):
        assert " AS builder" in self.DOCKERFILE or "AS runtime" in self.DOCKERFILE
        # Has at least two FROM instructions.
        assert len(re.findall(r"^FROM ", self.DOCKERFILE, re.MULTILINE)) >= 2

    def test_requirements_copied_and_installed(self):
        assert "COPY requirements.txt" in self.DOCKERFILE
        assert "pip install" in self.DOCKERFILE

    def test_logs_and_data_dirs_created(self):
        assert "/app/logs" in self.DOCKERFILE
        assert "/app/data" in self.DOCKERFILE

    def test_default_command_runs_dashboard(self):
        assert re.search(r'CMD\s*\[\s*"python"', self.DOCKERFILE), (
            "Default CMD should launch the bot (python main.py ui)"
        )
        assert "main.py" in self.DOCKERFILE


# ── .dockerignore ───────────────────────────────────────────────────────


class TestDockerignore:
    DOCKERIGNORE = _read(".dockerignore")

    def test_excludes_logs_dir(self):
        assert "logs" in self.DOCKERIGNORE

    def test_excludes_data_dir(self):
        assert "data" in self.DOCKERIGNORE

    def test_excludes_python_bytecode(self):
        assert "__pycache__" in self.DOCKERIGNORE
        assert ("*.pyc" in self.DOCKERIGNORE) or ("*.py[cod]" in self.DOCKERIGNORE)

    def test_excludes_git(self):
        assert ".git" in self.DOCKERIGNORE

    def test_excludes_env_file(self):
        assert re.search(r"^\.env$", self.DOCKERIGNORE, re.MULTILINE)
        assert ".env.example" in self.DOCKERIGNORE


class TestDockerCompose:
    COMPOSE = _read("docker-compose.yml")

    def test_service_defined(self):
        assert "services:" in self.COMPOSE
        assert "bot:" in self.COMPOSE

    def test_maps_port_8501(self):
        assert "8501:8501" in self.COMPOSE

    def test_mounts_logs_volume(self):
        assert re.search(r":/app/logs|:./logs", self.COMPOSE), (
            "docker-compose.yml must mount logs/"
        )

    def test_mounts_data_volume(self):
        assert re.search(r":/app/data|:./data", self.COMPOSE), (
            "docker-compose.yml must mount data/"
        )

    def test_uses_env_file(self):
        assert "env_file" in self.COMPOSE
        assert ".env" in self.COMPOSE

    def test_has_healthcheck(self):
        assert "healthcheck:" in self.COMPOSE
        assert "/_stcore/health" in self.COMPOSE


# ── install_service.ps1 ─────────────────────────────────────────────────


class TestInstallServiceScript:
    PS1 = _read("scripts/install_service.ps1")

    def test_runs_as_admin_marker(self):
        assert "#Requires -RunAsAdministrator" in self.PS1

    def test_uses_scheduled_task_xml(self):
        assert "Register-ScheduledTask" in self.PS1

    def test_launches_main_py_live(self):
        assert "main.py" in self.PS1
        assert "live" in self.PS1

    def test_sets_working_directory(self):
        assert "WorkingDirectory" in self.PS1 or "Set-Location" in self.PS1


class TestUninstallServiceScript:
    PS1 = _read("scripts/uninstall_service.ps1")

    def test_runs_as_admin_marker(self):
        assert "#Requires -RunAsAdministrator" in self.PS1

    def test_unregisters_task(self):
        assert "Unregister-ScheduledTask" in self.PS1


# ── Docker build scripts ────────────────────────────────────────────────


class TestBuildDockerScripts:
    SH = _read("scripts/build-docker.sh")
    PS1 = _read("scripts/build-docker.ps1")

    def test_sh_has_shebang(self):
        assert self.SH.startswith("#!/usr/bin/env bash")

    def test_sh_calls_docker_build(self):
        assert "docker build" in self.SH
        assert "stocktradingbot" in self.SH

    def test_ps1_accepts_tag_parameter(self):
        assert "-Tag" in self.PS1 or "Tag" in self.PS1
        assert "docker build" in self.PS1


# ── Docker run scripts ──────────────────────────────────────────────────


class TestRunDockerScripts:
    SH = _read("scripts/run-docker.sh")
    PS1 = _read("scripts/run-docker.ps1")

    @pytest.mark.parametrize("mode", ["dry-run", "live", "backtest", "ui"])
    def test_sh_supports_mode(self, mode):
        assert mode in self.SH, f"run-docker.sh must support mode {mode!r}"

    @pytest.mark.parametrize("mode", ["dry-run", "live", "backtest", "ui"])
    def test_ps1_supports_mode(self, mode):
        assert mode in self.PS1, f"run-docker.ps1 must support mode {mode!r}"

    def test_sh_mounts_logs(self):
        assert "logs:/app/logs" in self.SH or "/app/logs" in self.SH

    def test_sh_mounts_data(self):
        assert "data:/app/data" in self.SH or "/app/data" in self.SH

    def test_sh_exposes_port(self):
        assert "8501:8501" in self.SH