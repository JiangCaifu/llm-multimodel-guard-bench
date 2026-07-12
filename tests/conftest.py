"""Pytest 公共夹具."""
from __future__ import annotations

import os
from pathlib import Path

import pytest


@pytest.fixture(scope="session")
def project_root() -> Path:
    """项目根目录."""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def data_dir(project_root: Path) -> Path:
    """评测数据目录."""
    return project_root / "data"


@pytest.fixture(scope="session")
def configs_dir(project_root: Path) -> Path:
    """配置文件目录."""
    return project_root / "configs"


@pytest.fixture(autouse=True)
def _setup_env(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """测试环境隔离 - 避免真实 API 调用."""
    monkeypatch.setenv("LLM_GUARD_BENCH_ENV", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv("LLM_GUARD_BENCH_RESULTS_DIR", str(tmp_path / "results"))
