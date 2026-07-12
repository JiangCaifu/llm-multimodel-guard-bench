"""验证项目初始化是否正确."""
from __future__ import annotations

import sys
from pathlib import Path


def test_package_importable() -> None:
    """验证包可正常导入."""
    import llm_guard_bench

    assert llm_guard_bench.__version__ == "0.1.0"


def test_constants_exist() -> None:
    """验证常量模块可访问."""
    from llm_guard_bench.constants import (
        CONFIGS_DIR,
        DATA_DIR,
        DEFAULT_TEMPERATURE,
        EvalDimension,
        SeverityLevel,
    )

    assert DEFAULT_TEMPERATURE == 0.0
    assert SeverityLevel.P0.value == "P0"
    assert EvalDimension.SAFETY.value == "safety"
    assert CONFIGS_DIR.exists()
    assert DATA_DIR.exists()


def test_cli_parser_buildable() -> None:
    """验证 CLI 参数解析器可构建."""
    from llm_guard_bench.cli import build_parser

    parser = build_parser()
    args = parser.parse_args(["run", "--config", "test.yaml"])
    assert args.command == "run"
    assert args.config == "test.yaml"


def test_project_layout(project_root: Path) -> None:
    """验证项目目录结构."""
    assert (project_root / "pyproject.toml").exists()
    assert (project_root / "src" / "llm_guard_bench").exists()
    assert (project_root / "configs" / "models").exists()
    assert (project_root / "configs" / "datasets").exists()
    assert (project_root / "tests").exists()


def test_python_version_supported() -> None:
    """验证 Python 版本."""
    assert sys.version_info >= (3, 9), "项目要求 Python 3.9+"
