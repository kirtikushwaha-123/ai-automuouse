"""Shared pytest fixtures for RootCause AI tests."""

from pathlib import Path
import subprocess
import pytest


@pytest.fixture
def temp_git_repo(tmp_path: Path) -> Path:
    """Create a real, temporary Git repository with sample code and tests."""
    repo_dir = tmp_path / "sample_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    # Initialize git repo
    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

    # Create app source code
    app_dir = repo_dir / "src" / "app"
    app_dir.mkdir(parents=True, exist_ok=True)
    calc_py = app_dir / "calculator.py"
    calc_py.write_text(
        "def add(a: int, b: int) -> int:\n"
        "    return a + b\n\n"
        "def divide(a: int, b: int) -> float:\n"
        "    if b == 0:\n"
        "        raise ZeroDivisionError('division by zero')\n"
        "    return a / b\n",
        encoding="utf-8",
    )

    # Create tests
    tests_dir = repo_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_calc = tests_dir / "test_calculator.py"
    test_calc.write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
        "from app.calculator import add, divide\n\n"
        "def test_add():\n"
        "    assert add(2, 3) == 5\n\n"
        "def test_divide():\n"
        "    assert divide(10, 2) == 5.0\n",
        encoding="utf-8",
    )

    # Initial commit
    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Initial commit: add calculator and tests"], cwd=str(repo_dir), check=True, capture_output=True)

    return repo_dir


@pytest.fixture
def failing_git_repo(tmp_path: Path) -> Path:
    """Create a temporary Git repository with a failing test."""
    repo_dir = tmp_path / "failing_repo"
    repo_dir.mkdir(parents=True, exist_ok=True)

    subprocess.run(["git", "init"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.name", "Test Runner"], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "config", "user.email", "test@example.com"], cwd=str(repo_dir), check=True, capture_output=True)

    tests_dir = repo_dir / "tests"
    tests_dir.mkdir(parents=True, exist_ok=True)
    test_file = tests_dir / "test_bug.py"
    test_file.write_text(
        "def test_failing_case():\n"
        "    assert 1 == 2, 'Expected 1 to equal 2'\n",
        encoding="utf-8",
    )

    subprocess.run(["git", "add", "."], cwd=str(repo_dir), check=True, capture_output=True)
    subprocess.run(["git", "commit", "-m", "Add failing test"], cwd=str(repo_dir), check=True, capture_output=True)

    return repo_dir
