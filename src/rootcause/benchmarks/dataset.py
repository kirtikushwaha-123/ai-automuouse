"""Curated dataset registry of reproducible RootCause AI benchmark scenarios."""

from pathlib import Path
from typing import Any, Dict, List, Optional

from rootcause.benchmarks.fixtures import (
    create_fixture_repository,
    validate_fixture,
)
from rootcause.benchmarks.models import (
    BenchmarkCase,
    BenchmarkCategory,
    BenchmarkDifficulty,
)
from rootcause.core.models import InvestigationStatus

# =============================================================================
# Curated Benchmark Case Definitions
# =============================================================================

BENCHMARK_CASES: Dict[str, BenchmarkCase] = {
    "BM01_CONDITIONAL_LOGIC": BenchmarkCase(
        case_id="BM01_CONDITIONAL_LOGIC",
        name="Cart Discount Threshold Comparison",
        category=BenchmarkCategory.CONDITIONAL_LOGIC,
        difficulty=BenchmarkDifficulty.EASY,
        bug_description=(
            "calculate_discount does not apply discount when cart total is exactly equal to 100.0. "
            "Policy requires orders of $100.00 or more to receive discount, but code uses strictly greater."
        ),
        stack_trace="AssertionError: assert 0.0 == 10.0 in discount.py",
        known_root_cause=(
            "In calculate_discount in src/discount.py, the comparison uses strictly greater '>' "
            "instead of greater-than-or-equal '>=' against the 100.0 threshold."
        ),
        expected_root_cause_keywords=["discount", "threshold", "greater"],
        expected_symptom="AssertionError: assert 0.0 == 10.0",
        relevant_files=["src/discount.py"],
        expected_verification_outcome=InvestigationStatus.PATCH_VERIFIED,
        fixture_relative_dir="BM01_CONDITIONAL_LOGIC",
        known_good_patch=(
            "--- a/src/discount.py\n"
            "+++ b/src/discount.py\n"
            "@@ -4,3 +4,3 @@\n"
            "     \"\"\"\n"
            "-    if total > 100.0:\n"
            "+    if total >= 100.0:\n"
            "         return round(total * 0.10, 2)\n"
        ),
        regression_test_file="tests/test_reproduction.py",
        regression_test_code=(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
            "from discount import calculate_discount\n\n"
            "def test_threshold_order_gets_discount():\n"
            "    # Exactly $100.00 must qualify for $10.00 discount\n"
            "    assert calculate_discount(100.0) == 10.0, 'calculate_discount discount threshold policy check'\n"
        ),
    ),
    "BM02_BOUNDARY_OFF_BY_ONE": BenchmarkCase(
        case_id="BM02_BOUNDARY_OFF_BY_ONE",
        name="Pagination Slice Upper Bound Off-By-One",
        category=BenchmarkCategory.BOUNDARY_CONDITION,
        difficulty=BenchmarkDifficulty.EASY,
        bug_description=(
            "paginate_items omits the last item on a page due to an off-by-one slice calculation. "
            "A page of size 5 on a list of 5 items returns only 4 items."
        ),
        stack_trace="AssertionError: assert 4 == 5 in pagination.py",
        known_root_cause=(
            "In paginate_items in src/pagination.py, end_idx - 1 is used in slice instead of end_idx, "
            "omitting the final item on each page."
        ),
        expected_root_cause_keywords=["pagination", "slice", "boundary"],
        expected_symptom="AssertionError: assert 4 == 5",
        relevant_files=["src/pagination.py"],
        expected_verification_outcome=InvestigationStatus.PATCH_VERIFIED,
        fixture_relative_dir="BM02_BOUNDARY_OFF_BY_ONE",
        known_good_patch=(
            "--- a/src/pagination.py\n"
            "+++ b/src/pagination.py\n"
            "@@ -9,3 +9,3 @@\n"
            "     end_idx = start_idx + page_size\n"
            "     # Off-by-one bug: subtracting 1 omits the last item on full page\n"
            "-    return items[start_idx : end_idx - 1]\n"
            "+    return items[start_idx : end_idx]\n"
        ),
        regression_test_file="tests/test_reproduction.py",
        regression_test_code=(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
            "from pagination import paginate_items\n\n"
            "def test_full_page_returns_all_items():\n"
            "    items = ['a', 'b', 'c', 'd', 'e']\n"
            "    page = paginate_items(items, page=1, page_size=5)\n"
            "    assert len(page) == 5, f'paginate_items boundary slice error: {page}'\n"
            "    assert page == items\n"
        ),
    ),
    "BM03_DATA_HANDLING_NONE": BenchmarkCase(
        case_id="BM03_DATA_HANDLING_NONE",
        name="Profile Formatting Unhandled None AttributeError",
        category=BenchmarkCategory.DATA_HANDLING,
        difficulty=BenchmarkDifficulty.MEDIUM,
        bug_description=(
            "format_user_bio crashes with AttributeError when profile contains bio=None. "
            "Calling .strip() on None raises AttributeError."
        ),
        stack_trace="AttributeError: 'NoneType' object has no attribute 'strip' in profile.py",
        known_root_cause=(
            "In format_user_bio in src/profile.py, bio is stripped directly without checking if "
            "it is None or empty, causing AttributeError on NoneType."
        ),
        expected_root_cause_keywords=["profile", "none", "attributeerror"],
        expected_symptom="AttributeError: 'NoneType' object has no attribute 'strip'",
        relevant_files=["src/profile.py"],
        expected_verification_outcome=InvestigationStatus.PATCH_VERIFIED,
        fixture_relative_dir="BM03_DATA_HANDLING_NONE",
        known_good_patch=(
            "--- a/src/profile.py\n"
            "+++ b/src/profile.py\n"
            "@@ -6,3 +6,5 @@\n"
            "     bio = profile.get(\"bio\")\n"
            "     # Unhandled None: calling .strip() on None raises AttributeError\n"
            "-    return bio.strip().title()\n"
            "+    if not bio:\n"
            "+        return \"\"\n"
            "+    return bio.strip().title()\n"
        ),
        regression_test_file="tests/test_reproduction.py",
        regression_test_code=(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
            "from profile import format_user_bio\n\n"
            "def test_missing_or_none_bio():\n"
            "    profile = {'username': 'bob', 'bio': None}\n"
            "    assert format_user_bio(profile) == ''\n"
        ),
    ),
    "BM04_CODE_REGRESSION": BenchmarkCase(
        case_id="BM04_CODE_REGRESSION",
        name="Refactoring Return Type Regression in Title Formatter",
        category=BenchmarkCategory.REGRESSION,
        difficulty=BenchmarkDifficulty.MEDIUM,
        bug_description=(
            "format_title crashes with AttributeError after commit refactored sanitize_input to "
            "return a tuple (cleaned, modified). format_title still expects a string."
        ),
        stack_trace="AttributeError: 'tuple' object has no attribute 'capitalize' in formatter.py",
        known_root_cause=(
            "Commit refactored sanitize_input return type from str to Tuple[str, bool], but caller "
            "format_title in src/formatter.py was not updated to unpack the tuple."
        ),
        expected_root_cause_keywords=["formatter", "sanitize", "tuple"],
        expected_symptom="AttributeError: 'tuple' object has no attribute 'capitalize'",
        relevant_files=["src/formatter.py"],
        expected_verification_outcome=InvestigationStatus.PATCH_VERIFIED,
        fixture_relative_dir="BM04_CODE_REGRESSION",
        known_good_patch=(
            "--- a/src/formatter.py\n"
            "+++ b/src/formatter.py\n"
            "@@ -9,4 +9,4 @@\n"
            " def format_title(text: str) -> str:\n"
            "     # Regression: sanitize_input now returns tuple, but format_title expects str\n"
            "-    cleaned = sanitize_input(text)\n"
            "+    cleaned, _ = sanitize_input(text)\n"
            "     return f\"Title: {cleaned.capitalize()}\"\n"
        ),
        regression_test_file="tests/test_reproduction.py",
        regression_test_code=(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
            "from formatter import format_title\n\n"
            "def test_format_title_regression():\n"
            "    res = format_title('  welcome home  ')\n"
            "    assert res == 'Title: Welcome home'\n"
        ),
    ),
    "BM05_ZERO_DIVISION": BenchmarkCase(
        case_id="BM05_ZERO_DIVISION",
        name="Statistics Calculator ZeroDivisionError on Empty List",
        category=BenchmarkCategory.NUMERIC_ERROR,
        difficulty=BenchmarkDifficulty.EASY,
        bug_description=(
            "calculate_average crashes with ZeroDivisionError when passed an empty list. "
            "Code directly divides sum(scores) by len(scores) without checking if length is zero."
        ),
        stack_trace="ZeroDivisionError: division by zero in stats.py",
        known_root_cause=(
            "calculate_average in src/stats.py lacks an empty-list guard, causing ZeroDivisionError "
            "when dividing by len(scores) which is 0."
        ),
        expected_root_cause_keywords=["average", "empty", "zerodivisionerror"],
        expected_symptom="ZeroDivisionError: division by zero",
        relevant_files=["src/stats.py"],
        expected_verification_outcome=InvestigationStatus.PATCH_VERIFIED,
        fixture_relative_dir="BM05_ZERO_DIVISION",
        known_good_patch=(
            "--- a/src/stats.py\n"
            "+++ b/src/stats.py\n"
            "@@ -6,4 +6,6 @@\n"
            "     Returns 0.0 when score list is empty.\n"
            "     \"\"\"\n"
            "     # Bug: direct division without empty-list check raises ZeroDivisionError\n"
            "+    if not scores:\n"
            "+        return 0.0\n"
            "     return round(sum(scores) / len(scores), 2)\n"
        ),
        regression_test_file="tests/test_reproduction.py",
        regression_test_code=(
            "import sys\n"
            "from pathlib import Path\n"
            "sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))\n"
            "from stats import calculate_average\n\n"
            "def test_average_empty_list():\n"
            "    assert calculate_average([]) == 0.0\n"
        ),
    ),
}


def get_benchmark_cases() -> Dict[str, BenchmarkCase]:
    """Retrieve all registered benchmark cases in the suite."""
    return dict(BENCHMARK_CASES)


def get_benchmark_case(case_id: str) -> BenchmarkCase:
    """Retrieve a single benchmark case by ID.
    
    Raises:
        KeyError: If case_id is not registered.
    """
    if case_id not in BENCHMARK_CASES:
        raise KeyError(f"Benchmark case '{case_id}' is not registered in dataset.")
    return BENCHMARK_CASES[case_id]


def validate_all_fixtures(fixtures_root: Path) -> Dict[str, Dict[str, Any]]:
    """Validate all registered fixtures against acceptance criteria.
    
    Args:
        fixtures_root: Base directory where fixtures are instantiated.
        
    Returns:
        Mapping of case_id -> validation result dictionary.
    """
    results: Dict[str, Dict[str, Any]] = {}
    for case_id, case in BENCHMARK_CASES.items():
        repo_dir = fixtures_root / case.fixture_relative_dir
        create_fixture_repository(case_id, repo_dir)
        val_res = validate_fixture(
            case_id=case_id,
            repo_path=repo_dir,
            bug_description=case.bug_description,
            stack_trace=case.stack_trace,
            regression_test_file=case.regression_test_file,
            regression_test_code=case.regression_test_code,
            known_good_patch=case.known_good_patch,
        )
        results[case_id] = val_res
    return results
