#!/usr/bin/env python3
"""
Modular test runner for multiverse_dive.

Quick shortcuts for running specific test suites.

Usage:
    ./run_tests.py                     # Run all tests
    ./run_tests.py flood               # Run flood tests only
    ./run_tests.py wildfire            # Run wildfire tests only
    ./run_tests.py storm               # Run storm tests only
    ./run_tests.py flood --quick       # Run flood tests, skip slow
    ./run_tests.py --algorithm sar     # Run tests matching 'sar'
    ./run_tests.py --file flood        # Run test files matching 'flood'
    ./run_tests.py schemas             # Run schema validation tests
    ./run_tests.py intent              # Run intent resolution tests
    ./run_tests.py providers           # Run data provider tests
    ./run_tests.py --list              # List available test categories
"""

import argparse
import subprocess
import sys
from pathlib import Path


# Test categories with their pytest selectors
CATEGORIES = {
    # Hazard types
    "flood": {"marker": "flood", "desc": "Flood detection algorithms"},
    "wildfire": {"marker": "wildfire", "desc": "Wildfire/burn severity algorithms"},
    "storm": {"marker": "storm", "desc": "Storm damage algorithms"},

    # Components
    "schemas": {"marker": "schema", "desc": "JSON schema validation"},
    "intent": {"marker": "intent", "desc": "Intent resolution and classification"},
    "providers": {"marker": "provider", "desc": "Data provider implementations"},
    "registry": {"file": "test_algorithm_registry.py", "desc": "Algorithm registry"},

    # Test types
    "quick": {"marker": "not slow", "desc": "Fast tests only"},
    "slow": {"marker": "slow", "desc": "Slow/comprehensive tests"},
    "integration": {"marker": "integration", "desc": "Integration tests"},

    # All algorithms
    "algorithms": {"file": "test_*algorithms*.py", "desc": "All algorithm tests"},
}

# Algorithm shortcuts
ALGORITHMS = {
    "sar": "threshold_sar",
    "ndwi": "ndwi_optical",
    "change": "change_detection",
    "hand": "hand_model",
    "dnbr": "dnbr or nbr",
    "thermal": "thermal",
    "wind": "wind_damage",
    "structural": "structural_damage",
}


def list_categories():
    """Print available test categories."""
    print("\n📋 Available Test Categories:\n")

    print("  Hazard Types:")
    for name in ["flood", "wildfire", "storm"]:
        print(f"    {name:12} - {CATEGORIES[name]['desc']}")

    print("\n  Components:")
    for name in ["schemas", "intent", "providers", "registry"]:
        print(f"    {name:12} - {CATEGORIES[name]['desc']}")

    print("\n  Test Types:")
    for name in ["quick", "slow", "algorithms"]:
        print(f"    {name:12} - {CATEGORIES[name]['desc']}")

    print("\n  Algorithm Shortcuts (use with --algorithm):")
    for short, full in ALGORITHMS.items():
        print(f"    {short:12} - {full}")

    print("\n  Examples:")
    print("    ./run_tests.py flood")
    print("    ./run_tests.py wildfire --quick")
    print("    ./run_tests.py --algorithm sar")
    print("    ./run_tests.py storm -v")
    print()


def build_pytest_args(args):
    """Build pytest command arguments."""
    # Use venv pytest if available
    project_root = Path(__file__).parent
    venv_pytest = project_root / ".venv" / "bin" / "pytest"

    if venv_pytest.exists():
        pytest_args = [str(venv_pytest)]
    else:
        pytest_args = [sys.executable, "-m", "pytest"]

    # Add test directory
    tests_dir = project_root / "tests"

    markers = []
    file_patterns = []
    keyword = None

    # Process category
    if args.category:
        cat = args.category.lower()
        if cat in CATEGORIES:
            cat_info = CATEGORIES[cat]
            if "marker" in cat_info:
                markers.append(cat_info["marker"])
            if "file" in cat_info:
                file_patterns.append(cat_info["file"])
        else:
            # Try as a file pattern
            file_patterns.append(f"*{cat}*")

    # Process --quick flag
    if args.quick:
        markers.append("not slow")

    # Process --algorithm flag
    if args.algorithm:
        algo = args.algorithm.lower()
        # Expand shortcuts
        if algo in ALGORITHMS:
            keyword = ALGORITHMS[algo]
        else:
            keyword = algo

    # Process --file flag
    if args.file:
        file_patterns.append(f"*{args.file}*")

    # Build marker expression
    if markers:
        marker_expr = " and ".join(f"({m})" for m in markers)
        pytest_args.extend(["-m", marker_expr])

    # Build keyword expression
    if keyword:
        pytest_args.extend(["-k", keyword])

    # Add file patterns or default to tests/
    if file_patterns:
        for pattern in file_patterns:
            if "*" in pattern:
                # Glob pattern
                matching = list(tests_dir.glob(pattern))
                pytest_args.extend(str(f) for f in matching)
            else:
                pytest_args.append(str(tests_dir / pattern))
    else:
        pytest_args.append(str(tests_dir))

    # Add verbosity
    if args.verbose:
        pytest_args.append("-v")
    else:
        pytest_args.append("-v")  # Default to verbose

    # Add any extra pytest args
    if args.pytest_args:
        pytest_args.extend(args.pytest_args)

    # Show short summary
    pytest_args.append("--tb=short")

    return pytest_args


def main():
    parser = argparse.ArgumentParser(
        description="Modular test runner for multiverse_dive",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s flood              Run flood tests
  %(prog)s wildfire --quick   Run fast wildfire tests
  %(prog)s --algorithm sar    Run SAR algorithm tests
  %(prog)s schemas            Run schema tests
  %(prog)s --list             Show all categories
        """
    )

    parser.add_argument(
        "category",
        nargs="?",
        help="Test category: flood, wildfire, storm, schemas, intent, providers, algorithms"
    )
    parser.add_argument(
        "--quick", "-q",
        action="store_true",
        help="Skip slow tests"
    )
    parser.add_argument(
        "--algorithm", "-a",
        help="Filter by algorithm name (e.g., sar, ndwi, hand)"
    )
    parser.add_argument(
        "--file", "-f",
        help="Filter by test file name pattern"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    parser.add_argument(
        "--list", "-l",
        action="store_true",
        help="List available test categories"
    )
    parser.add_argument(
        "pytest_args",
        nargs="*",
        help="Additional arguments to pass to pytest"
    )

    args = parser.parse_args()

    if args.list:
        list_categories()
        return 0

    # Set PYTHONPATH
    import os
    project_root = Path(__file__).parent
    os.environ["PYTHONPATH"] = str(project_root)

    # Build and run pytest
    pytest_args = build_pytest_args(args)

    print(f"🧪 Running: {' '.join(pytest_args[2:])}\n")

    result = subprocess.run(pytest_args)
    return result.returncode


if __name__ == "__main__":
    sys.exit(main())
