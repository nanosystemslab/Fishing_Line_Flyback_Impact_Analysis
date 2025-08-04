#!/usr/bin/env python3
"""Test runner script with coverage reporting."""

import argparse
import subprocess
import sys


def run_command(cmd, description=""):
    """Run a command and handle errors."""
    print(f"\n{'=' * 60}")
    print(f"Running: {description}")
    print(f"Command: {' '.join(cmd)}")
    print(f"{'=' * 60}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"ERROR: {e}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        return False


def main():  # noqa: C901
    """Main Function."""
    parser = argparse.ArgumentParser(description="Run tests with various options")
    parser.add_argument("--fast", action="store_true", help="Run only fast tests")
    parser.add_argument("--unit", action="store_true", help="Run only unit tests")
    parser.add_argument(
        "--integration", action="store_true", help="Run only integration tests"
    )
    parser.add_argument(
        "--performance", action="store_true", help="Run only performance tests"
    )
    parser.add_argument("--gui", action="store_true", help="Include GUI tests")
    parser.add_argument(
        "--visualization", action="store_true", help="Include visualization tests"
    )
    parser.add_argument(
        "--coverage", action="store_true", default=True, help="Generate coverage report"
    )
    parser.add_argument(
        "--no-coverage", action="store_true", help="Skip coverage reporting"
    )
    parser.add_argument(
        "--html", action="store_true", help="Generate HTML coverage report"
    )
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--parallel", "-n", type=int, help="Number of parallel workers")

    args = parser.parse_args()

    # Determine test selection
    test_markers = []
    test_files = []

    if args.fast:
        test_markers.append("not slow")

    if args.unit:
        test_markers.append("not integration")
        test_markers.append("not performance")
    elif args.integration:
        test_files.append("tests/test_integration.py")
    elif args.performance:
        test_files.append("tests/test_performance.py")

    if not args.gui:
        test_markers.append("not gui")

    if not args.visualization:
        test_markers.append("not visualization")

    # Build pytest command
    cmd = ["python", "-m", "pytest"]

    # Add test files or default to all tests
    if test_files:
        cmd.extend(test_files)
    else:
        cmd.append("tests/")

    # Add markers
    if test_markers:
        cmd.extend(["-m", " and ".join(test_markers)])

    # Add coverage options
    if args.coverage and not args.no_coverage:
        cmd.extend(
            [
                "--cov=src/Fishing_Line_Flyback_Impact_Analysis",
                "--cov-report=term-missing",
                "--cov-report=xml",
            ]
        )

        if args.html:
            cmd.append("--cov-report=html")

    # Add other options
    if args.verbose:
        cmd.append("-v")

    if args.parallel:
        cmd.extend(["-n", str(args.parallel)])

    # Add other useful options
    cmd.extend(
        [
            "--tb=short",  # Shorter traceback format
            "--strict-markers",  # Enforce marker registration
            "--disable-warnings",  # Reduce noise
        ]
    )

    # Run tests
    success = run_command(cmd, "Running tests")

    if not success:
        print("\n❌ Tests failed!")
        return 1

    # Generate additional reports if requested
    if args.coverage and not args.no_coverage:
        print(f"\n{'=' * 60}")
        print("Coverage Summary:")
        print(f"{'=' * 60}")

        # Show coverage report
        subprocess.run(
            ["python", "-m", "coverage", "report", "--skip-covered", "--show-missing"]
        )

        if args.html:
            print("\n📊 HTML coverage report generated in htmlcov/")

    print("\n✅ All tests passed!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
