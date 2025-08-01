"""Test cases for the __main__ module."""

import sys
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis import __main__


class TestMainModule:
    """Test cases for main module functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self._create_test_data_structure()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data_structure(self) -> None:
        """Create test data directory structure."""
        # Create directory structure
        csv_dir = Path(self.temp_dir) / "data" / "csv"
        csv_dir.mkdir(parents=True, exist_ok=True)

        # Create sample CSV file with Dewesoft format
        csv_file = csv_dir / "STND-21-1.csv"
        sample_data = (
            "Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),"
            "AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n"
            "0.0,0.0,0.0,0.0,0.0\n"
            "0.1,1.0,1.1,0.9,1.2\n"
            "0.2,2.0,2.1,1.9,2.2\n"
            "0.3,3.0,3.1,2.9,3.2\n"
            "0.4,2.5,2.6,2.4,2.7\n"
            "0.5,1.0,1.1,0.9,1.2"
        )

        with open(csv_file, "w") as f:
            f.write(sample_data)

        self.test_csv = str(csv_file)

        # Create additional test files for batch processing
        for config in ["DF", "BR"]:
            for i in range(2):
                csv_file = csv_dir / f"{config}-21-{i}.csv"
                with open(csv_file, "w") as f:
                    f.write(sample_data)

    def test_setup_logging(self) -> None:
        """Test logging setup."""
        __main__.setup_logging(20)  # INFO level
        # No exception should be raised

    def test_parse_command_line_analyze(self) -> None:
        """Test command line parsing for analyze command."""
        test_args = [
            "program",
            "analyze",
            "-i",
            "test.csv",
            "--param-y",
            "SUM",
            "--param-x",
            "time",
        ]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "analyze"
        assert args["input"] == ["test.csv"]
        assert args["param_y"] == "SUM"
        assert args["param_x"] == "time"

    def test_parse_command_line_postprocess(self) -> None:
        """Test command line parsing for postprocess command."""
        test_args = [
            "program",
            "postprocess",
            "-i",
            "results.txt",
            "--plot-type",
            "box",
            "--generate-table",
        ]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "postprocess"
        assert args["input"] == "results.txt"
        assert args["plot_type"] == "box"
        assert args["generate_table"] is True

    def test_parse_command_line_batch(self) -> None:
        """Test command line parsing for batch command."""
        test_args = ["program", "batch", "-d", "data", "--summary"]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "batch"
        assert args["data_dir"] == "data"
        assert args["summary"] is True

    def test_parse_command_line_table(self) -> None:
        """Test command line parsing for table command."""
        test_args = ["program", "table", "-i", "results.txt", "-o", "output"]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "table"
        assert args["input"] == "results.txt"
        assert args["output"] == "output"

    def test_parse_command_line_visualize(self) -> None:
        """Test command line parsing for visualize command."""
        test_args = [
            "program",
            "visualize",
            "-i",
            "output.csv",
            "--x-param",
            "D",
            "--y-param",
            "KE",
        ]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "visualize"
        assert args["input"] == ["output.csv"]
        assert args["x_param"] == "D"
        assert args["y_param"] == "KE"

    def test_handle_analyze_command_success(self) -> None:
        """Test successful analyze command handling."""
        args = {
            "input": [self.test_csv],
            "output": self.temp_dir,
            "param_y": "SUM",
            "param_x": "time",
            "show_all_sensors": False,
        }

        result = __main__.handle_analyze_command(args)
        assert result == 0

        # Check if results file was created
        results_file = Path(self.temp_dir) / "run_results.txt"
        assert results_file.exists()

    def test_handle_analyze_command_show_all_sensors(self) -> None:
        """Test analyze command with show all sensors option."""
        args = {
            "input": [self.test_csv],
            "output": self.temp_dir,
            "param_y": "All",
            "param_x": "time",
            "show_all_sensors": True,
        }

        result = __main__.handle_analyze_command(args)
        assert result == 0

    def test_handle_analyze_command_multiple_files(self) -> None:
        """Test analyze command with multiple files."""
        # Create second test file
        csv_dir = Path(self.temp_dir) / "data" / "csv"
        second_csv = csv_dir / "STND-21-2.csv"
        with open(second_csv, "w") as f:
            f.write("Time (s),AI 1/AI 1 (lbf),AI 2/AI 2 (lbf),")
            f.write("AI 3/AI 3 (lbf),AI 4/AI 4 (lbf)\n")
            f.write("0.0,0.0,0.0,0.0,0.0\n")
            f.write("0.1,1.5,1.6,1.4,1.7")

        args = {
            "input": [self.test_csv, str(second_csv)],
            "output": self.temp_dir,
            "param_y": "SUM",
            "param_x": "time",
            "show_all_sensors": False,
        }

        result = __main__.handle_analyze_command(args)
        assert result == 0

    def test_handle_analyze_command_error(self) -> None:
        """Test analyze command error handling."""
        args = {
            "input": ["nonexistent.csv"],
            "output": self.temp_dir,
            "param_y": "SUM",
            "param_x": "time",
            "show_all_sensors": False,
        }

        result = __main__.handle_analyze_command(args)
        assert result == 1

    def test_handle_postprocess_command_success(self) -> None:
        """Test successful postprocess command handling."""
        # First create a results file
        results_file = Path(self.temp_dir) / "results.txt"
        with open(results_file, "w") as f:
            f.write("STND-21-1.csv,J=2.450,F=1250.32\n")
            f.write("DF-21-1.csv,J=1.200,F=800.15\n")

        args = {
            "input": str(results_file),
            "output": self.temp_dir,
            "plot_type": "all",
            "generate_table": False,
        }

        result = __main__.handle_postprocess_command(args)
        assert result == 0

    def test_handle_postprocess_command_with_table(self) -> None:
        """Test postprocess command with table generation."""
        results_file = Path(self.temp_dir) / "results.txt"
        with open(results_file, "w") as f:
            f.write("STND-21-1.csv,J=2.450,F=1250.32\n")
            f.write("DF-21-1.csv,J=1.200,F=800.15\n")

        args = {
            "input": str(results_file),
            "output": self.temp_dir,
            "plot_type": "box",
            "generate_table": True,
        }

        with patch("builtins.print"):  # Suppress table output
            result = __main__.handle_postprocess_command(args)
        assert result == 0

    def test_handle_postprocess_command_error(self) -> None:
        """Test postprocess command error handling."""
        args = {
            "input": "nonexistent.txt",
            "output": self.temp_dir,
            "plot_type": "all",
            "generate_table": False,
        }

        result = __main__.handle_postprocess_command(args)
        assert result == 1

    def test_handle_batch_command_success(self) -> None:
        """Test successful batch command handling."""
        data_dir = Path(self.temp_dir) / "data" / "csv"

        args = {
            "data_dir": str(data_dir),
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
        assert result == 0

        # Check if batch results file was created
        batch_results = Path(self.temp_dir) / "batch_results.txt"
        assert batch_results.exists()

    def test_handle_batch_command_with_summary(self) -> None:
        """Test batch command with summary generation."""
        data_dir = Path(self.temp_dir) / "data" / "csv"

        args = {
            "data_dir": str(data_dir),
            "output": self.temp_dir,
            "summary": True,
        }

        with patch("builtins.print"):  # Suppress output
            result = __main__.handle_batch_command(args)
        assert result == 0

        # Check if summary files were created
        summary_csv = Path(self.temp_dir) / "results.csv"
        summary_report = Path(self.temp_dir) / "summary_report.txt"
        assert summary_csv.exists()
        assert summary_report.exists()

    def test_handle_batch_command_nonexistent_dir(self) -> None:
        """Test batch command with nonexistent directory."""
        args = {
            "data_dir": "/nonexistent/path",
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
        assert result == 1

    def test_handle_batch_command_no_files(self) -> None:
        """Test batch command with directory containing no data files."""
        empty_dir = Path(self.temp_dir) / "empty"
        empty_dir.mkdir()

        args = {
            "data_dir": str(empty_dir),
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
        assert result == 1

    def test_handle_table_command_success(self) -> None:
        """Test successful table command handling."""
        results_file = Path(self.temp_dir) / "results.txt"
        with open(results_file, "w") as f:
            f.write("STND-21-1.csv,J=2.450,F=1250.32\n")
            f.write("DF-21-1.csv,J=1.200,F=800.15\n")
            f.write("BR-21-1.csv,J=0.500,F=400.75\n")

        args = {
            "input": str(results_file),
            "output": self.temp_dir,
        }

        with patch("builtins.print"):  # Suppress table output
            result = __main__.handle_table_command(args)
        assert result == 0

    def test_handle_table_command_error(self) -> None:
        """Test table command error handling."""
        args = {
            "input": "nonexistent.txt",
            "output": self.temp_dir,
        }

        result = __main__.handle_table_command(args)
        assert result == 1

    def test_handle_visualize_command_success(self) -> None:
        """Test successful visualize command handling."""
        # Create output CSV file
        output_csv = Path(self.temp_dir) / "output.csv"
        test_data = pd.DataFrame(
            {"diameter": [21, 23, 25], "kinetic_energy": [0.1, 0.2, 0.3]}
        )
        test_data.to_csv(output_csv, header=False, index=False)

        args = {
            "input": [str(output_csv)],
            "output": self.temp_dir,
            "x_param": "D",
            "y_param": "KE",
        }

        result = __main__.handle_visualize_command(args)
        assert result == 0

    def test_handle_visualize_command_error(self) -> None:
        """Test visualize command error handling."""
        args = {
            "input": ["nonexistent.csv"],
            "output": self.temp_dir,
            "x_param": "D",
            "y_param": "KE",
        }

        result = __main__.handle_visualize_command(args)
        assert result == 1

    def test_main_success(self) -> None:
        """Test main function success path."""
        test_args = ["program", "analyze", "-i", self.test_csv, "-o", self.temp_dir]

        with patch.object(sys, "argv", test_args):
            result = __main__.main()

        assert result == 0

    def test_main_unknown_command(self) -> None:
        """Test main function with unknown command."""
        with patch(
            "Fishing_Line_Flyback_Impact_Analysis.__main__.parse_command_line"
        ) as mock_parse:
            mock_parse.return_value = {
                "command": "unknown",
                "output": self.temp_dir,
                "verbosity": 30,
            }

            result = __main__.main()
            assert result == 1

    def test_main_keyboard_interrupt(self) -> None:
        """Test main function keyboard interrupt handling."""
        with patch(
            "Fishing_Line_Flyback_Impact_Analysis.__main__.parse_command_line"
        ) as mock_parse:
            mock_parse.side_effect = KeyboardInterrupt()

            with pytest.raises(SystemExit) as exc_info:
                __main__.main()
            assert exc_info.value.code == 1

    def test_main_general_exception(self) -> None:
        """Test main function general exception handling."""
        with patch(
            "Fishing_Line_Flyback_Impact_Analysis.__main__.parse_command_line"
        ) as mock_parse:
            mock_parse.side_effect = Exception("Test exception")

            result = __main__.main()
            assert result == 1

    def test_verbose_logging(self) -> None:
        """Test verbose logging setup."""
        test_args = ["program", "-v", "analyze", "-i", self.test_csv]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()
            assert args["verbosity"] == 20  # INFO level

    def test_very_verbose_logging(self) -> None:
        """Test very verbose logging setup."""
        test_args = ["program", "-vv", "analyze", "-i", self.test_csv]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()
            assert args["verbosity"] == 10  # DEBUG level

    def test_version_display(self) -> None:
        """Test version display."""
        test_args = ["program", "--version"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                __main__.parse_command_line()

    def test_help_display(self) -> None:
        """Test help display."""
        test_args = ["program", "--help"]

        with patch.object(sys, "argv", test_args):
            with pytest.raises(SystemExit):
                __main__.parse_command_line()

    def test_command_handlers_mapping(self) -> None:
        """Test that all commands have handlers."""
        # This tests the command_handlers dictionary in main()
        expected_commands = ["analyze", "postprocess", "batch", "table", "visualize"]

        # Create mock args for each command
        for command in expected_commands:
            with patch(
                "Fishing_Line_Flyback_Impact_Analysis.__main__.parse_command_line"
            ) as mock_parse:
                mock_parse.return_value = {
                    "command": command,
                    "verbosity": 30,
                    "input": [self.test_csv] if command != "batch" else None,
                    "data_dir": self.temp_dir if command == "batch" else None,
                    "output": self.temp_dir,
                    "param_y": "SUM",
                    "param_x": "time",
                    "show_all_sensors": False,
                    "plot_type": "all",
                    "generate_table": False,
                    "summary": False,
                    "x_param": "D",
                    "y_param": "KE",
                }

                # Mock the actual handlers to avoid real execution
                with patch(
                    f"Fishing_Line_Flyback_Impact_Analysis.__main__.handle_{command}_command"
                ) as mock_handler:
                    mock_handler.return_value = 0
                    result = __main__.main()
                    assert result == 0
                    mock_handler.assert_called_once()


def test_main_direct_call(capfd: Any) -> None:
    """Test main function direct call behavior."""
    # This test is separate to match your existing test style
    with patch.object(sys, "argv", ["program", "--help"]):
        with pytest.raises(SystemExit):
            __main__.main()
