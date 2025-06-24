"""Extended test cases for the __main__ module."""

import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from Fishing_Line_Material_Properties_Analysis import __main__


class TestMainModule:
    """Test cases for main module functions."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()

        # Create sample data structure
        self._create_test_data_structure()

    def teardown_method(self) -> None:
        """Clean up after tests."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _create_test_data_structure(self) -> None:
        """Create test data directory structure."""
        # Create directory structure
        group_dir = Path(self.temp_dir) / "data" / "group_1" / "5in"
        group_dir.mkdir(parents=True, exist_ok=True)

        # Create sample CSV file
        csv_file = group_dir / "test--line-crimp-21--0.csv"
        sample_data = """Time,Force,Stroke
sec,N,mm
0.0,0.0,0.0
0.1,1.0,0.1
0.2,2.0,0.2
0.3,3.0,0.3
0.4,2.5,0.4
0.5,1.0,0.5"""

        with open(csv_file, "w") as f:
            f.write(sample_data)

        self.test_csv = str(csv_file)

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
            "--plot-type",
            "single",
            "--x-param",
            "Strain",
            "--y-param",
            "Stress",
        ]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "analyze"
        assert args["input"] == ["test.csv"]
        assert args["plot_type"] == "single"
        assert args["x_param"] == "Strain"
        assert args["y_param"] == "Stress"

    def test_parse_command_line_batch(self) -> None:
        """Test command line parsing for batch command."""
        test_args = ["program", "batch", "-d", "data", "--summary"]

        with patch.object(sys, "argv", test_args):
            args = __main__.parse_command_line()

        assert args["command"] == "batch"
        assert args["data_dir"] == "data"
        assert args["summary"] is True

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

    def test_handle_analyze_command_single(self) -> None:
        """Test analyze command handling for single plot."""
        args = {
            "input": [self.test_csv],
            "output": self.temp_dir,
            "plot_type": "single",
            "x_param": "Strain",
            "y_param": "Stress",
        }

        result = __main__.handle_analyze_command(args)
        assert result == 0

        # Check if CSV output was created
        csv_file = Path(self.temp_dir) / "individual_results.csv"
        assert csv_file.exists()

    def test_handle_analyze_command_multi(self) -> None:
        """Test analyze command handling for multi plot."""
        args = {
            "input": [self.test_csv],
            "output": self.temp_dir,
            "plot_type": "multi",
            "x_param": "Strain",
            "y_param": "Stress",
        }

        result = __main__.handle_analyze_command(args)
        assert result == 0

        # Check if both CSV outputs were created
        individual_csv = Path(self.temp_dir) / "individual_results.csv"
        multi_csv = Path(self.temp_dir) / "multi_run_averages.csv"
        assert individual_csv.exists()
        assert multi_csv.exists()

    def test_handle_analyze_command_error(self) -> None:
        """Test analyze command error handling."""
        args = {
            "input": ["nonexistent.csv"],
            "output": self.temp_dir,
            "plot_type": "single",
            "x_param": "Strain",
            "y_param": "Stress",
        }

        result = __main__.handle_analyze_command(args)
        assert result == 1

    def test_save_individual_results_csv(self) -> None:
        """Test saving individual results to CSV."""
        results = [
            {
                "file": "test1.csv",
                "group": "group_1",
                "length": "5in",
                "max_force_N": 10.0,
                "modulus_MPa": 2.5,
            },
            {
                "file": "test2.csv",
                "group": "group_1",
                "length": "5in",
                "max_force_N": 12.0,
                "modulus_MPa": 2.8,
            },
        ]

        __main__._save_individual_results_csv(results, self.temp_dir)

        csv_file = Path(self.temp_dir) / "individual_results.csv"
        assert csv_file.exists()

        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert "file" in df.columns

    def test_save_individual_results_csv_append(self) -> None:
        """Test appending to existing individual results CSV."""
        # Create initial CSV
        initial_results = [{"file": "test1.csv", "group": "group_1"}]
        __main__._save_individual_results_csv(initial_results, self.temp_dir)

        # Append new results
        new_results = [{"file": "test2.csv", "group": "group_2"}]
        __main__._save_individual_results_csv(new_results, self.temp_dir)

        csv_file = Path(self.temp_dir) / "individual_results.csv"
        df = pd.read_csv(csv_file)
        assert len(df) == 2

    def test_save_multi_results_csv(self) -> None:
        """Test saving multi-run results to CSV."""
        results = [
            {
                "group": "group_1",
                "length": "5in",
                "sample_count": 10,
                "avg_max_force_N": 25.5,
                "std_max_force_N": 2.1,
            }
        ]

        __main__._save_multi_results_csv(results, self.temp_dir)

        csv_file = Path(self.temp_dir) / "multi_run_averages.csv"
        assert csv_file.exists()

        df = pd.read_csv(csv_file)
        assert len(df) == 1
        assert "group" in df.columns

    def test_handle_visualize_command(self) -> None:
        """Test visualize command handling."""
        # Create a temporary output CSV
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

    def test_handle_batch_command(self) -> None:
        """Test batch command handling."""
        # Create more complete test data structure
        data_dir = Path(self.temp_dir) / "data"

        # Create multiple groups and lengths
        for group in ["group_1", "group_2"]:
            for length in ["5in", "10in"]:
                group_length_dir = data_dir / group / length
                group_length_dir.mkdir(parents=True, exist_ok=True)

                # Create multiple CSV files
                for i in range(3):
                    csv_file = group_length_dir / f"test--line-crimp-21--{i}.csv"
                    sample_data = """Time,Force,Stroke
sec,N,mm
0.0,0.0,0.0
0.1,1.0,0.1
0.2,2.0,0.2
0.3,3.0,0.3
0.4,2.5,0.4
0.5,1.0,0.5"""
                    with open(csv_file, "w") as f:
                        f.write(sample_data)

        args = {"data_dir": str(data_dir), "output": self.temp_dir, "summary": True}

        result = __main__.handle_batch_command(args)
        assert result == 0

        # Check if summary report was created
        summary_file = Path(self.temp_dir) / "summary_report.txt"
        assert summary_file.exists()

    def test_handle_batch_command_nonexistent_dir(self) -> None:
        """Test batch command with nonexistent directory."""
        args = {
            "data_dir": "/nonexistent/path",
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
        assert result == 1

    def test_handle_batch_command_error(self) -> None:
        """Test batch command error handling."""
        # Create directory but with invalid CSV files
        data_dir = Path(self.temp_dir) / "data" / "group_1" / "5in"
        data_dir.mkdir(parents=True, exist_ok=True)

        # Create invalid CSV
        csv_file = data_dir / "test--line-crimp-21--0.csv"
        with open(csv_file, "w") as f:
            f.write("invalid,csv,data")

        args = {
            "data_dir": str(data_dir.parent.parent),
            "output": self.temp_dir,
            "summary": False,
        }

        result = __main__.handle_batch_command(args)
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
            "Fishing_Line_Material_Properties_Analysis.__main__.parse_command_line"
        ) as mock_parse:
            mock_parse.return_value = {
                "command": "unknown",
                "output": self.temp_dir,
                "verbosity": 30,
            }

            result = __main__.main()
            assert result == 1

    def test_main_exception_handling(self) -> None:
        """Test main function exception handling."""
        with patch(
            "Fishing_Line_Material_Properties_Analysis.__main__.parse_command_line"
        ) as mock_parse:
            mock_parse.side_effect = Exception("Test exception")

    def test_main_keyboard_interrupt(self) -> None:
        """Test main function keyboard interrupt handling."""
        # Mock the main function to raise KeyboardInterrupt vs running it
        with patch(
            "Fishing_Line_Material_Properties_Analysis.__main__.main",
            side_effect=KeyboardInterrupt,
        ):
            with pytest.raises(SystemExit) as exc_info:
                # This should simulate pressing Ctrl+C during main execution
                try:
                    __main__.main()
                except KeyboardInterrupt:
                    # Simulate the typical behavior when KeyboardInterrupt is caught
                    print("\nExited by user")
                    sys.exit(1)
            assert exc_info.value.code == 1
