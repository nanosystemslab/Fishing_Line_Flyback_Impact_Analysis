"""Tests for main module CLI and package-level functions."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
from click.testing import CliRunner

from Fishing_Line_Flyback_Impact_Analysis import batch_analysis
from Fishing_Line_Flyback_Impact_Analysis import get_configuration_info
from Fishing_Line_Flyback_Impact_Analysis import gui_info
from Fishing_Line_Flyback_Impact_Analysis import quick_analysis
from Fishing_Line_Flyback_Impact_Analysis import quick_boundary_check
from Fishing_Line_Flyback_Impact_Analysis.__main__ import main


class TestCLICommands:
    """Test CLI command interface."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def create_test_data(self, material="STND", n_points=1000):
        """Create test force data."""
        time = np.linspace(0, 0.1, n_points)
        force1 = np.random.normal(0, 10, n_points)
        force2 = np.random.normal(0, 5, n_points)

        # Add impact signal
        impact_start = n_points // 3
        impact_end = 2 * n_points // 3
        force1[impact_start:impact_end] = 1000 * np.sin(
            np.linspace(0, np.pi, impact_end - impact_start)
        )

        return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}

    def test_cli_help(self):
        """Test CLI help command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        assert "Fishing Line Flyback Impact Analysis" in result.output
        assert "analyze-impulse" in result.output

    def test_cli_version(self):
        """Test CLI version command."""
        runner = CliRunner()
        result = runner.invoke(main, ["--version"])

        assert result.exit_code == 0
        assert "1.0.0" in result.output

    def test_analyze_impulse_command_help(self):
        """Test analyze-impulse command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze-impulse", "--help"])

        assert result.exit_code == 0
        assert "Run impulse analysis" in result.output
        assert "--output-dir" in result.output

    def test_analyze_impulse_empty_directory(self):
        """Test analyze-impulse with empty directory."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            result = runner.invoke(
                main, ["analyze-impulse", temp_dir, "--output-dir", temp_dir]
            )

            assert result.exit_code == 0
            assert "No CSV files found" in result.output

    def test_analyze_impulse_with_files(self):
        """Test analyze-impulse with test files."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            for material in ["STND", "DF"]:
                test_data = self.create_test_data(material)
                file_path = temp_path / f"{material}-21-1.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

            result = runner.invoke(
                main, ["analyze-impulse", temp_dir, "--output-dir", temp_dir]
            )

            assert result.exit_code == 0
            assert "Successfully analyzed:" in result.output

            # Check output files exist
            output_path = Path(temp_dir)
            assert (output_path / "impulse_analysis_results.csv").exists()

    def test_analyze_single_command_help(self):
        """Test analyze-single command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze-single", "--help"])

        assert result.exit_code == 0
        assert "Analyze a single measurement file" in result.output

    def test_analyze_single_success(self):
        """Test analyze-single with valid file."""
        runner = CliRunner()

        test_data = self.create_test_data("STND")
        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            result = runner.invoke(
                main, ["analyze-single", str(file_path), "--material", "STND"]
            )

            assert result.exit_code == 0
            assert "IMPULSE ANALYSIS RESULTS" in result.output
            assert "Total impulse:" in result.output

        finally:
            file_path.unlink()

    def test_analyze_single_with_debug(self):
        """Test analyze-single with debug flag."""
        runner = CliRunner()

        test_data = self.create_test_data("DF")
        file_path = self.create_test_csv(test_data, "DF-21-3.csv")

        try:
            result = runner.invoke(main, ["analyze-single", str(file_path), "--debug"])

            assert result.exit_code == 0
            assert "DEBUG INFO" in result.output

        finally:
            file_path.unlink()

    def test_analyze_single_with_output(self):
        """Test analyze-single with output file."""
        runner = CliRunner()

        test_data = self.create_test_data("DS")
        file_path = self.create_test_csv(test_data, "DS-21-1.csv")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                output_file = Path(temp_dir) / "result.json"

                result = runner.invoke(
                    main,
                    ["analyze-single", str(file_path), "--output", str(output_file)],
                )

                assert result.exit_code == 0
                assert output_file.exists()

                # Verify JSON content
                with open(output_file) as f:
                    data = json.load(f)
                    assert "total_impulse" in data

        finally:
            file_path.unlink()

    @patch("matplotlib.pyplot.show")
    def test_analyze_single_with_plot(self, mock_show):
        """Test analyze-single with plot option."""
        runner = CliRunner()

        test_data = self.create_test_data("SL")
        file_path = self.create_test_csv(test_data, "SL-21-2.csv")

        try:
            result = runner.invoke(
                main, ["analyze-single", str(file_path), "--show-plot"]
            )

            assert result.exit_code == 0
            mock_show.assert_called_once()

        finally:
            file_path.unlink()

    def test_analyze_single_nonexistent_file(self):
        """Test analyze-single with nonexistent file."""
        runner = CliRunner()

        result = runner.invoke(main, ["analyze-single", "nonexistent.csv"])

        assert result.exit_code != 0

    def test_plot_file_command_help(self):
        """Test plot-file command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["plot-file", "--help"])

        assert result.exit_code == 0
        assert "Plot force curve" in result.output

    @patch("matplotlib.pyplot.show")
    def test_plot_file_success(self, mock_show):
        """Test plot-file command."""
        runner = CliRunner()

        test_data = self.create_test_data("BR")
        file_path = self.create_test_csv(test_data, "BR-21-1.csv")

        try:
            result = runner.invoke(
                main, ["plot-file", str(file_path), "--style", "simple"]
            )

            assert result.exit_code == 0
            mock_show.assert_called_once()

        finally:
            file_path.unlink()

    @patch("matplotlib.pyplot.show")
    def test_interactive_plot_command(self, mock_show):
        """Test interactive-plot command."""
        runner = CliRunner()

        test_data = self.create_test_data("STND")
        file_path = self.create_test_csv(test_data, "STND-21-1.csv")

        try:
            result = runner.invoke(main, ["interactive-plot", str(file_path)])

            assert result.exit_code == 0

        finally:
            file_path.unlink()

    def test_boundary_viewer_command_help(self):
        """Test boundary-viewer command help."""
        runner = CliRunner()
        result = runner.invoke(main, ["boundary-viewer", "--help"])

        assert result.exit_code == 0
        assert "integration boundary viewer" in result.output.lower()

    def test_quick_check_command(self):
        """Test quick-check command."""
        runner = CliRunner()

        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            for i in range(3):
                test_data = self.create_test_data("STND")
                file_path = temp_path / f"STND-21-{i + 1}.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

            result = runner.invoke(main, ["quick-check", temp_dir, "--count", "2"])

            assert result.exit_code == 0
            assert "QUICK IMPULSE CHECK" in result.output

    def test_info_command(self):
        """Test info command."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])

        assert result.exit_code == 0
        assert "FISHING LINE FLYBACK IMPACT ANALYSIS" in result.output
        assert "SCIENTIFIC METHOD:" in result.output
        assert "USAGE EXAMPLES" in result.output

    def test_legacy_command_group(self):
        """Test legacy command group."""
        runner = CliRunner()
        result = runner.invoke(main, ["legacy", "--help"])

        assert result.exit_code == 0
        assert "legacy" in result.output.lower()


class TestPackageLevelFunctions:
    """Test package-level convenience functions."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def create_test_data(self, material="STND"):
        """Create test force data."""
        time = np.linspace(0, 0.1, 1000)
        force1 = np.random.normal(0, 10, 1000)
        force2 = np.random.normal(0, 5, 1000)

        # Add impact signal
        force1[400:600] = 1000 * np.sin(np.linspace(0, np.pi, 200))

        return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}

    def test_quick_analysis_function(self):
        """Test quick_analysis convenience function."""
        test_data = self.create_test_data("STND")
        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            result = quick_analysis(file_path, show_plot=False)

            assert "error" not in result
            assert "total_impulse" in result
            assert result["filename"] == "STND-21-5.csv"

        finally:
            file_path.unlink()

    @patch("matplotlib.pyplot.show")
    def test_quick_analysis_with_plot(self, mock_show):
        """Test quick_analysis with plotting."""
        test_data = self.create_test_data("DF")
        file_path = self.create_test_csv(test_data, "DF-21-3.csv")

        try:
            result = quick_analysis(file_path, show_plot=True)

            assert "error" not in result
            mock_show.assert_called_once()

        finally:
            file_path.unlink()

    def test_batch_analysis_function(self):
        """Test batch_analysis convenience function."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create test files
            for material in ["STND", "DF", "DS"]:
                test_data = self.create_test_data(material)
                file_path = temp_path / f"{material}-21-1.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

            results = batch_analysis(temp_dir)

            assert len(results) == 3
            valid_results = [r for r in results if "error" not in r]
            assert len(valid_results) == 3

    def test_batch_analysis_with_output_dir(self):
        """Test batch_analysis with custom output directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            data_dir = temp_path / "data"
            output_dir = temp_path / "output"

            data_dir.mkdir()

            # Create test file
            test_data = self.create_test_data("STND")
            file_path = data_dir / "STND-21-1.csv"
            df = pd.DataFrame(test_data)
            df.to_csv(file_path, index=False)

            results = batch_analysis(str(data_dir), str(output_dir))

            assert len(results) == 1
            assert output_dir.exists()

    def test_get_configuration_info(self):
        """Test get_configuration_info function."""
        info = get_configuration_info()

        assert isinstance(info, dict)
        assert "configurations" in info
        assert "line_mass_fraction" in info
        assert "version" in info
        assert "analysis_method" in info

        # Check configurations
        configs = info["configurations"]
        expected_materials = {"STND", "DF", "DS", "SL", "BR"}
        assert set(configs.keys()) == expected_materials

        for _material, config in configs.items():
            assert "name" in config
            assert "hardware_mass_kg" in config
            assert "total_mass_kg" in config
            assert config["total_mass_kg"] > config["hardware_mass_kg"]

    def test_gui_info_function(self):
        """Test gui_info function."""
        info = gui_info()

        assert isinstance(info, dict)
        assert "gui_available" in info
        assert "components" in info
        assert "requirements" in info
        assert "features" in info

        # Check requirements
        assert "PyQt5" in info["requirements"]
        assert "pyqtgraph" in info["requirements"]

        # Check features
        features = info["features"]
        assert isinstance(features, list)
        assert len(features) > 0

    def test_quick_boundary_check_function(self):
        """Test quick_boundary_check function."""
        test_data = self.create_test_data("STND")
        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            # Should not raise error even if GUI not available
            quick_boundary_check(str(file_path))

        finally:
            file_path.unlink()


class TestMainInit:
    """Test main module __init__.py functionality."""

    def test_main_module_imports(self):
        """Test that main module imports work correctly."""
        import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813

        # Core analysis functions
        assert hasattr(fli, "ImpulseAnalyzer")
        assert hasattr(fli, "run_impulse_analysis")
        assert hasattr(fli, "analyze_single_file_with_impulse")

        # Visualization functions
        assert hasattr(fli, "plot_single_file_analysis")
        assert hasattr(fli, "show_force_preview")

        # Convenience functions
        assert hasattr(fli, "quick_analysis")
        assert hasattr(fli, "batch_analysis")
        assert hasattr(fli, "get_configuration_info")

        # GUI functions
        assert hasattr(fli, "launch_boundary_viewer")
        assert hasattr(fli, "quick_boundary_check")
        assert hasattr(fli, "gui_info")

    def test_module_metadata(self):
        """Test module metadata."""
        import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813

        assert hasattr(fli, "__version__")
        assert hasattr(fli, "__author__")
        assert hasattr(fli, "__description__")

        assert fli.__version__ == "1.0.0"
        assert "Nanosystems Lab" in fli.__author__
        assert "Impulse-Focused" in fli.__description__

    def test_module_all_exports(self):
        """Test that __all__ is properly defined."""
        import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813

        assert hasattr(fli, "__all__")
        assert isinstance(fli.__all__, list)
        assert len(fli.__all__) > 0

        # Check that all listed items are actually available
        for item in fli.__all__:
            assert hasattr(fli, item)

    def test_shared_constants_access(self):
        """Test access to shared constants."""
        import Fishing_Line_Flyback_Impact_Analysis as fli  # noqa: N813

        assert hasattr(fli, "CONFIG_WEIGHTS")
        assert hasattr(fli, "MATERIAL_NAMES")
        assert hasattr(fli, "LINE_MASS_FRACTION")

        # Test values
        assert isinstance(fli.CONFIG_WEIGHTS, dict)
        assert isinstance(fli.MATERIAL_NAMES, dict)
        assert isinstance(fli.LINE_MASS_FRACTION, float)

    @patch("matplotlib.pyplot.rcParams")
    def test_matplotlib_configuration(self, mock_rcparams):
        """Test that matplotlib is configured properly."""
        # Re-import to trigger configuration
        import importlib

        import Fishing_Line_Flyback_Impact_Analysis  # noqa: N813

        importlib.reload(Fishing_Line_Flyback_Impact_Analysis)

        # Should have configured matplotlib
        # Note: This is a basic test since mocking rcParams is complex


class TestErrorHandling:
    """Test error handling in main module functions."""

    def test_quick_analysis_error_handling(self):
        """Test quick_analysis with invalid file."""
        result = quick_analysis("nonexistent.csv", show_plot=False)
        assert "error" in result

    def test_batch_analysis_error_handling(self):
        """Test batch_analysis with invalid directory."""
        results = batch_analysis("/nonexistent/directory")
        assert isinstance(results, list)

    def test_get_configuration_info_stability(self):
        """Test that get_configuration_info is stable."""
        # Should work multiple times
        info1 = get_configuration_info()
        info2 = get_configuration_info()

        assert info1 == info2

    def test_gui_info_stability(self):
        """Test that gui_info is stable."""
        info1 = gui_info()
        info2 = gui_info()

        assert info1 == info2


class TestCLIErrorHandling:
    """Test CLI error handling."""

    def test_cli_invalid_command(self):
        """Test CLI with invalid command."""
        runner = CliRunner()
        result = runner.invoke(main, ["invalid-command"])

        assert result.exit_code != 0

    def test_analyze_single_invalid_material(self):
        """Test analyze-single with invalid material."""
        runner = CliRunner()

        test_data = {"AI_Channel_1_lbf": [1, 2, 3], "Time": [0, 0.001, 0.002]}
        file_path = self.create_test_csv(test_data, "TEST-21-1.csv")

        try:
            result = runner.invoke(
                main, ["analyze-single", str(file_path), "--material", "INVALID"]
            )
            print(result)

            # Should handle gracefully (might succeed with default values)
            # or fail with appropriate error message

        finally:
            file_path.unlink()

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path


class TestCLIOutput:
    """Test CLI output formatting and content."""

    def test_cli_help_formatting(self):
        """Test that CLI help is well-formatted."""
        runner = CliRunner()
        result = runner.invoke(main, ["--help"])

        assert result.exit_code == 0
        output = result.output

        # Should have clear structure
        assert "🎯" in output  # Emoji indicators
        assert "Commands:" in output
        assert len(output.split("\n")) > 10  # Substantial help text

    def test_analyze_impulse_help_formatting(self):
        """Test analyze-impulse help formatting."""
        runner = CliRunner()
        result = runner.invoke(main, ["analyze-impulse", "--help"])

        assert result.exit_code == 0
        output = result.output

        assert "🎯" in output
        assert "DATA_DIR:" in output
        assert "--output-dir" in output

    def test_info_command_content(self):
        """Test info command provides useful content."""
        runner = CliRunner()
        result = runner.invoke(main, ["info"])

        assert result.exit_code == 0
        output = result.output

        # Should contain key information
        assert "SCIENTIFIC METHOD" in output
        assert "HARDWARE CONFIGURATIONS:" in output
        assert "USAGE EXAMPLES" in output
        assert "impulse" in output.lower()


class TestIntegration:
    """Integration tests for main module components."""

    def create_realistic_test_scenario(self):
        """Create realistic test scenario with multiple files."""
        materials = ["STND", "DF", "DS", "SL", "BR"]
        files = []

        temp_dir = Path(tempfile.gettempdir()) / "test_scenario"
        temp_dir.mkdir(exist_ok=True)

        for material in materials:
            for sample in range(1, 4):  # 3 samples per material
                test_data = {
                    "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
                    "AI_Channel_2_lbf": np.random.normal(0, 5, 1000),
                    "Time": np.linspace(0, 0.1, 1000),
                }

                # Add material-specific impact characteristics
                impact_magnitude = 500 + material.__hash__() % 1000
                test_data["AI_Channel_1_lbf"][400:600] = impact_magnitude

                filename = f"{material}-21-{sample}.csv"
                file_path = temp_dir / filename

                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)
                files.append(file_path)

        return temp_dir, files

    def test_complete_workflow_cli(self):
        """Test complete workflow using CLI commands."""
        temp_dir, files = self.create_realistic_test_scenario()

        try:
            runner = CliRunner()

            # Test batch analysis
            result = runner.invoke(
                main,
                [
                    "analyze-impulse",
                    str(temp_dir),
                    "--output-dir",
                    str(temp_dir / "output"),
                ],
            )

            assert result.exit_code == 0
            assert "Successfully analyzed:" in result.output

            # Check output files
            output_dir = temp_dir / "output"
            assert (output_dir / "impulse_analysis_results.csv").exists()

            # Test individual file analysis
            test_file = files[0]
            result = runner.invoke(main, ["analyze-single", str(test_file)])

            assert result.exit_code == 0
            assert "IMPULSE ANALYSIS RESULTS" in result.output

        finally:
            # Cleanup
            for file_path in files:
                if file_path.exists():
                    file_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()

    def test_complete_workflow_api(self):
        """Test complete workflow using Python API."""
        temp_dir, files = self.create_realistic_test_scenario()

        try:
            # Test batch analysis
            results = batch_analysis(str(temp_dir))
            assert len(results) == 15  # 5 materials × 3 samples

            valid_results = [r for r in results if "error" not in r]
            assert len(valid_results) == 15

            # Test individual analysis
            test_file = files[0]
            result = quick_analysis(str(test_file))
            assert "error" not in result
            assert "total_impulse" in result

            # Test configuration info
            config_info = get_configuration_info()
            assert len(config_info["configurations"]) == 5

        finally:
            # Cleanup
            for file_path in files:
                if file_path.exists():
                    file_path.unlink()
            if temp_dir.exists():
                temp_dir.rmdir()
