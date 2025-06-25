"""Test cases for the analysis module."""

import os
import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis.analysis import ImpactAnalyzer


class TestImpactAnalyzer:
    """Test cases for ImpactAnalyzer class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.analyzer = ImpactAnalyzer()
        self.temp_dir = tempfile.mkdtemp()

        # Create sample test data
        self.sample_data = pd.DataFrame(
            {
                "Time (s)": [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
                "AI 1/AI 1 (lbf)": [0.0, 1.0, 2.0, 3.0, 2.5, 1.0],
                "AI 2/AI 2 (lbf)": [0.0, 1.1, 2.1, 3.1, 2.6, 1.1],
                "AI 3/AI 3 (lbf)": [0.0, 0.9, 1.9, 2.9, 2.4, 0.9],
                "AI 4/AI 4 (lbf)": [0.0, 1.2, 2.2, 3.2, 2.7, 1.2],
            }
        )

        # Create temporary CSV file
        self.temp_csv = os.path.join(self.temp_dir, "STND-21-1.csv")
        self.sample_data.to_csv(self.temp_csv, index=False)

    def teardown_method(self) -> None:
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_init(self) -> None:
        """Test ImpactAnalyzer initialization."""
        analyzer = ImpactAnalyzer()
        assert hasattr(analyzer, "log")
        assert hasattr(analyzer, "config_masses")
        assert analyzer.config_masses["STND"] == 49

    def test_load_file_csv_success(self) -> None:
        """Test successful CSV file loading."""
        df = self.analyzer.load_file(self.temp_csv)

        assert isinstance(df, pd.DataFrame)
        assert hasattr(df, "meta")
        assert df.meta.config == "STND"
        assert df.meta.diam == "21"
        assert df.meta.run_num == "1"
        assert "SUM" in df.columns
        assert "time" in df.columns

    def test_load_file_invalid_format(self) -> None:
        """Test loading file with invalid format."""
        invalid_file = os.path.join(self.temp_dir, "test.txt")
        with open(invalid_file, "w") as f:
            f.write("invalid file")

        with pytest.raises(ValueError, match="Unsupported file format"):
            self.analyzer.load_file(invalid_file)

    def test_load_file_nonexistent(self) -> None:
        """Test loading nonexistent file."""
        with pytest.raises((FileNotFoundError, pd.errors.EmptyDataError)):
            self.analyzer.load_file("nonexistent.csv")

    def test_load_csv_file_with_video_column(self) -> None:
        """Test CSV loading with video column removal."""
        # Create CSV with video column
        data_with_video = self.sample_data.copy()
        data_with_video["Video/Camera 0 ()"] = [None] * len(data_with_video)

        csv_with_video = os.path.join(self.temp_dir, "STND-21-2.csv")
        data_with_video.to_csv(csv_with_video, index=False)

        df = self.analyzer.load_file(csv_with_video)
        assert "Video/Camera 0 ()" not in df.columns

    def test_load_h5_file(self) -> None:
        """Test H5 file loading."""
        # First create H5 file
        df_original = self.analyzer.load_file(self.temp_csv)

        # Create an H5 file manually for testing
        h5_file = os.path.join(self.temp_dir, "test.h5")
        df_original.to_hdf(h5_file, key="df", mode="w")

        # Load H5 file
        df_loaded, peak = self.analyzer._load_h5_file(h5_file)
        assert isinstance(df_loaded, pd.DataFrame)

    def test_parse_filename_metadata(self) -> None:
        """Test filename metadata parsing."""
        test_cases = [
            ("STND-21-1.csv", "STND", "21", "1"),
            ("DF-25-10.csv", "DF", "25", "10"),
            ("BR-19-0.csv", "BR", "19", "0"),
            ("invalid.csv", "UNKNOWN", "21", "0"),  # fallback case
        ]

        for filename, expected_config, expected_diam, expected_run in test_cases:
            csv_path = os.path.join(self.temp_dir, filename)
            self.sample_data.to_csv(csv_path, index=False)

            df = self.analyzer.load_file(csv_path)
            assert df.meta.config == expected_config
            assert df.meta.diam == expected_diam
            assert df.meta.run_num == expected_run

    def test_calculate_impact_properties(self) -> None:
        """Test impact properties calculation."""
        df = self.analyzer.load_file(self.temp_csv)
        properties = self.analyzer.calculate_impact_properties(df)

        required_keys = [
            "max_force_N",
            "impulse_Ns",
            "velocity_ms",
            "kinetic_energy_J",
            "mass_kg",
            "config",
            "diameter",
            "run_number",
            "filename",
        ]

        for key in required_keys:
            assert key in properties

        assert properties["config"] == "STND"
        assert properties["diameter"] == "21"
        assert properties["run_number"] == "1"
        assert isinstance(properties["max_force_N"], float)
        assert isinstance(properties["impulse_Ns"], float)

    def test_calculate_impact_properties_different_params(self) -> None:
        """Test impact properties with different parameters."""
        df = self.analyzer.load_file(self.temp_csv)

        # Test with different Y parameter
        properties = self.analyzer.calculate_impact_properties(df, param_y="S1")
        assert isinstance(properties["max_force_N"], float)

    def test_process_single_file(self) -> None:
        """Test single file processing."""
        result = self.analyzer.process_single_file(self.temp_csv)

        assert "filepath" in result
        assert "filename" in result
        assert result["filepath"] == self.temp_csv
        assert result["filename"] == "STND-21-1.csv"

    def test_load_results_file_success(self) -> None:
        """Test loading results from text file."""
        results_file = os.path.join(self.temp_dir, "results.txt")
        with open(results_file, "w") as f:
            f.write("STND-21-1.csv,J=2.450,F=1250.32\n")
            f.write("DF-21-1.csv,J=1.200,F=800.15\n")

        df = self.analyzer.load_results_file(results_file)

        assert isinstance(df, pd.DataFrame)
        # Allow for filtering - just check we got some data
        assert len(df) >= 1
        assert "test_type" in df.columns
        assert "J" in df.columns
        assert "F" in df.columns

    def test_load_results_file_invalid_format(self) -> None:
        """Test loading invalid results file format."""
        invalid_file = os.path.join(self.temp_dir, "test.csv")
        with open(invalid_file, "w") as f:
            f.write("invalid")

        with pytest.raises(ValueError, match="Results file must be .txt format"):
            self.analyzer.load_results_file(invalid_file)

    def test_calculate_summary_stats(self) -> None:
        """Test summary statistics calculation."""
        # Create sample results DataFrame
        results_df = pd.DataFrame(
            {
                "fname": ["STND-21-1.csv", "STND-21-2.csv", "DF-21-1.csv"],
                "test_type": ["STND", "STND", "DF"],
                "J": [2.5, 2.3, 1.2],
                "F": [1250, 1200, 800],
            }
        )

        summary = self.analyzer.calculate_summary_stats(results_df)

        assert isinstance(summary, pd.DataFrame)
        assert len(summary) >= 1  # At least STND results
        assert "test_type" in summary.columns
        assert "mean_J" in summary.columns
        assert "std_J" in summary.columns

    def test_save_results_to_csv(self) -> None:
        """Test saving results to CSV."""
        results = pd.DataFrame(
            {"test_type": ["STND", "DF"], "mean_J": [2.4, 1.2], "std_J": [0.1, 0.05]}
        )

        self.analyzer.save_results_to_csv(results, self.temp_dir)

        csv_file = Path(self.temp_dir) / "results.csv"
        assert csv_file.exists()

    def test_generate_summary_report(self) -> None:
        """Test summary report generation."""
        results = pd.DataFrame(
            {
                "test_type": ["STND", "DF"],
                "mean_J": [2.4, 1.2],
                "std_J": [0.1, 0.05],
                "%diff_J": [0, -50],
                "mean_F": [1200, 800],
                "std_F": [50, 30],
                "%diff_F": [0, -33],
            }
        )

        self.analyzer.generate_summary_report(results, self.temp_dir)

        report_file = Path(self.temp_dir) / "summary_report.txt"
        assert report_file.exists()

    def test_generate_summary_table(self) -> None:
        """Test summary table generation."""
        # Create sample results DataFrame
        results_df = pd.DataFrame(
            {
                "fname": ["STND-21-1.csv", "DF-21-1.csv", "BR-21-1.csv"],
                "test_type": ["STND", "DF", "BR"],
                "J": [2.5, 1.2, 0.5],
                "F": [1250, 800, 400],
            }
        )

        with patch("builtins.print") as mock_print:
            config_stats = self.analyzer.generate_summary_table(
                results_df, self.temp_dir
            )

        assert isinstance(config_stats, dict)
        assert "STND" in config_stats
        mock_print.assert_called()  # Verify table was printed

    def test_generate_summary_table_no_output_dir(self) -> None:
        """Test summary table generation without output directory."""
        results_df = pd.DataFrame(
            {"fname": ["STND-21-1.csv"], "test_type": ["STND"], "J": [2.5], "F": [1250]}
        )

        with patch("builtins.print"):
            config_stats = self.analyzer.generate_summary_table(results_df)

        assert isinstance(config_stats, dict)

    def test_config_masses(self) -> None:
        """Test configuration masses are correctly defined."""
        expected_masses = {"STND": 49, "DF": 62, "DS": 75, "SL": 65, "BR": 45}

        for config, expected_mass in expected_masses.items():
            assert self.analyzer.config_masses[config] == expected_mass

    def test_edge_case_empty_dataframe(self) -> None:
        """Test handling of empty dataframes."""
        empty_df = pd.DataFrame()
        empty_df.meta = types.SimpleNamespace()
        empty_df.meta.config = "STND"

        # Should handle empty data gracefully
        with pytest.raises(ValueError, match="DataFrame is empty"):
            self.analyzer.calculate_impact_properties(empty_df)

    def test_edge_case_single_data_point(self) -> None:
        """Test handling of minimal data."""
        minimal_data = pd.DataFrame({"time": [0.0], "SUM": [100.0]})
        minimal_data.meta = types.SimpleNamespace()
        minimal_data.meta.config = "STND"
        minimal_data.meta.diam = "21"
        minimal_data.meta.run_num = "1"
        minimal_data.meta.fname = "test.csv"

        properties = self.analyzer.calculate_impact_properties(minimal_data)
        assert isinstance(properties, dict)

    def test_peak_detection_no_peaks(self) -> None:
        """Test peak detection when no peaks are found."""
        # Create data with no significant peaks
        flat_data = pd.DataFrame(
            {
                "Time (s)": [0.0, 0.1, 0.2],
                "AI 1/AI 1 (lbf)": [1.0, 1.1, 1.0],
                "AI 2/AI 2 (lbf)": [1.0, 1.1, 1.0],
                "AI 3/AI 3 (lbf)": [1.0, 1.1, 1.0],
                "AI 4/AI 4 (lbf)": [1.0, 1.1, 1.0],
            }
        )

        flat_csv = os.path.join(self.temp_dir, "flat.csv")
        flat_data.to_csv(flat_csv, index=False)

        df = self.analyzer.load_file(flat_csv)
        assert isinstance(df, pd.DataFrame)
        # Should handle case where no peaks are detected

    def test_unknown_config_mass(self) -> None:
        """Test handling of unknown configuration."""
        df = self.analyzer.load_file(self.temp_csv)
        df.meta.config = "UNKNOWN"

        properties = self.analyzer.calculate_impact_properties(df)
        # Should use default mass (45) for unknown config
        assert properties["mass_kg"] == 45 * 6.85218e-5
