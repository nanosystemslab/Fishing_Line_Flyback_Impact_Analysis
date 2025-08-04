"""Tests for impulse analysis module."""

import tempfile
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from Fishing_Line_Flyback_Impact_Analysis.impulse_analysis import ImpulseAnalyzer
from Fishing_Line_Flyback_Impact_Analysis.impulse_analysis import (
    analyze_single_file_with_impulse,
)
from Fishing_Line_Flyback_Impact_Analysis.impulse_analysis import (
    create_impulse_boxplots,
)
from Fishing_Line_Flyback_Impact_Analysis.impulse_analysis import run_impulse_analysis


class TestImpulseAnalyzer:
    """Test ImpulseAnalyzer class."""

    def test_init_with_material_code(self):
        """Test analyzer initialization with material code."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        assert analyzer.material_code == "STND"
        assert analyzer.include_line_mass is True
        assert analyzer.line_mass_fraction == 0.70
        assert analyzer.total_mass is not None
        assert analyzer.total_mass > 0

    def test_init_without_material_code(self):
        """Test analyzer initialization without material code."""
        analyzer = ImpulseAnalyzer()

        assert analyzer.material_code is None
        assert analyzer.total_mass is None

    def test_init_with_custom_parameters(self):
        """Test analyzer initialization with custom parameters."""
        analyzer = ImpulseAnalyzer(
            material_code="DF",
            include_line_mass=False,
            line_mass_fraction=0.5,
            sampling_rate=50000.0,
            impact_threshold_factor=0.05,
        )

        assert analyzer.material_code == "DF"
        assert analyzer.include_line_mass is False
        assert analyzer.line_mass_fraction == 0.5
        assert analyzer.sampling_rate == 50000.0
        assert analyzer.impact_threshold_factor == 0.05

    def test_find_impact_boundaries_normal_data(self):
        """Test impact boundary detection with normal data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Create synthetic force data with clear impact
        n_points = 1000
        force = np.zeros(n_points)

        # Add impact peak around middle
        peak_start = 400
        peak_end = 600
        force[peak_start:peak_end] = 1000 * np.sin(
            np.linspace(0, np.pi, peak_end - peak_start)
        )

        start_idx, end_idx = analyzer.find_impact_boundaries(force)

        assert 0 <= start_idx < end_idx < len(force)
        assert (
            peak_start <= start_idx <= peak_start + 50
        )  # Should be close to actual start
        assert (
            peak_end - 50 <= end_idx <= peak_end + 50
        )  # Should be close to actual end

    def test_find_impact_boundaries_empty_data(self):
        """Test impact boundary detection with empty data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        start_idx, end_idx = analyzer.find_impact_boundaries(np.array([]))
        assert start_idx == 0
        assert end_idx == 0

    def test_find_impact_boundaries_weak_signal(self):
        """Test impact boundary detection with weak signal."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Very weak signal - should fall back to minimal window
        force = np.random.normal(0, 0.1, 1000)
        force[500] = 1.0  # Small peak

        start_idx, end_idx = analyzer.find_impact_boundaries(force)

        assert 0 <= start_idx < end_idx < len(force)
        assert end_idx - start_idx >= 10  # Minimum duration enforced

    def test_calculate_impulse_metrics_valid_data(self):
        """Test impulse calculation with valid data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Create simple force pulse
        n_points = 1000
        dt = 0.0001
        time = np.arange(n_points) * dt
        force = np.zeros(n_points)
        force[400:600] = 1000.0  # 200-point pulse

        result = analyzer.calculate_impulse_metrics(force, time)

        assert "error" not in result
        assert "total_impulse" in result
        assert "total_abs_impulse" in result
        assert "peak_force" in result
        assert "impact_duration" in result

        # Check values are reasonable
        assert result["peak_force"] == 1000.0
        assert result["total_abs_impulse"] > 0
        assert result["impact_duration"] > 0
        assert result["sampling_rate_hz"] == analyzer.sampling_rate

    def test_calculate_impulse_metrics_empty_data(self):
        """Test impulse calculation with empty data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        result = analyzer.calculate_impulse_metrics(np.array([]), np.array([]))
        assert "error" in result
        assert "Empty force or time data" in result["error"]

    def test_calculate_impulse_metrics_with_mass(self):
        """Test impulse calculation with mass for velocity calculation."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Simple impulse
        time = np.linspace(0, 0.01, 100)
        force = np.ones(100) * 100  # 100N for 0.01s = 1 N⋅s impulse

        result = analyzer.calculate_impulse_metrics(force, time)

        assert "equivalent_velocity" in result
        assert "equivalent_kinetic_energy" in result
        assert not np.isnan(result["equivalent_velocity"])
        assert not np.isnan(result["equivalent_kinetic_energy"])

        # Should be approximately 1 N⋅s / mass
        expected_velocity = 1.0 / analyzer.total_mass
        assert abs(result["equivalent_velocity"] - expected_velocity) < 0.1

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def test_analyze_csv_file_success(self):
        """Test successful CSV file analysis."""
        analyzer = ImpulseAnalyzer(material_code="STND")
        print(analyzer)

        # Create test CSV with force data
        test_data = {
            "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
            "AI_Channel_2_lbf": np.random.normal(0, 5, 1000),
            "Time": np.linspace(0, 0.1, 1000),
        }
        # Add a clear impact signal
        test_data["AI_Channel_1_lbf"][400:600] = 500

        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            result = analyze_single_file_with_impulse(file_path)

            assert "error" not in result
            assert result["filename"] == "STND-21-5.csv"
            assert result["material_type"] == "STND"
            assert "total_impulse" in result

        finally:
            file_path.unlink()

    def test_analyze_single_file_with_impulse_auto_detect(self):
        """Test single file analysis with auto-detection."""
        test_data = {
            "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
            "Time": np.linspace(0, 0.1, 1000),
        }
        test_data["AI_Channel_1_lbf"][400:600] = 500

        file_path = self.create_test_csv(test_data, "DF-21-10.csv")

        try:
            result = analyze_single_file_with_impulse(file_path, material_code=None)

            assert "error" not in result
            assert result["material_type"] == "DF"  # Auto-detected

        finally:
            file_path.unlink()

    @patch("matplotlib.pyplot.show")
    def test_analyze_single_file_with_plot(self, mock_show):
        """Test single file analysis with plotting."""
        test_data = {
            "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
            "Time": np.linspace(0, 0.1, 1000),
        }
        test_data["AI_Channel_1_lbf"][400:600] = 500

        file_path = self.create_test_csv(test_data, "STND-21-1.csv")

        try:
            result = analyze_single_file_with_impulse(file_path, show_plot=True)

            assert "error" not in result
            mock_show.assert_called_once()

        finally:
            file_path.unlink()

    def test_run_impulse_analysis_empty_directory(self):
        """Test impulse analysis on empty directory."""
        with tempfile.TemporaryDirectory() as temp_dir:
            results = run_impulse_analysis(temp_dir, temp_dir)
            assert results == []

    def test_run_impulse_analysis_with_files(self):
        """Test impulse analysis with multiple files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create multiple test files
            for i, material in enumerate(["STND", "DF", "DS"]):
                test_data = {
                    "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
                    "Time": np.linspace(0, 0.1, 1000),
                }
                test_data["AI_Channel_1_lbf"][400:600] = 500 + i * 100

                file_path = temp_path / f"{material}-21-{i + 1}.csv"
                df = pd.DataFrame(test_data)
                df.to_csv(file_path, index=False)

            results = run_impulse_analysis(temp_dir, temp_dir)

            assert len(results) == 3
            materials_found = [
                r.get("material_type") for r in results if "error" not in r
            ]
            assert "STND" in materials_found
            assert "DF" in materials_found
            assert "DS" in materials_found

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_impulse_boxplots_si_units(self, mock_close, mock_savefig):
        """Test creating impulse box plots with SI units."""
        # Create mock results
        results = [
            {
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "material_type": "STND",
                "total_impulse": 0.001,
            },
            {
                "total_abs_impulse": 0.002,
                "peak_force": 1500,
                "material_type": "DF",
                "total_impulse": -0.002,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            create_impulse_boxplots(results, Path(temp_dir), "SI")

            # Should save plots
            assert mock_savefig.call_count == 2  # .svg and .png
            mock_close.assert_called_once()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_impulse_boxplots_mixed_units(self, mock_close, mock_savefig):
        """Test creating impulse box plots with mixed units."""
        results = [
            {"total_abs_impulse": 0.001, "peak_force": 1000, "material_type": "STND"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            create_impulse_boxplots(results, Path(temp_dir), "mixed")

            assert mock_savefig.call_count == 2

    def test_create_impulse_boxplots_no_valid_results(self):
        """Test creating box plots with no valid results."""
        results = [
            {"error": "Some error"},
            {"filename": "test.csv"},  # Missing required fields
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error, but should print message
            create_impulse_boxplots(results, Path(temp_dir), "SI")

    def test_create_impulse_boxplots_invalid_units(self):
        """Test creating box plots with invalid units."""
        results = [
            {"total_abs_impulse": 0.001, "peak_force": 1000, "material_type": "STND"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should fall back to SI units
            create_impulse_boxplots(results, Path(temp_dir), "invalid")


class TestEdgeCases:
    """Test edge cases and error conditions."""

    def test_impulse_analyzer_with_zero_mass(self):
        """Test analyzer with zero total mass."""
        analyzer = ImpulseAnalyzer(material_code="UNKNOWN")
        analyzer.total_mass = 0

        time = np.linspace(0, 0.01, 100)
        force = np.ones(100) * 100

        result = analyzer.calculate_impulse_metrics(force, time)

        # Should handle zero mass gracefully
        assert "error" not in result
        assert np.isnan(result["equivalent_velocity"])
        assert np.isnan(result["equivalent_kinetic_energy"])

    def test_impulse_analyzer_very_short_impact(self):
        """Test analyzer with very short impact duration."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Single-point impact
        force = np.zeros(1000)
        force[500] = 10000  # Very high, very brief
        time = np.linspace(0, 0.1, 1000)

        result = analyzer.calculate_impulse_metrics(force, time)

        assert "error" not in result
        assert result["impact_duration"] > 0
        assert result["peak_force"] == 10000

    def test_find_impact_boundaries_all_negative(self):
        """Test boundary detection with all negative forces."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        force = -np.abs(np.random.normal(0, 100, 1000))
        force[400:600] = -1000  # Strong negative peak

        start_idx, end_idx = analyzer.find_impact_boundaries(force)

        # Should still find boundaries based on absolute values
        assert 0 <= start_idx < end_idx < len(force)

    def test_calculate_impulse_metrics_mismatched_lengths(self):
        """Test impulse calculation with mismatched force/time lengths."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        force = np.ones(100)
        time = np.linspace(0, 0.01, 50)  # Different length

        # Should handle gracefully or raise appropriate error
        try:
            result = analyzer.calculate_impulse_metrics(force, time)
            # If it doesn't raise an error, check result
            assert "error" in result or "total_impulse" in result
        except (ValueError, IndexError):
            # This is also acceptable behavior
            pass


class TestIntegration:
    """Integration tests combining multiple components."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def create_realistic_test_data(self, material="STND", sample="21-5"):
        """Create realistic test data mimicking actual fishing line impact."""
        n_points = 10000
        dt = 1e-5  # 100kHz sampling
        time = np.arange(n_points) * dt

        # Base noise
        force1 = np.random.normal(0, 5, n_points)
        force2 = np.random.normal(0, 3, n_points)

        # Impact event (asymmetric pulse)
        impact_start = 3000
        impact_peak = 4000
        impact_end = 6000

        # Rising phase
        rise_indices = np.arange(impact_start, impact_peak)
        rise_profile = np.exp(
            3 * (rise_indices - impact_start) / (impact_peak - impact_start)
        )
        force1[impact_start:impact_peak] += 2000 * rise_profile

        # Falling phase
        fall_indices = np.arange(impact_peak, impact_end)
        fall_profile = np.exp(
            -2 * (fall_indices - impact_peak) / (impact_end - impact_peak)
        )
        force1[impact_peak:impact_end] += 2000 * fall_profile

        # Secondary sensors
        force2[impact_start:impact_end] += 0.3 * force1[impact_start:impact_end]

        return {"AI_Channel_1_lbf": force1, "AI_Channel_2_lbf": force2, "Time": time}

    def test_full_analysis_pipeline(self):
        """Test complete analysis pipeline from CSV to results."""
        test_data = self.create_realistic_test_data("STND", "21-5")

        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / "STND-21-5.csv"

        try:
            # Save test data
            df = pd.DataFrame(test_data)
            df.to_csv(file_path, index=False)

            # Run full analysis
            result = analyze_single_file_with_impulse(
                file_path,
                material_code="STND",
                include_line_mass=True,
                line_mass_fraction=0.7,
            )

            # Verify complete result structure
            assert "error" not in result

            # Metadata
            assert result["filename"] == "STND-21-5.csv"
            assert result["material_type"] == "STND"
            assert result["sample_number"] == "21-5"

            # Core metrics
            assert "total_impulse" in result
            assert "total_abs_impulse" in result
            assert "impact_impulse" in result
            assert "impact_abs_impulse" in result

            # Force characteristics
            assert "peak_force" in result
            assert "peak_force_positive" in result
            assert "peak_force_negative" in result
            assert "rms_force" in result

            # Timing
            assert "impact_duration" in result
            assert "impact_start_time" in result
            assert "impact_end_time" in result
            assert "total_duration" in result

            # Equivalent metrics
            assert "equivalent_velocity" in result
            assert "equivalent_kinetic_energy" in result

            # Analysis metadata
            assert "mass_kg" in result
            assert "analysis_method" in result
            assert result["analysis_method"] == "impulse_integration"

            # Verify reasonable values
            assert result["peak_force"] > 1000  # Should detect our 2000N peak
            assert result["impact_duration"] > 0
            assert result["total_abs_impulse"] > 0
            assert not np.isnan(result["equivalent_velocity"])

        finally:
            if file_path.exists():
                file_path.unlink()

    def test_batch_analysis_integration(self):
        """Test batch analysis with multiple realistic files."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_path = temp_path / "output"

            # Create multiple test files with different characteristics
            materials = ["STND", "DF", "DS", "SL", "BR"]
            for i, material in enumerate(materials):
                for j in range(2):  # 2 samples per material
                    test_data = self.create_realistic_test_data(material, f"21-{j + 1}")

                    # Vary impact characteristics
                    scale_factor = 0.5 + i * 0.3  # Different impact magnitudes
                    test_data["AI_Channel_1_lbf"] *= scale_factor

                    file_path = temp_path / f"{material}-21-{j + 1}.csv"
                    df = pd.DataFrame(test_data)
                    df.to_csv(file_path, index=False)

            # Run batch analysis
            results = run_impulse_analysis(str(temp_path), str(output_path))

            # Verify results
            assert len(results) == 10  # 5 materials × 2 samples

            valid_results = [r for r in results if "error" not in r]
            assert len(valid_results) == 10  # All should succeed

            # Check material distribution
            materials_found = {r["material_type"] for r in valid_results}
            assert materials_found == set(materials)

            # Verify output files exist
            assert (output_path / "impulse_analysis_results.csv").exists()
            assert (output_path / "impulse_analysis_full_results.json").exists()

            # Check that different materials have different impulse characteristics
            material_impulses = {}
            for result in valid_results:
                material = result["material_type"]
                if material not in material_impulses:
                    material_impulses[material] = []
                material_impulses[material].append(result["total_abs_impulse"])

            # Should have variation between materials due to our scaling
            all_impulses = [
                impulse
                for impulses in material_impulses.values()
                for impulse in impulses
            ]
            assert np.std(all_impulses) > 0  # Should have variation.csv")

        try:
            result = analyzer.analyze_csv_file(file_path)  # noqa: F821

            assert "error" not in result
            assert result["filename"] == "STND-21-5.csv"
            assert result["material_type"] == "STND"
            assert result["sample_number"] == "21-5"
            assert "total_impulse" in result
            assert "peak_force" in result

        finally:
            file_path.unlink()

    def test_analyze_csv_file_insufficient_data(self):
        """Test CSV analysis with insufficient data."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Very small dataset
        test_data = {"AI_Channel_1_lbf": [1, 2], "Time": [0, 0.001]}

        file_path = self.create_test_csv(test_data)

        try:
            result = analyzer.analyze_csv_file(file_path)
            assert "error" in result
            assert "Insufficient data points" in result["error"]

        finally:
            file_path.unlink()

    def test_analyze_csv_file_nonexistent(self):
        """Test CSV analysis with nonexistent file."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        result = analyzer.analyze_csv_file("nonexistent.csv")
        assert "error" in result
        assert result["filename"] == "nonexistent.csv"

    @patch("matplotlib.pyplot.show")
    def test_plot_impulse_boundaries(self, mock_show):
        """Test plotting impulse boundaries."""
        analyzer = ImpulseAnalyzer(material_code="STND")

        # Create test data
        time = np.linspace(0, 0.01, 1000)
        force = np.zeros(1000)
        force[400:600] = 1000

        # Should not raise any errors
        analyzer.plot_impulse_boundaries(force, time, 400, 600, "test.csv")
        mock_show.assert_called_once()


class TestModuleFunctions:
    """Test module-level functions."""

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    def test_analyze_single_file_with_impulse_success(self):
        """Test single file analysis function."""
        # Create test data
        test_data = {
            "AI_Channel_1_lbf": np.random.normal(0, 10, 1000),
            "AI_Channel_2_lbf": np.random.normal(0, 5, 1000),
            "Time": np.linspace(0, 0.1, 1000),
        }
        test_data["AI_Channel_1_lbf"][400:600] = 500

        file_path = self.create_test_csv(test_data, "STND-21-5.csv")
        print(file_path)
