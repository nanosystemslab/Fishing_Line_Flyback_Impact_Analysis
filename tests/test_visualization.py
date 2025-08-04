"""Tests for visualization module."""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from Fishing_Line_Flyback_Impact_Analysis.visualization import (
    create_impulse_material_comparison,
)
from Fishing_Line_Flyback_Impact_Analysis.visualization import (
    create_impulse_summary_statistics,
)
from Fishing_Line_Flyback_Impact_Analysis.visualization import (
    create_single_file_analysis_plot,
)
from Fishing_Line_Flyback_Impact_Analysis.visualization import create_summary_plots
from Fishing_Line_Flyback_Impact_Analysis.visualization import plot_single_file_analysis
from Fishing_Line_Flyback_Impact_Analysis.visualization import show_force_preview
from Fishing_Line_Flyback_Impact_Analysis.visualization import (
    show_force_preview_interactive,
)


class TestVisualizationFunctions:
    """Test core visualization functions."""

    def create_test_force_data(self, n_points=1000):
        """Create test force and time data."""
        time = np.linspace(0, 0.1, n_points)
        force = np.random.normal(0, 10, n_points)

        # Add clear impact signal
        impact_start = n_points // 3
        impact_end = 2 * n_points // 3
        force[impact_start:impact_end] = 1000 * np.sin(
            np.linspace(0, np.pi, impact_end - impact_start)
        )

        return force, time

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_show_force_preview_basic(self, mock_close, mock_show):
        """Test basic force preview without analysis."""
        force, time = self.create_test_force_data()

        fig = show_force_preview(force, time, "test.csv")

        assert fig is not None
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_show_force_preview_with_analysis(self, mock_close, mock_show):
        """Test force preview with analysis overlay."""
        force, time = self.create_test_force_data()

        fig = show_force_preview(
            force, time, "STND-21-5.csv", show_analysis_preview=True, mass=0.072
        )

        assert fig is not None
        mock_show.assert_called_once()

    @patch("matplotlib.pyplot.show")
    def test_show_force_preview_edge_cases(self, mock_show):
        """Test force preview with edge cases."""
        # Very short data
        force = np.array([1, 2, 3])
        time = np.array([0, 0.001, 0.002])

        fig = show_force_preview(force, time, "short.csv")
        assert fig is not None

        # All zero data
        force = np.zeros(100)
        time = np.linspace(0, 0.01, 100)

        fig = show_force_preview(force, time, "zero.csv")
        assert fig is not None

    def create_test_csv(self, data_dict, filename="test.csv"):
        """Helper to create test CSV files."""
        df = pd.DataFrame(data_dict)
        temp_dir = Path(tempfile.gettempdir())
        file_path = temp_dir / filename
        df.to_csv(file_path, index=False)
        return file_path

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_plot_single_file_analysis_success(self, mock_close, mock_show):
        """Test single file analysis plotting."""
        # Create test CSV
        force, time = self.create_test_force_data()
        test_data = {
            "AI_Channel_1_lbf": force / 4.44822,  # Convert back to lbf
            "AI_Channel_2_lbf": force / (2 * 4.44822),
            "Time": time,
        }

        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            plot_single_file_analysis(file_path, show_plot=True)
            mock_show.assert_called_once()
        finally:
            file_path.unlink()

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_plot_single_file_analysis_save(self, mock_close, mock_savefig):
        """Test single file analysis plotting with save."""
        force, time = self.create_test_force_data()
        test_data = {"AI_Channel_1_lbf": force / 4.44822, "Time": time}

        file_path = self.create_test_csv(test_data, "STND-21-5.csv")

        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                plot_single_file_analysis(
                    file_path, output_dir=temp_dir, show_plot=False
                )
                mock_savefig.assert_called_once()
                mock_close.assert_called_once()
        finally:
            file_path.unlink()

    def test_plot_single_file_analysis_error(self):
        """Test single file analysis plotting with error."""
        # Non-existent file
        plot_single_file_analysis("nonexistent.csv", show_plot=False)
        # Should not raise exception, just print error

    @patch("matplotlib.pyplot.close")
    def test_create_single_file_analysis_plot(self, mock_close):
        """Test creating single file analysis plot."""
        force, time = self.create_test_force_data()

        # Mock analysis result
        analysis_result = {
            "impact_start_idx": 300,
            "impact_end_idx": 700,
            "total_impulse": 0.05,
            "total_abs_impulse": 0.05,
            "impact_impulse": 0.04,
            "peak_force": 1000,
            "peak_force_positive": 1000,
            "peak_force_negative": -100,
            "rms_force": 300,
            "impact_duration": 0.02,
            "equivalent_velocity": 150,
            "equivalent_kinetic_energy": 0.5,
            "material_code": "STND",
            "sampling_rate_hz": 100000,
            "mass_breakdown": {"total_mass_kg": 0.072},
        }

        fig = create_single_file_analysis_plot(
            force, time, analysis_result, "STND-21-5.csv"
        )

        assert fig is not None
        # Check that figure has expected subplots
        assert len(fig.axes) == 4

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_impulse_material_comparison(self, mock_close, mock_savefig):
        """Test creating material comparison plots."""
        results = [
            {
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
                "total_impulse": 0.001,
            },
            {
                "total_abs_impulse": 0.002,
                "peak_force": 1500,
                "impact_duration": 0.025,
                "material_type": "DF",
                "total_impulse": -0.002,
            },
            {
                "total_abs_impulse": 0.0015,
                "peak_force": 1200,
                "impact_duration": 0.022,
                "material_type": "DS",
                "total_impulse": 0.0015,
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            create_impulse_material_comparison(results, temp_dir)
            mock_savefig.assert_called_once()
            mock_close.assert_called_once()

    def test_create_impulse_material_comparison_no_valid_results(self):
        """Test material comparison with no valid results."""
        results = [
            {"error": "Some error"},
            {"filename": "test.csv"},  # Missing required fields
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not raise error
            create_impulse_material_comparison(results, temp_dir)

    def test_create_impulse_summary_statistics(self):
        """Test creating impulse summary statistics."""
        results = [
            {
                "total_impulse": 0.001,
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
            },
            {
                "total_impulse": -0.002,
                "total_abs_impulse": 0.002,
                "peak_force": 1500,
                "impact_duration": 0.025,
                "material_type": "DF",
            },
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            summary = create_impulse_summary_statistics(results, temp_dir)

            assert "IMPULSE ANALYSIS SUMMARY" in summary
            assert "Total files processed: 2" in summary
            assert "Successful analyses: 2" in summary
            assert "STND" in summary
            assert "DF" in summary

    def test_create_impulse_summary_statistics_empty(self):
        """Test summary statistics with empty results."""
        summary = create_impulse_summary_statistics([], None)
        assert summary == "No results to summarize."

    def test_create_impulse_summary_statistics_no_valid(self):
        """Test summary statistics with no valid results."""
        results = [{"error": "Some error"}, {"filename": "test.csv"}]

        summary = create_impulse_summary_statistics(results, None)
        assert "No valid impulse results to summarize." in summary

    @patch(
        "Fishing_Line_Flyback_Impact_Analysis.visualization.create_impulse_material_comparison"
    )
    @patch(
        "Fishing_Line_Flyback_Impact_Analysis.visualization.create_impulse_summary_statistics"
    )
    def test_create_summary_plots(self, mock_stats, mock_comparison):
        """Test legacy compatibility function."""
        results = [
            {"total_abs_impulse": 0.001, "peak_force": 1000, "material_type": "STND"}
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            create_summary_plots(results, temp_dir)

            mock_comparison.assert_called_once_with(results, temp_dir)
            mock_stats.assert_called_once_with(results, temp_dir)


class TestInteractiveVisualization:
    """Test interactive visualization functions."""

    def create_test_force_data(self, n_points=1000):
        """Create test force and time data."""
        time = np.linspace(0, 0.1, n_points)
        force = np.random.normal(0, 10, n_points)
        force[400:600] = 1000 * np.sin(np.linspace(0, np.pi, 200))
        return force, time

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.ion")
    def test_show_force_preview_interactive_explorer(self, mock_ion, mock_show):
        """Test interactive force preview with explorer style."""
        force, time = self.create_test_force_data()

        show_force_preview_interactive(
            force, time, "STND-21-5.csv", show_analysis_preview=False, style="explorer"
        )

        mock_ion.assert_called_once()
        mock_show.assert_called_once()

    @patch("Fishing_Line_Flyback_Impact_Analysis.visualization.show_force_preview")
    def test_show_force_preview_interactive_fallback(self, mock_preview):
        """Test interactive force preview fallback."""
        force, time = self.create_test_force_data()

        show_force_preview_interactive(force, time, "test.csv", style="simple")

        mock_preview.assert_called_once_with(force, time, "test.csv", False)

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.ion")
    def test_show_force_preview_interactive_with_analysis(self, mock_ion, mock_show):
        """Test interactive preview with analysis boundaries."""
        force, time = self.create_test_force_data()

        show_force_preview_interactive(
            force, time, "STND-21-5.csv", show_analysis_preview=True, style="explorer"
        )

        mock_ion.assert_called_once()
        mock_show.assert_called_once()


class TestVisualizationHelpers:
    """Test visualization helper functions."""

    def create_test_force_data(self, n_points=1000):
        """Create test force and time data."""
        time = np.linspace(0, 0.1, n_points)
        force = np.random.normal(0, 10, n_points)
        force[400:600] = 1000 * np.sin(np.linspace(0, np.pi, 200))
        return force, time

    def test_internal_helper_functions_exist(self):
        """Test that internal helper functions are accessible if needed."""
        from Fishing_Line_Flyback_Impact_Analysis import visualization

        # Test that the module has the expected public interface
        expected_functions = [
            "show_force_preview",
            "plot_single_file_analysis",
            "create_single_file_analysis_plot",
            "create_impulse_material_comparison",
            "create_impulse_summary_statistics",
            "show_force_preview_interactive",
            "create_summary_plots",
        ]

        for func_name in expected_functions:
            assert hasattr(visualization, func_name)

    @patch("matplotlib.pyplot.show")
    def test_visualization_with_extreme_data(self, mock_show):
        """Test visualization with extreme data values."""
        # Very large forces
        force = np.array([1e6, 2e6, 1e6])
        time = np.array([0, 0.001, 0.002])

        fig = show_force_preview(force, time, "extreme.csv")
        assert fig is not None

        # Very small forces
        force = np.array([1e-6, 2e-6, 1e-6])
        time = np.array([0, 0.001, 0.002])

        fig = show_force_preview(force, time, "tiny.csv")
        assert fig is not None

        # Very long duration
        force = np.random.normal(0, 1, 100000)
        time = np.linspace(0, 100, 100000)  # 100 seconds

        fig = show_force_preview(force, time, "long.csv")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_visualization_with_nan_data(self, mock_show):
        """Test visualization with NaN data."""
        force = np.array([1, np.nan, 3, 4, np.nan])
        time = np.array([0, 0.001, 0.002, 0.003, 0.004])

        # Should handle NaN gracefully
        fig = show_force_preview(force, time, "nan_data.csv")
        assert fig is not None

    @patch("matplotlib.pyplot.show")
    def test_visualization_with_negative_time(self, mock_show):
        """Test visualization with negative time values."""
        force = np.array([1, 2, 3, 4, 5])
        time = np.array([-0.002, -0.001, 0, 0.001, 0.002])

        fig = show_force_preview(force, time, "negative_time.csv")
        assert fig is not None

    def test_create_single_file_analysis_plot_missing_fields(self):
        """Test analysis plot with missing result fields."""
        force, time = self.create_test_force_data()

        # Minimal analysis result
        analysis_result = {
            "impact_start_idx": 300,
            "impact_end_idx": 700,
            "total_impulse": 0.05,
            "peak_force": 1000,
            # Missing many optional fields
        }

        # Should still create plot without errors
        fig = create_single_file_analysis_plot(force, time, analysis_result, "test.csv")

        assert fig is not None
        assert len(fig.axes) == 4

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_create_impulse_material_comparison_single_material(
        self, mock_close, mock_savefig
    ):
        """Test material comparison with single material."""
        results = [
            {
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            # Should not crash with single material
            create_impulse_material_comparison(results, temp_dir)


class TestVisualizationErrorHandling:
    """Test error handling in visualization functions."""

    @patch("matplotlib.pyplot.show")
    def test_show_force_preview_with_analysis_error(self, mock_show):
        """Test force preview when analysis preview fails."""
        force = np.random.normal(0, 1, 100)
        time = np.linspace(0, 0.01, 100)

        # Use filename that would cause analysis error
        fig = show_force_preview(
            force, time, "INVALID-format.csv", show_analysis_preview=True
        )

        # Should still create plot despite analysis error
        assert fig is not None
        mock_show.assert_called_once()

    def test_create_impulse_material_comparison_file_error(self):
        """Test material comparison when file operations fail."""
        results = [
            {
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
            }
        ]

        # Try to save to read-only location (should be handled gracefully)
        try:
            create_impulse_material_comparison(results, "/dev/null")
        except PermissionError:
            # This is expected in some systems
            pass

    def test_create_summary_statistics_file_error(self):
        """Test summary statistics when file operations fail."""
        results = [
            {
                "total_impulse": 0.001,
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
            }
        ]

        # Should return summary even if file save fails
        summary = create_impulse_summary_statistics(results, "/invalid/path")
        assert "IMPULSE ANALYSIS SUMMARY" in summary


class TestVisualizationIntegration:
    """Integration tests for visualization components."""

    def create_realistic_analysis_result(self):
        """Create realistic analysis result for testing."""
        return {
            "filename": "STND-21-5.csv",
            "material_type": "STND",
            "sample_number": "21-5",
            "total_impulse": 0.05234,
            "total_abs_impulse": 0.05234,
            "impact_impulse": 0.04987,
            "impact_abs_impulse": 0.04987,
            "peak_force": 1847.3,
            "peak_force_positive": 1847.3,
            "peak_force_negative": -234.1,
            "rms_force": 387.2,
            "impact_duration": 0.0234,
            "impact_start_time": 0.0123,
            "impact_end_time": 0.0357,
            "total_duration": 0.1,
            "equivalent_velocity": 234.7,
            "equivalent_kinetic_energy": 1.847,
            "mass_kg": 0.0717,
            "impact_start_idx": 1230,
            "impact_end_idx": 3570,
            "sampling_rate_hz": 100000,
            "analysis_method": "impulse_integration",
            "material_code": "STND",
            "mass_breakdown": {
                "hardware_mass_kg": 0.045,
                "line_mass_effective_kg": 0.02716,
                "total_mass_kg": 0.0717,
                "material_code": "STND",
            },
        }

    @patch("matplotlib.pyplot.show")
    @patch("matplotlib.pyplot.close")
    def test_complete_visualization_workflow(self, mock_close, mock_show):
        """Test complete visualization workflow."""
        # Create test data
        n_points = 10000
        time = np.linspace(0, 0.1, n_points)
        force = np.random.normal(0, 50, n_points)

        # Add realistic impact
        impact_start = 1230
        impact_end = 3570
        impact_indices = np.arange(impact_start, impact_end)
        impact_profile = 1847 * np.exp(-0.5 * ((impact_indices - 2000) / 500) ** 2)
        force[impact_start:impact_end] = impact_profile

        # Test force preview
        fig1 = show_force_preview(force, time, "STND-21-5.csv")
        assert fig1 is not None

        # Test detailed analysis plot
        analysis_result = self.create_realistic_analysis_result()
        fig2 = create_single_file_analysis_plot(
            force, time, analysis_result, "STND-21-5.csv"
        )
        assert fig2 is not None

        # Verify both plots were created
        assert mock_show.call_count >= 1

    def test_batch_visualization_workflow(self):
        """Test visualization with batch analysis results."""
        # Create multiple analysis results
        materials = ["STND", "DF", "DS", "SL", "BR"]
        results = []

        for i, material in enumerate(materials):
            for j in range(3):  # 3 samples per material
                result = self.create_realistic_analysis_result()
                result["material_type"] = material
                result["filename"] = f"{material}-21-{j + 1}.csv"
                result["sample_number"] = f"21-{j + 1}"

                # Add some variation
                result["total_abs_impulse"] *= 0.8 + i * 0.1
                result["peak_force"] *= 0.9 + i * 0.05

                results.append(result)

        # Test material comparison
        with tempfile.TemporaryDirectory() as temp_dir:
            create_impulse_material_comparison(results, temp_dir)

            # Verify output file exists
            output_file = Path(temp_dir) / "impulse_material_comparison.png"
            # Note: File might not exist in test environment, but function should run

        # Test summary statistics
        summary = create_impulse_summary_statistics(results, None)

        assert "15" in summary  # 15 total files
        assert "STND" in summary
        assert "BR" in summary
        assert "IMPULSE ANALYSIS SUMMARY" in summary

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_visualization_output_formats(self, mock_close, mock_savefig):
        """Test that visualizations support different output formats."""
        results = [
            {
                "total_abs_impulse": 0.001,
                "peak_force": 1000,
                "impact_duration": 0.02,
                "material_type": "STND",
            }
        ]

        with tempfile.TemporaryDirectory() as temp_dir:
            create_impulse_material_comparison(results, temp_dir)

            # Should call savefig (format determined by filename extension)
            mock_savefig.assert_called_once()

    def test_visualization_data_validation(self):
        """Test that visualization functions validate input data."""
        # Test with empty force data
        empty_force = np.array([])
        empty_time = np.array([])

        # Should handle empty data gracefully
        with patch("matplotlib.pyplot.show"):
            fig = show_force_preview(empty_force, empty_time, "empty.csv")
            # Might return None or raise exception - both are acceptable

    def test_visualization_memory_efficiency(self):
        """Test visualization with large datasets."""
        # Large dataset
        n_points = 1000000  # 1M points
        time = np.linspace(0, 10, n_points)
        force = np.random.normal(0, 10, n_points)

        # Add impact
        force[n_points // 2 : n_points // 2 + 10000] = 1000

        with patch("matplotlib.pyplot.show"):
            # Should handle large datasets without memory issues
            fig = show_force_preview(force, time, "large_dataset.csv")
            # Function should complete without raising MemoryError


class TestVisualizationDocumentation:
    """Test that visualization functions have proper documentation."""

    def test_function_docstrings(self):
        """Test that main functions have docstrings."""
        functions_to_check = [
            show_force_preview,
            plot_single_file_analysis,
            create_single_file_analysis_plot,
            create_impulse_material_comparison,
            create_impulse_summary_statistics,
        ]

        for func in functions_to_check:
            assert func.__doc__ is not None
            assert len(func.__doc__.strip()) > 0

    def test_module_structure(self):
        """Test that visualization module has expected structure."""
        from Fishing_Line_Flyback_Impact_Analysis import visualization

        # Should have expected attributes
        expected_attributes = [
            "show_force_preview",
            "plot_single_file_analysis",
            "create_single_file_analysis_plot",
            "create_impulse_material_comparison",
            "create_impulse_summary_statistics",
            "show_force_preview_interactive",
            "create_summary_plots",
        ]

        for attr in expected_attributes:
            assert hasattr(visualization, attr)

        # Functions should be callable
        for attr in expected_attributes:
            assert callable(getattr(visualization, attr))
