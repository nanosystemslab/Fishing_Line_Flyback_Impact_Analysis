"""Test cases for the visualization module."""

import tempfile
import types
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Fishing_Line_Flyback_Impact_Analysis.visualization import ImpactVisualizer


class TestImpactVisualizer:
    """Test cases for ImpactVisualizer class."""

    def setup_method(self) -> None:
        """Set up test fixtures before each test method."""
        self.temp_dir = tempfile.mkdtemp()
        self.visualizer = ImpactVisualizer(output_dir=self.temp_dir)

        # Create sample test data with metadata
        self.sample_df = pd.DataFrame(
            {
                "time": np.linspace(0, 1, 100),
                "S1": np.sin(np.linspace(0, 2 * np.pi, 100)) * 100,
                "S2": np.sin(np.linspace(0, 2 * np.pi, 100)) * 110,
                "S3": np.sin(np.linspace(0, 2 * np.pi, 100)) * 90,
                "S4": np.sin(np.linspace(0, 2 * np.pi, 100)) * 120,
                "SUM": np.sin(np.linspace(0, 2 * np.pi, 100)) * 400,
                "Max": np.sin(np.linspace(0, 2 * np.pi, 100)) * 120,
            }
        )

        # Add metadata
        self.sample_df.meta = types.SimpleNamespace()
        self.sample_df.meta.config = "STND"
        self.sample_df.meta.diam = "21"
        self.sample_df.meta.run_num = "1"
        self.sample_df.meta.fname = "STND-21-1.csv"

    def teardown_method(self) -> None:
        """Clean up after each test method."""
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)
        plt.close("all")  # Close all matplotlib figures

    def test_init(self) -> None:
        """Test ImpactVisualizer initialization."""
        visualizer = ImpactVisualizer()
        assert hasattr(visualizer, "output_dir")
        assert hasattr(visualizer, "plot_params")
        assert hasattr(visualizer, "log")

    def test_init_with_custom_output_dir(self) -> None:
        """Test initialization with custom output directory."""
        custom_dir = self.temp_dir + "/custom"
        visualizer = ImpactVisualizer(output_dir=custom_dir)
        assert str(visualizer.output_dir) == custom_dir
        assert Path(custom_dir).exists()

    def test_plot_params_configuration(self) -> None:
        """Test plot parameters are properly configured."""
        required_params = [
            "time",
            "S1",
            "S2",
            "S3",
            "S4",
            "SUM",
            "Max",
            "All",
            "J",
            "F",
        ]

        for param in required_params:
            assert param in self.visualizer.plot_params
            assert "label" in self.visualizer.plot_params[param]
            assert "unit" in self.visualizer.plot_params[param]

    def test_plot_time_series_single_sensor(self) -> None:
        """Test time series plotting with single sensor."""
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_time_series(self.sample_df, param_y="SUM")

        # Check if plot file was created
        plot_files = list(Path(self.temp_dir).glob("plot-time_vs_SUM--*.png"))
        assert len(plot_files) > 0

    def test_plot_time_series_all_sensors(self) -> None:
        """Test time series plotting with all sensors."""
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_time_series(
                self.sample_df, param_y="All", show_all_sensors=True
            )

        plot_files = list(Path(self.temp_dir).glob("plot-time_vs_All--*.png"))
        assert len(plot_files) > 0

    def test_plot_time_series_different_params(self) -> None:
        """Test time series plotting with different parameters."""
        test_params = ["S1", "S2", "S3", "S4", "Max"]

        for param in test_params:
            with patch("matplotlib.pyplot.show"):
                self.visualizer.plot_time_series(self.sample_df, param_y=param)

            plot_files = list(Path(self.temp_dir).glob(f"plot-time_vs_{param}--*.png"))
            assert len(plot_files) > 0

    def test_plot_box_plots(self) -> None:
        """Test box plot generation."""
        # Create sample results DataFrame
        results_df = pd.DataFrame(
            {
                "test_type": ["STND", "DF", "BR"] * 3,
                "F": [1200, 800, 400, 1250, 850, 420, 1180, 780, 380],
                "J": [2.5, 1.2, 0.5, 2.6, 1.3, 0.6, 2.4, 1.1, 0.4],
            }
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_box_plots(results_df, param="F")
            self.visualizer.plot_box_plots(results_df, param="J")

        # Check if plots were created
        f_plots = list(Path(self.temp_dir).glob("plot-box-F.png"))
        j_plots = list(Path(self.temp_dir).glob("plot-box-J.png"))
        assert len(f_plots) > 0
        assert len(j_plots) > 0

    def test_plot_box_plots_with_unit_conversion(self) -> None:
        """Test box plots with unit conversion."""
        results_df = pd.DataFrame(
            {"test_type": ["STND", "DF"], "F": [1000, 800], "J": [2.0, 1.5]}
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_box_plots(results_df, param="F", convert_units=True)

        # Should create plot with converted units
        plot_files = list(Path(self.temp_dir).glob("plot-box-F.png"))
        assert len(plot_files) > 0

    def test_plot_violin_plots(self) -> None:
        """Test violin plot generation."""
        results_df = pd.DataFrame(
            {
                "test_type": ["STND", "DF", "BR"] * 2,
                "F": [1200, 800, 400, 1250, 850, 420],
                "J": [2.5, 1.2, 0.5, 2.6, 1.3, 0.6],
            }
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_violin_plots(results_df, param="F")
            self.visualizer.plot_violin_plots(results_df, param="J")

        # Check if plots were created
        f_plots = list(Path(self.temp_dir).glob("plot-violin-F.png"))
        j_plots = list(Path(self.temp_dir).glob("plot-violin-J.png"))
        assert len(f_plots) > 0
        assert len(j_plots) > 0

    def test_plot_dual_box_plots_png(self) -> None:
        """Test dual box plots in PNG format."""
        results_df = pd.DataFrame(
            {
                "test_type": ["STND", "DF", "BR"] * 2,
                "F": [1200, 800, 400, 1250, 850, 420],
                "J": [2.5, 1.2, 0.5, 2.6, 1.3, 0.6],
            }
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_dual_box_plots(results_df, save_format="png")

        plot_files = list(Path(self.temp_dir).glob("plot-box-J-F.png"))
        assert len(plot_files) > 0

    def test_plot_dual_box_plots_svg(self) -> None:
        """Test dual box plots in SVG format."""
        results_df = pd.DataFrame(
            {"test_type": ["STND", "DF"], "F": [1200, 800], "J": [2.5, 1.2]}
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_dual_box_plots(results_df, save_format="svg")

        plot_files = list(Path(self.temp_dir).glob("plot-box-J-F.svg"))
        assert len(plot_files) > 0

    def test_create_summary_plots(self) -> None:
        """Test creation of all summary plots."""
        results_df = pd.DataFrame(
            {
                "test_type": ["STND", "DF", "BR"] * 3,
                "F": [1200, 800, 400, 1250, 850, 420, 1180, 780, 380],
                "J": [2.5, 1.2, 0.5, 2.6, 1.3, 0.6, 2.4, 1.1, 0.4],
            }
        )

        with patch("matplotlib.pyplot.show"):
            self.visualizer.create_summary_plots(results_df)

        # Check if multiple plot types were created
        all_plots = list(Path(self.temp_dir).glob("plot-*.png"))
        assert len(all_plots) >= 4  # box, violin, dual plots

    def test_get_mass_for_config(self) -> None:
        """Test mass retrieval for different configurations."""
        test_cases = [
            ("STND", 49),
            ("DF", 62),
            ("DS", 75),
            ("SL", 65),
            ("BR", 45),
            ("UNKNOWN", 45),  # default
        ]

        for config, expected_mass in test_cases:
            mass = self.visualizer._get_mass_for_config(config)
            assert mass == expected_mass

    def test_plot_output_data(self) -> None:
        """Test plotting output data from CSV."""
        # Create sample output CSV
        output_csv = Path(self.temp_dir) / "output.csv"
        output_data = pd.DataFrame(
            {"diameter": [21, 23, 25], "kinetic_energy": [0.1, 0.2, 0.3]}
        )
        output_data.to_csv(output_csv, header=False, index=False)

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_output_data(str(output_csv), x_param="D", y_param="KE")

        plot_files = list(Path(self.temp_dir).glob("output-D_vs_KE.png"))
        assert len(plot_files) > 0

    def test_plot_output_data_empty_file(self) -> None:
        """Test plotting with empty output file."""
        empty_csv = Path(self.temp_dir) / "empty.csv"
        empty_data = pd.DataFrame()
        empty_data.to_csv(empty_csv, header=False, index=False)

        # Should handle empty file gracefully
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_output_data(str(empty_csv))

    def test_plot_output_data_single_column(self) -> None:
        """Test plotting with insufficient columns."""
        single_col_csv = Path(self.temp_dir) / "single.csv"
        single_data = pd.DataFrame({"col1": [1, 2, 3]})
        single_data.to_csv(single_col_csv, header=False, index=False)

        # Should handle insufficient columns gracefully
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_output_data(str(single_col_csv))

    def test_time_series_time_adjustment(self) -> None:
        """Test time series with time parameter adjustment."""
        # Test data with non-zero start time
        time_data = self.sample_df.copy()
        time_data["time"] = time_data["time"] + 10.0  # Start at 10 seconds

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_time_series(time_data, param_x="time", param_y="SUM")

        # Should create plot with adjusted time (starting from 0)
        plot_files = list(Path(self.temp_dir).glob("plot-time_vs_SUM--*.png"))
        assert len(plot_files) > 0

    def test_plot_with_missing_sensors(self) -> None:
        """Test plotting when some sensors are missing."""
        incomplete_df = self.sample_df[["time", "S1", "S2", "SUM"]].copy()
        incomplete_df.meta = self.sample_df.meta

        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_time_series(
                incomplete_df, param_y="All", show_all_sensors=True
            )

        # Should handle missing sensors gracefully
        plot_files = list(Path(self.temp_dir).glob("plot-time_vs_All--*.png"))
        assert len(plot_files) > 0

    def test_plot_params_unknown_parameter(self) -> None:
        """Test plotting with unknown parameter names."""
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_output_data(
                str(Path(self.temp_dir) / "test.csv"),
                x_param="UNKNOWN_X",
                y_param="UNKNOWN_Y",
            )
        # Should use default parameter labels

    def test_figure_settings(self) -> None:
        """Test that figure settings are properly configured."""
        assert hasattr(self.visualizer, "figsize")
        assert hasattr(self.visualizer, "figdpi")
        assert hasattr(self.visualizer, "hwratio")

        assert isinstance(self.visualizer.figsize, (int, float))
        assert isinstance(self.visualizer.figdpi, (int, float))
        assert isinstance(self.visualizer.hwratio, (int, float))

    def test_seaborn_style_application(self) -> None:
        """Test that seaborn styles are properly applied."""
        # This is mostly a smoke test to ensure no errors occur
        with patch("matplotlib.pyplot.show"):
            self.visualizer.plot_time_series(self.sample_df)

        # If we get here without exceptions, seaborn styling
